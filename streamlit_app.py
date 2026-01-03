import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from transformers import AutoModel, AutoTokenizer

# Docling models can be prefetched for offline use
download_models()

artifacts_path = str(Path.home() / '.cache' / 'docling' / 'models')

MODEL_OPTIONS = {
    "Granite Embedding Small English R2": "ibm-granite/granite-embedding-small-english-r2",
    "Granite Embedding English R2": "ibm-granite/granite-embedding-english-r2"
}

def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_accelerator_device(device):
    """Map the torch device to docling AcceleratorDevice."""
    if device == "mps":
        return AcceleratorDevice.MPS
    elif device == "cuda":
        return AcceleratorDevice.CUDA
    else:
        return AcceleratorDevice.CPU

@st.cache_resource
def load_embedding_models(device):
    """Cache embedding models at application startup."""
    models = {}
    for display_name, model_path in MODEL_OPTIONS.items():
        model = AutoModel.from_pretrained(model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        models[display_name] = {
            "model": model,
            "tokenizer": tokenizer
        }
    return models

def convert(source, doc_converter):
    """Convert a source file to a Docling document and export to Markdown."""
    result = doc_converter.convert(
        source=source,
        max_num_pages=100,
        max_file_size=20971520
    )
    doc = result.document
    doc_markdown = doc.export_to_markdown()
    return doc_markdown

def embed(doc_markdown, model, tokenizer, device):
    """Generate a vector embedding from input text using a transformers model."""
    tokenized_input = tokenizer([doc_markdown], padding=True, truncation=True, return_tensors='pt')
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    
    with torch.no_grad():
        model_output = model(**tokenized_input)
        embedding = model_output[0][:, 0]
    
    embedding = torch.nn.functional.normalize(embedding, dim=1)
    
    return embedding.cpu().numpy().tolist()[0]

st.title("Embedding Pipeline")
st.write("Generate vector embeddings from text with IBM Granite Embedding R2 models.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()
accelerator_device = get_accelerator_device(device)

with st.spinner(f"Loading models on {device.upper()}..."):
    embedding_models = load_embedding_models(device)

st.subheader("Embedding Models")
selected_model_name = st.radio(
    "Select model",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    help="Select a model for generating vector embeddings from text."
)
selected_model_path = MODEL_OPTIONS[selected_model_name]

st.subheader("PDF Table Extraction")
use_structure_prediction = st.toggle(
    "Use text cells from structure prediction",
    value=False,
    help="Uses text cells predicted from the table structure model instead of mapping back to PDF cells. This can improve output quality if multiple columns in tables are erroneously merged."
)

tableformer_mode = st.radio(
    "TableFormer Mode",
    options=["Accurate", "Fast"],
    index=0,
    help="Accurate mode provides better quality with difficult table structures. Fast mode is faster but less accurate."
)

st.subheader("Enrichment")
code_understanding = st.toggle(
    "Code understanding",
    value=False,
    help="Uses advanced parsing for code blocks found in the document."
)

formula_understanding = st.toggle(
    "Formula understanding",
    value=False,
    help="Analyzes equation formulas in documents and extracts their LaTeX representation."
)

picture_classification = st.toggle(
    "Picture classification",
    value=False,
    help="Classifies pictures in the document (charts, diagrams, logos, signatures) using the DocumentFigureClassifier model."
)

if st.button("Embed", type="primary"):
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_table_structure=True
    )
    
    if use_structure_prediction:
        pipeline_options.table_structure_options.do_cell_matching = False
    
    if tableformer_mode == "Accurate":
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    else:
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    
    if code_understanding:
        pipeline_options.do_code_enrichment = True
    
    if formula_understanding:
        pipeline_options.do_formula_enrichment = True
    
    if picture_classification:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2
        pipeline_options.do_picture_classification = True
    
    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=accelerator_device
    )
    pipeline_options.accelerator_options = accelerator_options
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if uploaded_file is not None:
        try:
            model = embedding_models[selected_model_name]["model"]
            tokenizer = embedding_models[selected_model_name]["tokenizer"]
            
            with st.spinner("Converting document..."):
                # Save uploaded file temporarily for Docling to process                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                doc_markdown = convert(tmp_file_path, doc_converter)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            with st.spinner("Generating embedding..."):
                start_time = time.time_ns()
                embedding_vector = embed(doc_markdown, model, tokenizer, device)
                end_time = time.time_ns()
                total_duration_ns = end_time - start_time
            
            st.success("Done.")
            
            st.subheader("Metrics")
            
            st.metric("Model", selected_model_name)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedding Size", len(embedding_vector))
            with col2:
                st.metric("Total Duration (nanoseconds)", total_duration_ns)
            
            # Prepare JSON for download
            embedding_data = {
                "model": selected_model_path,
                "embedding_size": len(embedding_vector),
                "total_duration_ns": total_duration_ns,
                "embedding": embedding_vector
            }
            
            json_str = json.dumps(embedding_data, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{uploaded_file.name}_embedding.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Syntax error: {str(e)}")
    else:
        st.warning("Upload a PDF file.")
