# Import packages
import tempfile
import os
import docling
import ollama
import streamlit as st
from pathlib import Path
from docling.utils.model_downloader import download_models
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# Docling models can be prefetched for offline use
docling.utils.model_downloader.download_models()

artifacts_path = str(Path.home() / '.cache' / 'docling' / 'models')

embedding_model = 'embeddinggemma'

# Function to convert source file to a Docling document AND export to markdown
def convert(source):
    result = doc_converter.convert(
        source=source,
        max_num_pages=100,
        max_file_size=20971520
    )
    doc = result.document
    doc_markdown = doc.export_to_markdown()
    return doc_markdown

# Function to generate embedding using Ollama model
def embed(doc_markdown):
    doc_markdown_embedding = ollama.embed(
        model=embedding_model,
        input=doc_markdown
    )
    return doc_markdown_embedding

st.title("Embedding Pipeline")
st.write("Generate embeddings using an Ollama model.")

# File uploader for PDF file
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# PDF table extraction options
st.subheader("PDF table extraction options")
use_structure_prediction = st.checkbox(
    "Use text cells from structure prediction",
    value=False,
    help="Uses text cells predicted from the table structure model instead of mapping back to PDF cells. This can improve output quality if multiple columns in tables are erroneously merged."
)

# TableFormer mode selector
tableformer_mode = st.selectbox(
    "TableFormer mode",
    options=["Accurate", "Fast"],
    index=0, # Default to "Accurate"
    help="Accurate mode provides better quality with difficult table structures. Fast mode is faster but less accurate."
)

# Enrichment features
st.subheader("Enrichment features")
code_understanding = st.checkbox(
    "Code understanding",
    value=False,
    help="Uses advanced parsing for code blocks found in the document."
)

formula_understanding = st.checkbox(
    "Formula understanding",
    value=False,
    help="Analyzes equation formulas in documents and extracts their LaTeX representation."
)

picture_classification = st.checkbox(
    "Picture classification",
    value=False,
    help="Classifies pictures in the document (charts, diagrams, logos, signatures) using the DocumentFigureClassifier model."
)

# Accelerator options
st.subheader("Accelerator options")
accelerator_mode = st.selectbox(
    "Accelerator mode",
    options=["AUTO", "CPU", "MPS", "CUDA"],
    index=0, # Default to "AUTO"
    help="Runs conversion with a specific accelerator configuration. CPU mode works everywhere. MPS is macOS only. CUDA requires a compatible GPU and CUDA enabled PyTorch build."
)

if st.button("Generate embedding", type="primary"):
    # Create pipeline options based on user selection
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_table_structure=True
    )

    # Set cell matching based on checkbox
    if use_structure_prediction:
        pipeline_options.table_structure_options.do_cell_matching = False

    # Set TableFormer mode based on selectbox
    if tableformer_mode == "Accurate":
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    else: # "Fast"
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    # Set code understanding enrichment based on checkbox
    if code_understanding:
        pipeline_options.do_code_enrichment = True

    # Set formula understanding enrichment based on checkbox
    if formula_understanding:
        pipeline_options.do_formula_enrichment = True

    # Set picture classification enrichment based on checkbox
    if picture_classification:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2
        pipeline_options.do_picture_classification = True

    # Set accelerator mode based on selectbox
    if accelerator_mode == "AUTO":
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO
        )
        pipeline_options.accelerator_options = accelerator_options
    elif accelerator_mode == "CPU":
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CPU
        )
        pipeline_options.accelerator_options = accelerator_options
    elif accelerator_mode == "MPS":
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.MPS
        )
        pipeline_options.accelerator_options = accelerator_options
    else: # CUDA
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA
        )
        pipeline_options.accelerator_options = accelerator_options

    # Create converter with current options
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if uploaded_file is not None:
        try:
            # Show progress
            with st.spinner("Converting PDF to markdown..."):
                # Save uploaded file temporarily for Docling to process                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Convert PDF to markdown
                doc_markdown = convert(tmp_file_path)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            with st.spinner("Generating embedding..."):
                # Generate embedding
                doc_markdown_embedding = embed(doc_markdown)
            
            st.success("Embedding generated successfully!")
            
            # Display metadata
            st.subheader("Embedding Metadata")
            
            embedding_vector = doc_markdown_embedding['embeddings'][0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", embedding_model)
                st.metric("Embedding Dimensions", len(embedding_vector))
            with col2:
                st.metric("Document", uploaded_file.name)
                st.metric("Markdown Length", f"{len(doc_markdown)} characters")
            
            # Show sample of embedding values
            with st.expander("View sample embedding values (first 10)"):
                st.json(embedding_vector[:10])
            
            # Prepare JSON for download
            import json
            embedding_data = {
                "model": embedding_model,
                "document_name": uploaded_file.name,
                "embedding_dimensions": len(embedding_vector),
                "markdown_length": len(doc_markdown),
                "embedding": embedding_vector,
                "markdown_text": doc_markdown
            }
            
            json_str = json.dumps(embedding_data, indent=2)
            
            # Download button
            st.download_button(
                label="Download Embedding as JSON",
                data=json_str,
                file_name=f"{uploaded_file.name}_embedding.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a PDF file first.")
