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
from transformers import AutoModel, AutoTokenizer

ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")

MODEL_OPTIONS = {
    "Granite Embedding Small English R2": "ibm-granite/granite-embedding-small-english-r2",
    "Granite Embedding English R2": "ibm-granite/granite-embedding-english-r2",
}

DEVICE_MAP = {
    "mps": AcceleratorDevice.MPS,
    "cuda": AcceleratorDevice.CUDA,
    "cpu": AcceleratorDevice.CPU,
}


def get_device():
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(model_path: str, device: str):
    """Load a single embedding model on demand."""
    return {
        "model": AutoModel.from_pretrained(model_path, device_map=device),
        "tokenizer": AutoTokenizer.from_pretrained(model_path),
    }


def convert(source: str, doc_converter: DocumentConverter) -> str:
    """Convert PDF to markdown."""
    result = doc_converter.convert(source=source, max_num_pages=100, max_file_size=20971520)
    return result.document.export_to_markdown()


def embed(text: str, model, tokenizer, device: str) -> list[float]:
    """Generate normalized embedding vector from text."""
    tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)
        embedding = output[0][:, 0]

    return torch.nn.functional.normalize(embedding, dim=1).cpu().numpy().tolist()[0]


def build_pipeline_options(
    tableformer_mode: str,
    use_structure_prediction: bool,
    code_understanding: bool,
    formula_understanding: bool,
    picture_classification: bool,
    accelerator_device: AcceleratorDevice,
) -> PdfPipelineOptions:
    """Build PDF pipeline options from UI settings."""
    options = PdfPipelineOptions(artifacts_path=ARTIFACTS_PATH, do_table_structure=True)

    options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if tableformer_mode == "Accurate" else TableFormerMode.FAST
    )
    options.table_structure_options.do_cell_matching = not use_structure_prediction
    options.do_code_enrichment = code_understanding
    options.do_formula_enrichment = formula_understanding

    if picture_classification:
        options.generate_picture_images = True
        options.images_scale = 2
        options.do_picture_classification = True

    options.accelerator_options = AcceleratorOptions(num_threads=8, device=accelerator_device)
    return options


# Initialize session state
if "embedding_result" not in st.session_state:
    st.session_state.embedding_result = None

# UI
st.title("Embedding Pipeline")
st.write("Generate vector embeddings from text with IBM Granite Embedding R2 models.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

device = get_device()
accelerator_device = DEVICE_MAP[device]

st.subheader("Embedding Models")
selected_model_name = st.radio(
    "Select model",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    help="Select a model for generating vector embeddings from text.",
)

st.subheader("PDF Table Extraction")
use_structure_prediction = st.toggle(
    "Use text cells from structure prediction",
    value=False,
    help="Uses text cells predicted from the table structure model instead of mapping back to PDF cells.",
)

tableformer_mode = st.radio(
    "TableFormer Mode",
    options=["Accurate", "Fast"],
    index=0,
    help="Accurate mode provides better quality. Fast mode is faster but less accurate.",
)

st.subheader("Enrichment")
code_understanding = st.toggle("Code understanding", value=False, help="Advanced parsing for code blocks.")
formula_understanding = st.toggle("Formula understanding", value=False, help="Extracts LaTeX from equations.")
picture_classification = st.toggle("Picture classification", value=False, help="Classifies pictures in the document.")

if st.button("Embed", type="primary"):
    if uploaded_file is None:
        st.warning("Upload a PDF file.")
    else:
        tmp_file_path = None
        try:
            # Load only the selected model
            with st.spinner(f"Loading model on {device.upper()}..."):
                model_data = load_model(MODEL_OPTIONS[selected_model_name], device)

            # Build converter with pipeline options
            pipeline_options = build_pipeline_options(
                tableformer_mode,
                use_structure_prediction,
                code_understanding,
                formula_understanding,
                picture_classification,
                accelerator_device,
            )
            doc_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            # Convert PDF to markdown
            with st.spinner("Converting document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                doc_markdown = convert(tmp_file_path, doc_converter)

            # Generate embedding
            with st.spinner("Generating embedding..."):
                start = time.perf_counter()
                embedding_vector = embed(
                    doc_markdown, model_data["model"], model_data["tokenizer"], device
                )
                duration_ms = (time.perf_counter() - start) * 1000

            # Store results in session state
            st.session_state.embedding_result = {
                "model_name": selected_model_name,
                "model_path": MODEL_OPTIONS[selected_model_name],
                "embedding_size": len(embedding_vector),
                "duration_ms": round(duration_ms, 2),
                "embedding": embedding_vector,
                "filename": uploaded_file.name,
            }

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

# Display results from session state
if st.session_state.embedding_result:
    result = st.session_state.embedding_result
    st.success("Done.")

    st.subheader("Metrics")
    st.metric("Model", result["model_name"])
    col1, col2 = st.columns(2)
    col1.metric("Embedding Size", result["embedding_size"])
    col2.metric("Duration (ms)", f"{result['duration_ms']:.2f}")

    embedding_data = {
        "model": result["model_path"],
        "embedding_size": result["embedding_size"],
        "duration_ms": result["duration_ms"],
        "embedding": result["embedding"],
    }
    st.download_button(
        label="Download JSON",
        data=json.dumps(embedding_data, indent=2),
        file_name=f"{result['filename']}_embedding.json",
        mime="application/json",
    )
