import os
from src.mmore.process.processors.docx_processor import DOCXProcessor
from src.mmore.process.processors.eml_processor import EMLProcessor
from src.mmore.process.processors.md_processor import MarkdownProcessor
from src.mmore.process.processors.media_processor import MediaProcessor
from src.mmore.process.processors.pdf_processor import PDFProcessor
from src.mmore.process.processors.pptx_processor import PPTXProcessor
from src.mmore.process.processors.spreadsheet_processor import SpreadsheetProcessor
from src.mmore.process.processors.processor import ProcessorConfig
from src.mmore.type import FileDescriptor

SAMPLES_DIR = "tests/samples/"

def get_file_descriptor(file_path):
    return FileDescriptor.from_filename(file_path)

# ------------------ DOCX Processor Tests ------------------
def test_docx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = DOCXProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

def test_docx_process_fast():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = DOCXProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

#  ------------------ EML Processor Tests ------------------
def test_eml_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = EMLProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

# ------------------ Markdown Processor Tests ------------------
def test_md_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = MarkdownProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

# ------------------ Media Processor Tests ------------------
def test_media_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"normal_model": "model-name"})
    processor = MediaProcessor([get_file_descriptor(sample_file)], config=config)
    processor.load_models()
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

def test_media_process_fast():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"fast_model": "model-name"})
    processor = MediaProcessor([get_file_descriptor(sample_file)], config=config)
    processor.load_models(fast_mode=True)
    result = processor.process_fast_implementation(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

# ------------------ PDF Processor Tests ------------------
def test_pptx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = PPTXProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

# ------------------ Spreadsheet Processor Tests ------------------
def test_spreadsheet_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig()
    processor = SpreadsheetProcessor([get_file_descriptor(sample_file)], config=config)
    result = processor.process_one_file(sample_file)
    assert result["text"], "Text should not be empty"
    assert isinstance(result["modalities"], list), "Modalities should be a list"

## TODO: Write tests for pdf_processor.py
## ------------------ PDF Processor Tests ------------------
# def test_pdf_process_standard():
#     pass

# def test_pdf_process_fast():
#     pass 