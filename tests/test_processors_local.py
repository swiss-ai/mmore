import os

from marker.output import MarkdownOutput

from mmore.process.processors.base import MediaProcessorConfig, ProcessorConfig
from mmore.process.processors.docx_processor import DOCXProcessor
from mmore.process.processors.eml_processor import EMLProcessor
from mmore.process.processors.md_processor import MarkdownProcessor
from mmore.process.processors.media_processor import MediaProcessor
from mmore.process.processors.pdf_processor import PDFProcessor
from mmore.process.processors.pptx_processor import PPTXProcessor
from mmore.process.processors.spreadsheet_processor import SpreadsheetProcessor
from mmore.process.processors.txt_processor import TextProcessor
from mmore.process.processors.url_processor import URLProcessor
from mmore.type import FileDescriptor

"""
If you get an import error when running with pytest, run:
    PYTHONPATH=$(pwd) pytest tests/test_processors_local.py
This is required because the project follows a "src" layout.
"""

SAMPLES_DIR = "examples/sample_data/"


def get_file_descriptor(file_path):
    return FileDescriptor.from_filename(file_path)


# ------------------ DOCX Processor Tests ------------------
def test_docx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = DOCXProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_docx_no_image_extraction():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    config = ProcessorConfig(output_path="tmp", extract_images=False)
    processor = DOCXProcessor(config=config)
    result = processor.process(sample_file)

    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    assert "<attachment>" not in combined_text, (
        "Attachment tag should not appear when image extraction is disabled."
    )
    assert len(result.modalities) == 0, (
        "Expected no images when image extraction is disabled."
    )


#  ------------------ EML Processor Tests ------------------
def test_eml_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = EMLProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_eml_headers_present():
    """Test that the processed EML file includes email headers."""
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = EMLProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    for header in ["From:", "To:", "Subject:", "Date:"]:
        assert header in combined_text, f"{header} not found in the processed text"


# ------------------ Markdown Processor Tests ------------------
def test_md_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = MarkdownProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_md_image_extraction():
    """Test that the processor correctly extracts images and inserts attachment tags."""
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp", extract_images=True)
    processor = MarkdownProcessor(config=config)
    result = processor.process(sample_file)
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    placeholder_count = combined_text.count("<attachment>")
    assert placeholder_count == 2, (
        f"Expected 2 attachment placeholders, found {placeholder_count}"
    )
    assert isinstance(result.modalities, list), "Modalities should be a list"
    assert len(result.modalities) == 2, (
        f"Expected 2 images in modalities, found {len(result.modalities)}"
    )


# ------------------ Media Processor Tests ------------------
def test_media_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = MediaProcessorConfig(output_path="tmp", normal_model="model-name")
    processor = MediaProcessor(config=config)
    # Override _load_pipelines to bypass actual model loading
    processor._load_pipelines = lambda fast_mode=False: setattr(
        processor, "pipelines", [lambda x: {"text": "dummy transcription"}]
    )
    processor._load_pipelines()
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_media_process_batch():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    files = [sample_file, sample_file, sample_file]
    config = MediaProcessorConfig(output_path="tmp", normal_model="model-name")
    processor = MediaProcessor(config=config)
    processor._load_pipelines = lambda fast_mode=False: setattr(
        processor,
        "pipelines",
        [lambda x: {"text": "dummy transcription"} for _ in processor.devices],
    )
    processor._load_pipelines()
    results = processor.process_batch(files, fast_mode=False, num_workers=1)
    assert len(results) == len(files), (
        "Number of results should match number of files processed."
    )
    for result in results:
        assert result.text, "Text should not be empty"
        assert isinstance(result.modalities, list), "Modalities should be a list"


# ------------------ PPTX Processor Tests ------------------
def test_pptx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = PPTXProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_pptx_extract_notes():
    """Verify that PPTXProcessor correctly extracts slide notes."""
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = PPTXProcessor(config=config)
    result = processor.process(sample_file)
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    expected_text = "Data analysis has multiple facets and approaches"
    assert expected_text in combined_text, (
        f"Expected notes not found in extracted text: {combined_text}"
    )


# ------------------ Spreadsheet Processor Tests ------------------
def test_spreadsheet_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = SpreadsheetProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_spreadsheet_multi_sheet_content():
    """Test that SpreadsheetProcessor correctly extracts text from multiple sheets."""
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp", extract_images=True)
    processor = SpreadsheetProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Expected some text in spreadsheet."

    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )

    for sheet_name in ["Form Responses 1"]:
        assert sheet_name in combined_text, (
            f"Didn't find '{sheet_name}' in extracted text."
        )

    for snippet in [
        "What is your current educational enrollment status?",
        "What is your gender?",
        "Master's Degree",
        "Female",
    ]:
        assert snippet in combined_text, (
            f"Expected '{snippet}' not found in spreadsheet text."
        )

    assert isinstance(result.modalities, list), "Modalities should be a list."
    assert len(result.modalities) == 0, "Expected no images in this spreadsheet."


## ------------------ PDF Processor Tests ------------------
def test_pdf_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(
        output_path="tmp",
        extract_images=True,
        attachment_tag="<attachment>",
    )
    processor = PDFProcessor(config=config)
    processor.converter = (  # pyright: ignore[reportAttributeAccessIssue]
        lambda file_path: MarkdownOutput(
            markdown="dummy rendered content with ![](dummy.jpg)",
            images={"dummy.jpg": "dummy image data"},
            metadata={},
        )
    )
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_pdf_process_fast():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(
        output_path="tmp",
        extract_images=True,
        attachment_tag="<attachment>",
    )
    processor = PDFProcessor(config=config)
    processor.converter = (  # pyright: ignore[reportAttributeAccessIssue]
        lambda file_path: "dummy rendered content with ![](dummy.jpg)"
    )
    result = processor.process_fast(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


# ------------------ Text Processor Tests ------------------
def test_text_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "txt", "test.txt")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(output_path="tmp")
    processor = TextProcessor(config=config)
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list) and len(result.modalities) == 0, (
        "Modalities should be an empty list"
    )


# ------------------ URL Processor Tests ------------------
def test_url_process_standard():
    sample_url = "http://example.com"
    config = ProcessorConfig(
        output_path="tmp",
        extract_images=False,
        attachment_tag="<attachment>",
    )
    processor = URLProcessor(config=config)
    result = processor.process(sample_url)
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    assert "This domain" in combined_text, (
        "Expected 'This domain' in extracted text from http://example.com"
    )
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_url_process_invalid():
    sample_url = "http://thisurldoesnotexist.tld"
    config = ProcessorConfig(
        output_path="tmp",
        extract_images=False,
        attachment_tag="<attachment>",
    )
    processor = URLProcessor(config=config)
    result = processor.process(sample_url)
    assert not result.text, "Expected empty text for invalid URL"
    assert isinstance(result.modalities, list) and len(result.modalities) == 0, (
        "Expected no modalities for invalid URL"
    )
