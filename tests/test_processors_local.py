import os
from mmore.process.processors.docx_processor import DOCXProcessor
from mmore.process.processors.eml_processor import EMLProcessor
from mmore.process.processors.md_processor import MarkdownProcessor
from mmore.process.processors.media_processor import MediaProcessor
from mmore.process.processors.pptx_processor import PPTXProcessor
from mmore.process.processors.spreadsheet_processor import SpreadsheetProcessor
from mmore.process.processors.base import ProcessorConfig
from mmore.process.processors.pdf_processor import PDFProcessor
from mmore.type import FileDescriptor

from marker.output import MarkdownOutput

SAMPLES_DIR = "examples/sample_data/"

def get_file_descriptor(file_path):
    return FileDescriptor.from_filename(file_path)

# ------------------ DOCX Processor Tests ------------------
def test_docx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = DOCXProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

#  ------------------ EML Processor Tests ------------------
def test_eml_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = EMLProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

def test_eml_headers_present():
    """
    Test that the processed EML file includes email headers: From, To, Subject, and Date
    """
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = EMLProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    # Ensure that the text field is not empty
    assert result.text, "Text should not be empty"
    # Combine text segments into one string
    combined_text = " ".join(result.text) if isinstance(result.text, list) else result.text
    expected_headers = ["From:", "To:", "Subject:", "Date:"]
    # Assert each header is present
    for header in expected_headers:
        assert header in combined_text, f"{header} not found in the processed text"

# ------------------ Markdown Processor Tests ------------------
def test_md_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = MarkdownProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

def test_md_image_extraction():
    """
    Test that the processor correctly extracts images and replaces image tags with the attachment placeholder.
    The sample md file contains one local image and one remote image. We expect: the processed text to contain two occurrences of the attachment placeholder and the modalities list to contain two entries
    """
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    custom_attachment_tag = "<attachment>"
    # Set extract images to True explicitly
    config = ProcessorConfig(custom_config={"output_path": "tmp", "extract_images": True})
    processor = MarkdownProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    combined_text = " ".join(result.text) if isinstance(result.text, list) else result.text
    # Count the number of attachment placeholders inserted in text
    placeholder_count = combined_text.count(custom_attachment_tag)
    assert placeholder_count == 2, f"Expected 2 attachment placeholders, found {placeholder_count}"
    # Assert that modalities is a list and that two images were extracted
    assert isinstance(result.modalities, list), "Modalities should be a list"
    assert len(result.modalities) == 2, f"Expected 2 images in modalities, found {len(result.modalities)}"

# ------------------ Media Processor Tests ------------------
def test_media_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"normal_model": "model-name", "output_path": "tmp"})
    processor = MediaProcessor(config=config)
    # Overriding load_models function with a lambda to bypass the actual model loading and heavy dependencies
    processor.load_models = lambda fast_mode=False: setattr(processor, 'pipelines', [lambda x: {"text": "dummy transcription"}])
    processor.load_models()
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

def test_media_process_batch():
    # Create a list of sample files
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    files = [sample_file, sample_file, sample_file]
    # Create a configuration and processor instance
    config = ProcessorConfig(custom_config={"normal_model": "model-name", "output_path": "tmp"})
    processor = MediaProcessor(config=config)
    # Overriding load_models function with a lambda to bypass the actual model loading and heavy dependencies
    processor.load_models = lambda fast_mode=False: setattr(
        processor, 'pipelines', [lambda x: {"text": "dummy transcription"} for _ in processor.devices]
    )
    processor.load_models()
    # Call process_batch with a dummy num_workers value
    results = processor.process_batch(files, fast_mode=False, num_workers=1)
    # Verify that each file in the batch produces a result with non-empty text and a list of modalities.
    assert len(results) == len(files), "Number of results should match number of files processed."
    for result in results:
        assert result.text, "Text should not be empty"
        assert isinstance(result.modalities, list), "Modalities should be a list"

# ------------------ PDF Processor Tests ------------------
def test_pptx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = PPTXProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

# ------------------ Spreadsheet Processor Tests ------------------
def test_spreadsheet_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = SpreadsheetProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

## ------------------ PDF Processor Tests ------------------
def test_pdf_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(custom_config={
        "extract_images": True,
        "attachment_tag": "<attachment>",
        "output_path": "tmp"
    })
    processor = PDFProcessor(config=config)
    processor.converter = lambda file_path: MarkdownOutput(
    markdown="dummy rendered content with ![](dummy.jpg)",
    images={"dummy.jpg": "dummy image data"},
    metadata={})
    # Process file
    result = processor.process(sample_file)   
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

def test_pdf_process_fast():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(custom_config={
        "extract_images": True,
        "attachment_tag": "<attachment>",
        "output_path": "tmp"
    })
    processor = PDFProcessor(config=config)
    processor.converter = lambda file_path: "dummy rendered content with ![](dummy.jpg)"
    # Process file
    result = processor.process_fast(sample_file)   
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"