# docling_benchmark.py

import json
import logging
import time
from pathlib import Path
from typing import Iterable
import yaml
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem
            # Export Docling document format to JSON:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(conv_res.document.export_to_dict()))

            # Export Docling document format to YAML:
            with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

            # Export Docling document format to doctags:
            with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                fp.write(conv_res.document.export_to_document_tokens())

            # Export Docling document format to markdown:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(conv_res.document.export_to_markdown())

            # Export Docling document format to text:
            with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                fp.write(conv_res.document.export_to_markdown(strict_text=True))
        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)
    doc_converter = DocumentConverter()

    # Example: process all PDFs in test_data/pdf
    pdf_folder = Path("test_data/pdf")
    input_doc_paths = list(pdf_folder.glob("*.pdf"))

    start_time = time.time()

    conv_results = doc_converter.convert_all(input_doc_paths, raises_on_error=False)
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=Path("scratch")
    )

    total_time = time.time() - start_time
    _log.info(f"[Docling] Processed {len(input_doc_paths)} PDFs in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
