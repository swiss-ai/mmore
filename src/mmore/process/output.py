from typing import List
import pickle
from .processors.processor import ProcessorResult
import os


def output(results: List[ProcessorResult], print_output: bool = True):
    print("Outputting results...")
    all_output = []
    for result in results:
        if print_output:
            for sample in result.samples:
                try:
                    print("Extracted Text:")
                    print(sample.text)
                    print("Modalities:")
                    print(sample.modalities)
                except:
                    try:
                        print("Extracted location:")
                        print(sample.location)
                    except:
                        continue

        print(result.samples)
        if len(result.samples) != 0:
            output_file_path = result.samples[-1].metadata.get("file_path", None)
            if output_file_path is None:
                output_file_path = ".yeet"
            file_extension = output_file_path.split(".")[-1]
            if file_extension == ".txt":
                output_file_path = "output/text/txt_output.jsonl"
            elif file_extension in [".pdf"]:
                output_file_path = "output/text/pdf_output.jsonl"
            elif file_extension in [".pptx"]:
                output_file_path = "output/text/pptx_output.jsonl"
            elif file_extension in [".mp4", ".avi", ".mov", ".mkv", ".mp3", ".flac", ".wav"]:
                output_file_path = "output/text/media_output.jsonl"
            elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
                output_file_path = "output/images"
            elif file_extension in [".xlsx", ".xls", ".csv"]:
                output_file_path = "output/spreadsheet_output.jsonl"
            elif file_extension in [".docx"]:
                output_file_path = "output/document_output.jsonl"
            elif file_extension in [".md"]:
                output_file_path = "output/markdown_output.jsonl"
            else:
                output_file_path = "output/unknown_output.jsonl"

            with open(output_file_path, "a") as f:
                for sample in result.samples:
                    f.write(str(sample.to_dict()))
                    f.write("\n")

            for sample in result.samples:
                all_output.append(str(sample.to_dict()))

    # Make dir and output file if it doesn't exist
    mode = 'a'
    if not os.path.isfile("output/all_output.jsonl"):
        mode = 'w'
        os.makedirs("output", exist_ok=True)

    with open("output/all_output.jsonl", mode) as f:
        for sample in all_output:
            f.write(sample)
            f.write("\n")
    print(f"All {len(all_output)} outputs saved to output/all_output.jsonl")


def output_to_pickle(results: List[ProcessorResult], output_result_path):
    with open(output_result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"All {len(results)} outputs saved to ", output_result_path)
