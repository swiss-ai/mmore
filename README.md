<h1 align="center"> 

![image](https://github.com/user-attachments/assets/502e2c7e-1200-498a-9ebd-10a27ed48ab6)

</h1>


<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/github/v/release/OpenMeditron/End2End" alt="Release">
</p>

####  <center>Massive Multimodal Open RAG & Extraction</center>

A scalable multimodal pipeline for processing, indexing, and querying multimodal documents

Ever needed to take 8000 PDFs, 2000 videos, and 500 spreadsheets and feed them to an LLM as a knowledge base?
Well, MMORE is here to help you!

## :bulb: Quickstart

### Installation

#### (Step 0 – Install system dependencies)

Our package requires system dependencies. This snippet will take care of installing them!

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 \
  libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice \
  libpango-1.0-0 libpangoft2-1.0-0 weasyprint
```

#### Step 1 – Install MMORE

To install the package simply run:

```bash
pip install mmore
```

> :warning: This is a big package with a lot of dependencies, so we recommend to use `uv` to handle `pip` installations. [Check our tutorial on uv](./docs/uv.md).

### Minimal Example

You can use our predefined CLI commands to execute parts of the pipeline. Note that you might need to prepend `python -m` to the command if the package does not properly create bash aliases.

```bash
# Run processing
python -m mmore process --config-file examples/process/config.yaml
python -m mmore postprocess --config-file examples/postprocessor/config.yaml --input-data examples/process/outputs/merged/merged_results.jsonl

# Run indexer
python -m mmore index --config-file examples/index/config.yaml --documents-path examples/process/outputs/merged/final_pp.jsonl

# Run RAG
python -m mmore rag --config-file examples/rag/config.yaml
```

You can also use our package in python code as shown here:

```python
from mmore.process.processors.pdf_processor import PDFProcessor 
from mmore.process.processors.base import ProcessorConfig
from mmore.type import MultimodalSample

pdf_file_paths = ["/path/to/examples/sample_data/pdf/calendar.pdf"] #write here the full path, not a relative path
out_file = "/path/to/examples/process/outputs/example.jsonl"

pdf_processor_config = ProcessorConfig(custom_config={"output_path": "examples/process/outputs"})
pdf_processor = PDFProcessor(config=pdf_processor_config)
result_pdf = pdf_processor.process_batch(pdf_file_paths, False, 1) # args: file_paths, fast mode (True/False), num_workers

MultimodalSample.to_jsonl(out_file, result_pdf)
```

---

### Usage

To launch the MMORE pipeline, follow the specialised instructions in the docs.

![The MMORE pipelines archicture](https://github.com/user-attachments/assets/0cd61466-1680-43ed-9d55-7bd483a04a09)


1. **:page_facing_up: Input Documents**  
   Upload your multimodal documents (PDFs, videos, spreadsheets, and m(m)ore) into the pipeline.

2. [**:mag: Process**](./docs/process.md) 
   Extracts and standardizes text, metadata, and multimedia content from diverse file formats. Easily extensible! You can add your own processors to handle new file types.
   *Supports fast processing for specific types.*

3. [**:file_folder: Index**](./docs/index.md) 
   Organizes extracted data into a **hybrid retrieval-ready Vector Store DB**, combining dense and sparse indexing through [Milvus](https://milvus.io/). Your vector DB can also be remotely hosted and then you only have to provide a standard API. There is also an [HTTP Index API](./docs/index_api.md) for adding new files on the fly with HTTP requests.

4. [**:robot: RAG**](./docs/rag.md) 
   Use the indexed documents inside a **Retrieval-Augmented Generation (RAG) system**  that provides a [LangChain](https://www.langchain.com/) interface. Plug in any LLM with a compatible interface or add new ones through an easy-to-use interface.
   *Supports API hosting or local inference.*

5. **:tada: Evaluation**  
   *Coming soon*
   An easy way to evaluate the performance of your RAG system using Ragas.

See [the `/docs` directory](./docs) for additional details on each modules and hands-on tutorials on parts of the pipeline.


#### :construction: Supported File Types  

| **Category**      | **File Types**                           | **Supported Device**      |  **Fast Mode**      |
|--------------------|------------------------------------------|--------------------------| --------------------------|
| **Text Documents** | DOCX, MD, PPTX, XLSX, TXT, EML           | CPU                      | :x:
| **PDFs**           | PDF                                     | GPU/CPU                  | :white_check_mark:
| **Media Files**    | MP4, MOV, AVI, MKV, MP3, WAV, AAC       | GPU/CPU                  | :white_check_mark:
| **Web Content**    | HTML                                    | CPU                      | :x:


## Contributing

We welcome contributions to improve the current state of the pipeline, feel free to:

- Open an issue to report a bug or ask for a new feature
- Open a pull request to fix a bug or add a new feature
- You can find ongoing new features and bugs in the [Issues]
   
Don't hesitate to star the project :star: if you find it interesting! (you would be our star).

## License

This project is licensed under the Apache 2.0 License, see the [LICENSE :mortar_board:](LICENSE) file for details.
