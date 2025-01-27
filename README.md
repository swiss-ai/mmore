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

## MMORE Installation Guide

### Installation Option 1: pip (recommended)

To install all dependencies, run:

```bash
pip install -e '.[all]'
```

To install only processor-related dependencies, run:

```bash
pip install -e '.[processor]'
```

To install only RAG-related dependencies, run:

```bash
pip install -e '.[rag]'
```

---

### Installation Option 2: uv

#### Step 1: Install system dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 \
  libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice
```

#### Step 2: Install `uv`

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 3: Clone this repository

```bash
git clone https://github.com/swiss-ai/mmore
cd mmore
```

#### Step 4: Install project and dependencies

```bash
uv sync
```

For CPU-only installation, use:

```bash
uv sync --extra cpu
```

#### Step 5: Run a test command

Activate the virtual environment before running commands:

```bash
source .venv/bin/activate
```

Alternatively, prepend each command with `uv run`:

```bash
# Run processing
python -m mmore process --config_file examples/process_config.yaml

# Run indexer
python -m mmore index --config-file ./examples/index/indexer_config.yaml

# Run RAG
python -m mmore rag --config-file ./examples/rag/rag_config_local.yaml
```

---

### Installation Option 3: Docker

**Note:** For manual installation without Docker, refer to the section below.

#### Step 1: Install Docker

Follow the official [Docker installation guide](https://docs.docker.com/get-started/get-docker/).

#### Step 2: Build the Docker image

```bash
docker build . --tag mmore
```

To build for CPU-only platforms (results in a smaller image size):

```bash
docker build --build-arg PLATFORM=cpu -t mmore .
```

#### Step 3: Start an interactive session

```bash
docker run -it -v ./test_data:/app/test_data mmore
```

*Note:* The `test_data` folder is mapped to `/app/test_data` inside the container, corresponding to the default path in `examples/process_config.yaml`.

#### Step 4: Run the application inside the container

```bash
# Run processing
mmore process --config-file examples/process/config.yaml

# Run indexer
mmore index --config-file ./examples/index/indexer_config.yaml

# Run RAG
mmore rag --config-file ./examples/rag/rag_config_local.yaml
```

---


### Minimal Example

```python
from mmore.process.processors.pdf_processor import PDFProcessor 
from mmore.process.processors.base import ProcessorConfig
from mmore.type import MultimodalSample

pdf_file_paths = ["examples/sample_data/pdf/calendar.pdf"]
out_file = "results/example.jsonl"

pdf_processor_config = ProcessorConfig(custom_config={"output_path": "results"})
pdf_processor = PDFProcessor(config=pdf_processor_config)
result_pdf = pdf_processor.process_batch(pdf_file_paths, True, 1) # args: file_paths, fast mode (True/False), num_workers

MultimodalSample.to_jsonl(out_file, result_pdf)
```

### Usage

To launch the MMORE pipeline follow the specialised instructions in the docs.

![The MMORE pipelines archicture](https://github.com/user-attachments/assets/0cd61466-1680-43ed-9d55-7bd483a04a09)


1. **:page_facing_up: Input Documents**  
   Upload your multimodal documents (PDFs, videos, spreadsheets, and more) into the pipeline.

2. [**:mag: Process**](./docs/process.md) 
   Extracts and standardizes text, metadata, and multimedia content from diverse file formats. Easily extensible ! Add your own processors to handle new file types.  
   *Supports fast processing for specific types.*

3. [**:file_folder: Index**](./docs/index.md) 
   Organizes extracted data into a **hybrid retrieval-ready Vector Store DB**, combining dense and sparse indexing through [Milvus](https://milvus.io/). Your vector DB can also be remotely hosted and only need to provide a standard API. 

4. [**:robot: RAG**](./docs/rag.md) 
   Use the indexed documents inside a **Retrieval-Augmented Generation (RAG) system**  that provides a [LangChain](https://www.langchain.com/) interface. Plug in any LLM with a compatible interface or add new ones through an easy-to-use interface.
   *Supports API hosting or local inference.*

5. **:tada: Evaluation**  
   *Coming soon*
   An easy way to evaluate the performance of your RAG system using Ragas

See [the `/docs` directory](/docs) for additional details on each modules and hands-on tutorials on parts of the pipeline.


#### :construction: Supported File Types  

| **Category**      | **File Types**                           | **Supported Device**      |  **Fast Mode**      |
|--------------------|------------------------------------------|--------------------------| --------------------------|
| **Text Documents** | DOCX, MD, PPTX, XLSX, TXT, EML              | CPU                      | :x:
| **PDFs**           | PDF                                     | GPU/CPU                  | :white_check_mark:
| **Media Files**    | MP4, MOV, AVI, MKV, MP3, WAV, AAC       | GPU/CPU                  | :white_check_mark:
| **Web Content (TBD)**    | Webpages                                | GPU/CPU                  | :white_check_mark:


## Contributing

We welcome contributions to improve the current state of the pipeline, feel free to:

- Open an issue to report a bug or ask for a new feature
- Open a pull request to fix a bug or add a new feature
- You can find ongoing new features and bugs in the [Issues]
   
Don't hesitate to star the project :star: if you find it interesting! (you would be our star)

## License
This project is licensed under the Apache 2.0 License, see the [LICENSE :mortar_board:](LICENSE) file for details.

## Acknowledgements

This project is part of the [**OpenMeditron**](https://huggingface.co/OpenMeditron) initiative developed in [LiGHT](www.yale-light.org) lab at EPFL/Yale/CMU Africa in collaboration with the [**SwissAI**](https://www.swiss-ai.org/) initiative. Thank you Scott Mahoney, Mary-Anne Hartley
