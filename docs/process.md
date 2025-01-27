# :gear: Process

The process module enables the extraction and standardization of text and images from diverse file formats (listed below), making it ideal for creating datasets for applications such as RAG, multimodal content generation, and preprocessing data for multimodal LLMs/LLMs.

## :hammer: Quick Start
#### :technologist: Global installation
Setup the project in each device you want to use using our setup script or looking at what it does and doing it manually.
```bash
pip install -e '.[all]'
```

#### :computer: Running locally
You need to specify the input folder by modifying the [config file](https://github.com/OpenMeditron/End2End/blob/dask-cuda-poc/examples/process_config.yaml). You can also twist the parameters to your needs. Once ready, you can run the process using the following command:
```bash
python -m mmore process --config-file examples/process/config.yaml
```
The output of the pipeline has the following structure:
```
output_path
├── processors
│   ├── Processor_type_1
│   │   └── results.jsonl
│   ├── Processor_type_2
│   │   └── results.jsonl
│   ├── ...
│   
└── merged
│    └── merged_results.jsonl
|
└── images
```
#### :rocket: Running on distributed nodes

We provide [a simple bash script](./entrypoint_distributed.sh) to run the process on distributed mode. Please call it with your arguments.
```bash
bash scripts/process_distributed.sh -f /path/to/my/input/folder 
```

#### :scroll: Examples
You can find more examples scripts in [the `/examples` directory](../examples/).

## :zap: Optimization
### :racing_car: Fast mode

For some file types, we provide a fast mode that will allow you to process the files faster, using a different method. To use it, set the `use_fast_processors` to `true` in the config file.

Be aware that the fast mode might not be as accurate as the default mode, especially for scanned non-native PDFs, which may require Optical Character Recognition (OCR) for more accurate extraction.

### :rocket: Distributed mode

The project is designed to be easily scalable to a multi GPU / multi node environment. To use it, To use it, set the `distribued` to `true` in the config file., and follow the steps described in the [](../README.md#hammer-manual-installation) section.

### :wrench: File type parameters tuning

Many parameters are hardware-dependent and can be customized to suit your needs. For example, you can adjust the processor batch size, dispatcher batch size, and the number of threads per worker to optimize performance.

You can configure parameters by providing a custom config file. You can find an example of a config file in the [examples folder](examples/process_config.yaml).

:rotating_light: Not all parameters are configurable yet :wink:

## :scroll: More information on what's under the hood

### :construction: Pipeline architecture

Our pipeline is a 3 steps process:
- **Crawling**: We first crawl over the file/folder to list all the files we need to process.
- **Dispatching**: We then dispatch the files to the workers, using a dispatcher that will send the files to the workers in batches. This part is in charge of the load balancing between different nodes if the project is running in a distributed environment.
- **Processing**: The workers then process the files, using the appropriate tools for each file type. They extract the text, images, audio, and video frames, and send them to the next step. Our goal is to provide an easy way to add new processors for new file types, or even other types of processing for existing file types.

## 🛠️ Used tools

The project supports multiple file types and utilizes various AI-based tools for processing. Below is a table summarizing the supported file types and corresponding tools (N/A means no choice):

| **File Type**                         | **Default Mode Tool(s)**                                                                                                          | **Fast Mode Tool(s)**                                                                                                         |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **DOCX**                              | [python-docx](https://python-docx.readthedocs.io/en/latest/) to extract the text and images.                                      | N/A                                                                                                                         |
| **MD**                                | [markdown](https://python-markdown.github.io/) for text extraction, [markdownify](https://pypi.org/project/markdownify/) for HTML conversion | N/A                                                                                                                         |
| **PPTX**                              | [python-pptx](https://python-pptx.readthedocs.io/en/latest/) to extract the text and images.                                      | N/A                                                                                                                         |
| **XLSX**                              | [openpyxl](https://openpyxl.readthedocs.io/en/stable/) to extract the text and images.                                           | N/A                                                                                                                         |
| **TXT**                               | [python built-in library](https://docs.python.org/3/library/functions.html#open)                                                 | N/A                                                                                                                         |
| **EML**                               | [python built-in library](https://docs.python.org/3/library/email.html) | N/A                                                                                                                         |
| **MP4, MOV, AVI, MKV, MP3, WAV, AAC** | [moviepy](https://pypi.org/project/moviepy/) for video frame extraction; [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for transcription | [whisper-tiny](https://huggingface.co/openai/whisper-tiny)                                                                  |
| **PDF**                               | [marker-pdf](https://github.com/VikParuchuri/marker) for OCR and structured data extraction                                      | [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for text and image extraction                                                 |
| **Webpages (TBD)**                         | TODO| [selenium](https://selenium-python.readthedocs.io/) to navigate the webpage and extract content; [requests](https://docs.python-requests.org/en/master/) for images; [trafilatura](https://trafilatura.readthedocs.io/en/latest/) for content extraction |
---
We also use [Dask distributed](https://distributed.dask.org/en/latest/) to manage the distributed environment.

## :wrench: Customization
The system is designed to be extensible, allowing you to register custom processors for handling new file types or specialized processing. To implement a new processor you need to inherit the `Processor` class and implement only two methods:
- accepts: defines the file types your processor supports (e.g. docx)
- process: how to process a single file (input:file type, output: Multimodal sample, see other processors for reference)

See `TextProcessor` in `src/process/processors/text_processor.py` for a minimal example.
