import argparse
import logging
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, Dataset

import click
import fitz
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ..utils import load_config

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

PROCESS_EMOJI = "🚀"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[Process {PROCESS_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class PDFProcessConfig:
    data_path: Union[List[str], str]
    output_path: str
    model_name: str = "vidore/colpali-v1.3"
    skip_already_processed: bool = False
    num_workers: int = 2
    batch_size: int = 8

class PDFConverter:
    def __init__(self, dpi: int = 200):
        self.dpi = dpi
        self.tmp_root = Path(tempfile.mkdtemp(prefix="pdf2png_"))
        logger.debug(f"Temporary image directory: {self.tmp_root}")

    def convert_to_pngs(self, pdf_file: Path) -> List[Path]:
        png_paths = []
        out_dir = self.tmp_root / pdf_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_file)

        for page_num in range(len(doc)):
            out_path = out_dir / f"page_{page_num+1}.png"
            page = doc[page_num]
            pix = page.get_pixmap(dpi=self.dpi)
            pix.save(out_path)
            png_paths.append(out_path)

        doc.close()
        return png_paths

    def cleanup(self):
        import shutil
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
            logger.debug(f"Cleaned temporary directory {self.tmp_root}")


class ColPaliEmbedder:
    def __init__(self, model_name: str = "vidore/colpali-v1.3", device: str = "cuda:0"):
        self.device = device
        dtype = torch.bfloat16
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
    
    def get_images(self, paths: list[str]) -> List[Image.Image]:
        return [Image.open(path) for path in paths]

    def embed_images(self, image_paths:list[str], batch_size = 5):    
        images = self.get_images(image_paths)
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to(self.device))))
        ds_np = [d.float().cpu().numpy() for d in ds]
        return ds_np
    
def crawl_pdfs(data_paths: Union[str, List[str]]) -> List[Path]:
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    all_pdfs = []
    for root_dir in data_paths:
        root_path = Path(root_dir)
        pdf_files = list(root_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF(s) in {root_dir}")
        all_pdfs.extend(pdf_files)
    return all_pdfs

"""
def crawl_pdfs(data_paths: Union[str, List[str]]) -> List[Path]:
    if isinstance(data_paths, str):
        data_paths = [data_paths]
        
    config = CrawlerConfig(
        root_dirs=data_paths,
        supported_extensions=[".pdf"]
    )
    
    crawler = Crawler(config=config)
    result = crawler.crawl() 
    return [Path(file_desc.file_path) for file_desc in result()]
"""

def process_single_pdf(pdf_path: Path, model: ColPaliEmbedder, converter: PDFConverter) -> List[dict]:
    try:
        png_paths = converter.convert_to_pngs(pdf_path)
        image_embeddings = model.embed_images(png_paths)

        doc = fitz.open(pdf_path)
        page_records = []
        for page_num, embedding in enumerate(image_embeddings):
            text = doc[page_num].get_text("text").strip()
            page_records.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "page_number": page_num + 1,
                "text": text,
                "images": [],
                "embedding": embedding.astype("float32"),
            })
        doc.close()
        return page_records
    except Exception as e:
        logger.error(f"❌ Failed to process {pdf_path.name}: {e}")
        return []


def save_results(records: List[dict], output_path: Path):
    df = pd.DataFrame(records)
    df["embedding"] = df["embedding"].apply(lambda x: x.tolist())
    parquet_path = output_path / "pdf_page_objects.parquet"

    logger.info(f"Saving {len(df)} records to {parquet_path}")
    try:
        df.to_parquet(parquet_path, index=False, compression="zstd")
    except Exception as e:
        logger.error(f"Failed to write Parquet: {e}")
    return parquet_path

def process_pdf_batch(batch_pdfs: List[Path], config: PDFProcessConfig) -> List[dict]:
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ColPaliEmbedder(config.model_name, device=device)
        converter = PDFConverter()
        batch_records = []

        for pdf_path in tqdm(batch_pdfs, desc=f"Batch on {device}", ncols=100):
            batch_records.extend(process_single_pdf(pdf_path, model, converter))

        converter.cleanup()
        del model
        torch.cuda.empty_cache()

        return batch_records
    except Exception as e:
        logger.error(f"❌ Batch failed: {e}")
        return []

def run_process(config_file: str):
    click.echo(f"Processing configuration file path: {config_file}")
    overall_start_time = time.time()

    config = load_config(config_file, PDFProcessConfig)
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "pdf_page_objects.parquet"
    already_processed_pdfs = set()

    if config.skip_already_processed and parquet_path.exists():
        logger.info(f"Skip mode enabled — loading existing {parquet_path}")
        try:
            df_existing = pd.read_parquet(parquet_path, columns=["pdf_name"])
            already_processed_pdfs = set(df_existing["pdf_name"].unique())
            logger.info(f"Found {len(already_processed_pdfs)} processed PDFs.")
        except Exception as e:
            logger.warning(f"Could not read existing parquet: {e}")

    pdf_files = crawl_pdfs(config.data_path)
    pdf_files = [p for p in pdf_files if p.name not in already_processed_pdfs]

    if not pdf_files:
        logger.info("No new PDFs to process.")
        return

    logger.info(f"Processing {len(pdf_files)} PDFs in parallel batches of {config.batch_size} using {config.num_workers} workers...")

    batches = [pdf_files[i:i + config.batch_size] for i in range(0, len(pdf_files), config.batch_size)]
    all_page_records = []

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_pdf_batch, batch, config): idx
            for idx, batch in enumerate(batches)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches", ncols=100):
            batch_result = future.result()
            all_page_records.extend(batch_result)

    if all_page_records:
        if parquet_path.exists():
            logger.info("Merging new results with existing parquet...")
            old_df = pd.read_parquet(parquet_path)
            new_df = pd.DataFrame(all_page_records)
            new_df["embedding"] = new_df["embedding"].apply(lambda x: x.tolist())
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
            merged_df.to_parquet(parquet_path, index=False, compression="zstd")
        else:
            save_results(all_page_records, output_dir)
    else:
        logger.info("No new PDFs to process.")

    overall_end_time = time.time()
    logger.info(f"✅ All done! Total time: {overall_end_time - overall_start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs and store page embeddings in Parquet.")
    parser.add_argument("--config_file", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    process(args.config_file)
