import os
import logging
import warnings
import json
from io import BytesIO
from typing import List

from langchain_core.documents import Document
import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pythainlp.util import normalize

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        logger.info("Initializing DocumentProcessor with Docling...")
        
        # Setup English Converter (Standard PDF Parsing)
        en_pipeline_opts = PdfPipelineOptions(do_table_structure=True)
        self.converter_en = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=en_pipeline_opts)
            }
        )

        # Setup Thai Converter (Image -> EasyOCR)
        ocr_options = EasyOcrOptions(lang=['th', 'en'], use_gpu=False) 
        
        th_pipeline_opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=ocr_options
        )
        
        self.converter_th = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=th_pipeline_opts)
            }
        )
        logger.info("Docling converters ready.")

    def _get_logical_page(self, filename: str, physical_page_idx: int) -> str:
        return str(physical_page_idx + 1)

    def ingest_manual(self, dataset_folder: str) -> List[Document]:
        documents = []
        if not os.path.exists(dataset_folder):
            logger.error(f"Folder not found: {dataset_folder}")
            return []

        files = [f for f in os.listdir(dataset_folder) if f.endswith(".pdf")]
        logger.info(f"Found {len(files)} PDF files in '{dataset_folder}'")
        
        thai_filenames = [
            "Internal-Memo-2025", 
            "Employee-Handbook-2024", 
            "Benefits-Reimbursement-Policy-2024"
        ]

        for file in files:
            file_path = os.path.join(dataset_folder, file)
            logger.info(f"Processing: {file}...")

            try:
                if any(name in file for name in thai_filenames):
                    logger.info(f"Detected Thai document: {file}")
                    docs = self._process_thai_pdf(file_path, file)
                else:
                    logger.info(f"Detected Standard/English document: {file}")
                    docs = self._process_english_pdf(file_path, file)
                
                documents.extend(docs)
                logger.info(f"Finished {file}: Obtained {len(docs)} chunks.")
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")

        return documents

    def _process_english_pdf(self, file_path: str, filename: str) -> List[Document]:
        conv_result = self.converter_en.convert(file_path)
        docs = []

        sorted_page_nums = sorted(conv_result.document.pages.keys())

        for i, page_no in enumerate(sorted_page_nums):
            text = conv_result.document.export_to_markdown(page_no=page_no)
            
            if not text.strip():
                continue

            logical_page = self._get_logical_page(filename, i)
            
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "logical_page": logical_page,
                }
            ))
            
        return docs

    def _process_thai_pdf(self, file_path: str, filename: str) -> List[Document]:
        pdf = pdfium.PdfDocument(file_path)
        docs = []

        logger.info(f"Starting Image+OCR Pipeline for {filename}...")
        
        for i in range(len(pdf)):
            # Rasterize Page
            page = pdf[i]
            bitmap = page.render(scale=3.0) 
            pil_image = bitmap.to_pil()

            # Preprocess
            pil_image = ImageOps.grayscale(pil_image)
            pil_image = ImageOps.autocontrast(pil_image) 
            
            # Convert via Docling (Image Input)
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            doc_stream = DocumentStream(name=f"page_{i}.png", stream=img_byte_arr)

            try:
                conv_result = self.converter_th.convert(doc_stream)
                raw_text = conv_result.document.export_to_markdown()
            except Exception as e:
                logger.warning(f"OCR failed on page {i}: {e}")
                raw_text = ""

            # Normalize
            clean_text = normalize(raw_text)

            if not clean_text.strip():
                continue

            logical_page = self._get_logical_page(filename, i)

            docs.append(Document(
                page_content=clean_text,
                metadata={
                    "source": filename,
                    "logical_page": logical_page,
                }
            ))
            
            if (i + 1) % 5 == 0:
                logger.info(f"   Processed {i+1}/{len(pdf)} pages...")

        return docs

if __name__ == "__main__":
    print("\n--- Starting Ingestion Test ---")
    processor = DocumentProcessor()
    
    output_dir = "ingested_data"
    output_file = os.path.join(output_dir, "ingested_documents.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists("dataset"):
        docs = processor.ingest_manual("dataset/")
        
        if docs:
            print(f"\nSUCCESS: Processed {len(docs)} pages.")
            
            # Save output for verification
            debug_data = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            print(f"Data saved to '{output_file}'")

    else:
        print("Error: 'dataset' folder not found.")