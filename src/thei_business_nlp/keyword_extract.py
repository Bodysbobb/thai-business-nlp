"""
Thai-English PDF Keyword Extraction

Author: Pattawee Puangchit
Version: 0.1.0
Date: 2025-10-19
"""

import os
import re
import json
import logging
import warnings
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import fitz
import pytesseract
from PIL import Image
from pythainlp import word_tokenize, sent_tokenize
from pythainlp.corpus import thai_words
from tqdm import tqdm
import cv2
import numpy as np

# -------------------------------------------------------------
# Import config from package (relative import)
# -------------------------------------------------------------
from .config import (
    ensure_directories,
    get_log_file,
    LOG_LEVEL,
    ENABLE_WORD_VALIDATION,
    ENABLE_PARALLEL_PROCESSING,
    MAX_WORKERS,
    LANGUAGE_MODE,
    MIN_TEXT_LENGTH,
    MIN_WORD_LENGTH,
    MIN_WORD_FREQUENCY,
    MAX_AUTO_KEYWORDS,
    MAX_SENTENCES_PER_KEYWORD,
    VALIDATE_SENTENCE_WITH_THAI_NLP,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    KEYWORDS_COUNT_FILE,
    JSON_FOLDER,
    KEYWORD_FILE,
    KEYWORD_SHEET,
    STOPWORDS_FILE,
    EXCLUDED_KEYWORDS_FILE,
    TESSERACT_PATH,
    USE_OCR_FALLBACK,
    OCR_RESOLUTION,
    OCR_LANGUAGES,
    FILTER_YEARS,
    FILTER_SECTORS,
    FILTER_INDUSTRIES,
)
# -------------------------------------------------------------

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
warnings.filterwarnings("ignore")


class KeywordExtractor:
    """Extract predefined and automatically generated keywords from Thai/English PDFs."""

    def __init__(self):
        self.setup_logging()
        self.setup_word_validation()
        self.load_external_resources()
        self.load_predefined_keywords()
        self.keyword_counts = []
        self.keyword_sentences = defaultdict(lambda: defaultdict(set))
        self._lock = threading.Lock()

    # -------------------- SETUP & LOGGING --------------------

    def setup_logging(self):
        ensure_directories()
        log_file = get_log_file()
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_word_validation(self):
        try:
            self.thai_words_set = set(thai_words()) if ENABLE_WORD_VALIDATION else set()
        except Exception:
            self.thai_words_set = set()

    # -------------------- RESOURCE LOADING --------------------

    def load_external_resources(self):
        """Load stopwords and excluded keywords."""
        self.all_stopwords = set()
        try:
            with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
                stopwords_data = json.load(f)
                self.all_stopwords.update(stopwords_data.get("thai", []))
                self.all_stopwords.update(stopwords_data.get("english", []))
        except Exception:
            self.logger.warning("Could not load stopwords from JSON, using empty set")

        self.excluded_keywords = set()
        try:
            with open(EXCLUDED_KEYWORDS_FILE, "r", encoding="utf-8") as f:
                excluded_data = json.load(f)
                self.excluded_keywords.update(excluded_data.get("thai", []))
                self.excluded_keywords.update(excluded_data.get("english", []))
        except Exception:
            self.logger.warning("Could not load excluded keywords from JSON, using empty set")

    def load_predefined_keywords(self):
        """Load sector-specific keywords from Excel mapping file."""
        self.predefined_keywords = {}
        try:
            df = pd.read_excel(KEYWORD_FILE, sheet_name=KEYWORD_SHEET)
            for _, row in df.iterrows():
                sector = str(row.get("Sector", "")).strip()
                thai_keyword = str(row.get("Thai", "")).strip()
                english_keyword = str(row.get("English", "")).strip()
                esg_category = str(row.get("ESG", "")).strip()

                if not sector or sector == "nan":
                    continue

                entry = {
                    "thai": thai_keyword if thai_keyword != "nan" else "",
                    "english": english_keyword if english_keyword != "nan" else "",
                    "esg": esg_category if esg_category != "nan" else "",
                    "type": "predefined",
                }

                self.predefined_keywords.setdefault(sector, []).append(entry)

        except Exception as e:
            self.logger.error(f"Failed to load predefined keywords: {str(e)}")
            self.predefined_keywords = {}

    # -------------------- PDF PROCESSING --------------------

    def collect_pdf_files(self):
        """Traverse input folders and collect all valid PDF paths."""
        pdf_files = []
        valid_sectors = set(self.predefined_keywords.keys())

        for year_folder in os.listdir(INPUT_FOLDER):
            year_path = os.path.join(INPUT_FOLDER, year_folder)
            if not os.path.isdir(year_path):
                continue
            if FILTER_YEARS and year_folder not in FILTER_YEARS:
                continue

            for sector_folder in os.listdir(year_path):
                sector_path = os.path.join(year_path, sector_folder)
                if not os.path.isdir(sector_path):
                    continue
                if sector_folder not in valid_sectors:
                    continue
                if FILTER_SECTORS and sector_folder not in FILTER_SECTORS:
                    continue

                for industry_folder in os.listdir(sector_path):
                    industry_path = os.path.join(sector_path, industry_folder)
                    if not os.path.isdir(industry_path):
                        continue
                    if FILTER_INDUSTRIES and industry_folder not in FILTER_INDUSTRIES:
                        continue

                    for filename in os.listdir(industry_path):
                        if filename.lower().endswith(".pdf"):
                            business_name = os.path.splitext(filename)[0]
                            pdf_files.append(
                                {
                                    "year": year_folder,
                                    "sector": sector_folder,
                                    "industry": industry_folder,
                                    "business": business_name,
                                    "filename": filename,
                                    "filepath": os.path.join(industry_path, filename),
                                }
                            )
        return pdf_files

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract readable text from a PDF, with OCR fallback if needed."""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text and len(page_text.strip()) >= MIN_TEXT_LENGTH:
                    text += page_text + "\n"
                    continue

                if USE_OCR_FALLBACK:
                    try:
                        mat = fitz.Matrix(OCR_RESOLUTION / 72, OCR_RESOLUTION / 72)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))

                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(
                            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        ocr_text = pytesseract.image_to_string(
                            Image.fromarray(thresh), lang=OCR_LANGUAGES
                        )
                        if ocr_text and len(ocr_text.strip()) >= MIN_TEXT_LENGTH:
                            text += ocr_text + "\n"
                    except Exception as e:
                        self.logger.debug(f"OCR failed for page {page_num + 1}: {str(e)}")
            doc.close()
        except Exception as e:
            self.logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return text

    # -------------------- KEYWORD EXTRACTION --------------------

    def is_valid_thai_word(self, word):
        return True if not ENABLE_WORD_VALIDATION else word in self.thai_words_set

    def extract_auto_keywords(self, text: str):
        """Extract automatically frequent Thai/English words."""
        thai_pattern = re.compile(r"[\u0E00-\u0E7F]+")
        english_pattern = re.compile(r"\b[a-zA-Z]+\b")

        all_tokens = []
        if LANGUAGE_MODE in ["Thai", "Both"]:
            all_tokens += word_tokenize(text, engine="newmm", keep_whitespace=False)
        if LANGUAGE_MODE in ["Eng", "Both"]:
            all_tokens += english_pattern.findall(text)

        word_freq = Counter()
        for token in all_tokens:
            if (
                len(token) >= MIN_WORD_LENGTH
                and not token.isdigit()
                and token.lower() not in self.all_stopwords
                and token not in self.excluded_keywords
            ):
                if thai_pattern.match(token) and not self.is_valid_thai_word(token):
                    continue
                word_freq[token] += 1

        filtered = [(w, c) for w, c in word_freq.items() if c >= MIN_WORD_FREQUENCY]
        return Counter(dict(filtered)).most_common(MAX_AUTO_KEYWORDS)

    def count_predefined_keywords(self, text, sector):
        """Count predefined keywords for a given sector."""
        if sector not in self.predefined_keywords:
            return []
        text_lower = text.lower()

        results = []
        for entry in self.predefined_keywords[sector]:
            thai_kw = entry["thai"]
            eng_kw = entry["english"]
            esg = entry["esg"]

            count = 0
            keyword_text = ""
            if thai_kw and LANGUAGE_MODE in ["Thai", "Both"]:
                c = text.count(thai_kw)
                if c:
                    keyword_text = thai_kw
                    count += c
            if eng_kw and LANGUAGE_MODE in ["Eng", "Both"]:
                c = text_lower.count(eng_kw.lower())
                if c and not keyword_text:
                    keyword_text = eng_kw
                    count += c
            if count:
                results.append({"keyword": keyword_text, "count": count, "esg": esg, "type": "predefined"})
        return results

    # -------------------- SENTENCE EXTRACTION --------------------

    def extract_sentence_based_snippets(self, text, keyword):
        """Return all sentences containing a given keyword."""
        snippets = set()
        collected = 0
        try:
            sentences = (
                sent_tokenize(text, engine="crfcut")
                if VALIDATE_SENTENCE_WITH_THAI_NLP
                else re.split(r"[.!?।]+", text)
            )
            sentences = [s.strip() for s in sentences if s.strip()]
        except Exception:
            sentences = re.split(r"[.!?।]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

        keyword_lower = keyword.lower()
        for sentence in sentences:
            if collected >= MAX_SENTENCES_PER_KEYWORD or len(sentence) < 20:
                continue
            sentence_lower = sentence.lower()
            contains_kw = (
                keyword in sentence
                if re.search(r"[\u0E00-\u0E7F]", keyword)
                else keyword_lower in sentence_lower
            )
            if not contains_kw:
                continue
            snippets.add(sentence)
            collected += 1
        return list(snippets)

    # -------------------- MAIN PROCESSING --------------------

    def process_single_pdf(self, pdf_info):
        """Extract keywords from one PDF file."""
        try:
            text = self.extract_text_from_pdf(pdf_info["filepath"])
            if not text or len(text.strip()) < MIN_TEXT_LENGTH:
                return False, {"error": "No valid text extracted"}

            auto_kw = self.extract_auto_keywords(text)
            predefined_kw = self.count_predefined_keywords(text, pdf_info["sector"])

            valid_keywords = [
                *[kw for kw in predefined_kw if kw["count"] > 0],
                *[
                    {"keyword": k, "count": c, "esg": "", "type": "auto"}
                    for k, c in auto_kw
                    if k
                    and k not in self.excluded_keywords
                    and k.lower() not in self.all_stopwords
                ],
            ]

            with self._lock:
                for kw in valid_keywords:
                    self.keyword_counts.append(
                        {
                            "year": pdf_info["year"],
                            "sector": pdf_info["sector"],
                            "industry": pdf_info["industry"],
                            "business": pdf_info["business"],
                            "filename": pdf_info["filename"],
                            "keyword": kw["keyword"],
                            "count": kw["count"],
                            "esg": kw["esg"],
                            "type": kw["type"],
                        }
                    )

                file_key = f"{pdf_info['year']}_{pdf_info['sector']}_{pdf_info['industry']}_{pdf_info['business']}"
                for kw in valid_keywords:
                    if kw["count"] > 0:
                        snippets = self.extract_sentence_based_snippets(text, kw["keyword"])
                        if snippets:
                            self.keyword_sentences[file_key][kw["keyword"]].update(snippets)

            return True, {"processed": True}
        except Exception as e:
            return False, {"error": str(e)}

    # -------------------- OUTPUT SAVING --------------------

    def save_keyword_counts(self):
        if not self.keyword_counts:
            self.logger.warning("No keyword counts to save")
            return
        try:
            df = pd.DataFrame(self.keyword_counts).sort_values(
                ["year", "sector", "industry", "business", "count"],
                ascending=[True, True, True, True, False],
            )
            output_path = Path(OUTPUT_FOLDER) / KEYWORDS_COUNT_FILE
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"Saved {len(df)} keyword entries to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save keyword counts: {str(e)}")

    def save_keyword_sentences(self):
        if not self.keyword_sentences:
            self.logger.warning("No keyword sentences to save")
            return
        saved = 0
        json_root = Path(OUTPUT_FOLDER) / JSON_FOLDER
        for file_key, keywords in self.keyword_sentences.items():
            try:
                parts = file_key.split("_")
                if len(parts) < 4:
                    continue
                year, sector, industry, business = parts[0], parts[1], parts[2], "_".join(parts[3:])
                out_dir = json_root / year / sector / industry
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / f"{business}_sentences.json"

                json_data = {
                    "metadata": {
                        "year": year,
                        "sector": sector,
                        "industry": industry,
                        "business": business,
                        "extraction_date": datetime.now().isoformat(),
                        "total_keywords": len(keywords),
                        "total_sentences": sum(len(v) for v in keywords.values()),
                    },
                    "keywords": {k: list(v) for k, v in keywords.items()},
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                saved += 1
            except Exception as e:
                self.logger.error(f"Failed to save file {file_key}: {str(e)}")
        self.logger.info(f"Saved {saved} JSON files")

    # -------------------- MAIN ENTRY --------------------

    def run(self):
        pdf_files = self.collect_pdf_files()
        if not pdf_files:
            self.logger.warning("No PDF files found")
            return

        self.logger.info(f"Processing {len(pdf_files)} PDF files...")
        if ENABLE_PARALLEL_PROCESSING and MAX_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(self.process_single_pdf, pdf) for pdf in pdf_files]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                    success, result = f.result()
                    if not success:
                        self.logger.warning(f"Failed: {result.get('error', 'Unknown error')}")
        else:
            for pdf in tqdm(pdf_files, desc="Processing PDFs"):
                success, result = self.process_single_pdf(pdf)
                if not success:
                    self.logger.warning(f"Failed: {result.get('error', 'Unknown error')}")

        self.save_keyword_counts()
        self.save_keyword_sentences()


def main():
    extractor = KeywordExtractor()
    try:
        extractor.run()
    except Exception as e:
        print(f"Keyword extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
