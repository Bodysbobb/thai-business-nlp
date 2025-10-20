"""
Thai-English Sentiment Analysis for PDF Keywords

Author: Pattawee Puangchit
Version: 0.1.0
Date: 2025-10-19
"""

import os
import json
import logging
import re
import requests
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------
# Import configuration from the package
# -------------------------------------------------------------
from thai_business_nlp.config import (
    ensure_directories,
    get_log_file,
    LOG_LEVEL,
    ENABLE_PARALLEL_PROCESSING,
    MAX_WORKERS,
    OUTPUT_FOLDER,
    JSON_FOLDER,
    SENTIMENT_LEXICON_FILE,
    CONTEXTUAL_PATTERNS_FILE,
    KEYWORD_FILE,
    DEFAULT_SENTIMENT,
    SENTIMENT_SUMMARY_FILE,
    SENTIMENT_DETAIL_FILE,
    ENABLE_AI_SENTIMENT,
    AI_MODEL_URL,
    AI_MODEL_NAME,
    AI_MIN_TEXT_LENGTH,
    AI_MAX_CACHE_SIZE,
)
# -------------------------------------------------------------


class SentimentAnalyzer:
    """Analyze sentiment from JSON keyword snippets produced by KeywordExtractor."""

    def __init__(self):
        self.setup_logging()
        self.load_sentiment_resources()
        self.sentiment_details = []
        self.sentiment_summaries = []
        self._lock = threading.Lock()
        self._language_cache = {}
        self._ai_cache = {}

    # -------------------- LOGGING --------------------

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

    # -------------------- RESOURCE LOADING --------------------

    def load_sentiment_resources(self):
        """Load lexicons and contextual rules."""
        self.sentiment_lexicon = self.load_sentiment_from_json()
        self.contextual_patterns = self.load_contextual_patterns_from_json()
        self.load_sentiment_from_excel()

        # Optimized lookup
        self.thai_sentiment_words = {
            w: s for w, s in self.sentiment_lexicon.items() if any("\u0E00" <= c <= "\u0E7F" for c in w)
        }
        self.english_sentiment_words = {
            w.lower(): s for w, s in self.sentiment_lexicon.items() if w.isascii()
        }

        # AI model config
        self.ai_enabled = ENABLE_AI_SENTIMENT
        self.ai_model_url = AI_MODEL_URL
        self.ai_model_name = AI_MODEL_NAME

        self.logger.info(
            f"Loaded {len(self.sentiment_lexicon)} sentiment words, "
            f"{len(self.contextual_patterns)} contextual patterns, "
            f"AI enabled: {self.ai_enabled}"
        )

    def load_sentiment_from_json(self):
        data = {}
        try:
            with open(SENTIMENT_LEXICON_FILE, "r", encoding="utf-8") as f:
                src = json.load(f)
                for s_type, langs in src.items():
                    for lang, words in langs.items():
                        for w in words:
                            data[w] = s_type.capitalize()
        except Exception as e:
            self.logger.warning(f"Could not load sentiment lexicon: {str(e)}")
        return data

    def load_contextual_patterns_from_json(self):
        try:
            with open(CONTEXTUAL_PATTERNS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load contextual patterns: {str(e)}")
            return {}

    def load_sentiment_from_excel(self):
        """Optionally add sentiment words from Excel."""
        try:
            df = pd.read_excel(KEYWORD_FILE, sheet_name="sentiment")
            for _, row in df.iterrows():
                for s_type in ["Positive", "Negative", "Neutral"]:
                    words = str(row.get(s_type, "")).strip()
                    if not words or words == "nan":
                        continue
                    for w in words.split(","):
                        w = w.strip()
                        if w:
                            self.sentiment_lexicon[w] = s_type
        except Exception:
            self.logger.warning("Could not load sentiment words from Excel")

    # -------------------- DETECTION HELPERS --------------------

    def detect_language(self, text: str) -> str:
        """Roughly detect language of a text."""
        if text in self._language_cache:
            return self._language_cache[text]

        th = len(re.findall(r"[\u0E00-\u0E7F]", text))
        en = len(re.findall(r"[a-zA-Z]", text))
        result = "Thai" if th > en else "English" if en > th else "Mixed"

        if len(self._language_cache) < 10000:
            self._language_cache[text] = result
        return result

    # -------------------- AI & CONTEXTUAL --------------------

    def analyze_with_ai_batch(self, texts_and_keywords):
        """Batch AI inference for efficiency."""
        if not self.ai_enabled or not texts_and_keywords:
            return {}
        results = {}
        for batch_start in range(0, len(texts_and_keywords), 10):
            batch = texts_and_keywords[batch_start: batch_start + 10]
            try:
                prompt = "Analyze sentiment for these Thai/English sentences.\nRespond only with format: <n>:Positive/Negative/Neutral\n\n"
                for i, (txt, kw) in enumerate(batch, 1):
                    prompt += f"{i}. Keyword '{kw}': {txt[:150]}\n"

                response = requests.post(
                    self.ai_model_url,
                    json={
                        "model": self.ai_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 50, "top_p": 0.1},
                    },
                    timeout=15,
                )
                if response.status_code == 200:
                    txt = response.json().get("response", "").lower()
                    matches = re.findall(r"(\d+):(positive|negative|neutral)", txt)
                    for n, sent in matches:
                        idx = int(n) - 1
                        if 0 <= idx < len(batch):
                            text, keyword = batch[idx]
                            cache_key = f"{keyword}:{hashlib.md5(text.encode()).hexdigest()[:16]}"
                            results[cache_key] = (sent.capitalize(), "ai_batch", 0.8)
            except Exception as e:
                self.logger.debug(f"AI batch failed: {str(e)}")
        return results

    def analyze_with_ai(self, text, keyword):
        if not self.ai_enabled:
            return None, "", 0.0
        if len(text.strip()) < AI_MIN_TEXT_LENGTH or len(text.strip()) > 500:
            return None, "", 0.0

        cache_key = f"{keyword}:{hashlib.md5(text.encode()).hexdigest()[:16]}"
        if cache_key in self._ai_cache:
            return self._ai_cache[cache_key]

        try:
            if any("\u0E00" <= c <= "\u0E7F" for c in text):
                prompt = f'วิเคราะห์ sentiment ของคำว่า "{keyword}" ในประโยค: "{text[:100]}"\nตอบเฉพาะ: Positive/Negative/Neutral'
            else:
                prompt = f'Sentiment for "{keyword}" in: "{text[:100]}"\nRespond only: Positive/Negative/Neutral'

            response = requests.post(
                self.ai_model_url,
                json={
                    "model": self.ai_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5, "top_p": 0.1},
                },
                timeout=3,
            )

            if response.status_code == 200:
                reply = response.json().get("response", "").lower()
                if "positive" in reply or "บวก" in reply:
                    sentiment = "Positive"
                elif "negative" in reply or "ลบ" in reply:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                result = (sentiment, "ai_analysis", 0.8)
                if len(self._ai_cache) < AI_MAX_CACHE_SIZE:
                    self._ai_cache[cache_key] = result
                return result
        except Exception as e:
            self.logger.debug(f"AI single analysis failed: {str(e)}")
        return None, "", 0.0

    def check_contextual_patterns(self, text, keyword):
        for name, info in self.contextual_patterns.items():
            for pattern in info.get("patterns", []):
                if re.search(pattern, text, re.IGNORECASE):
                    return info["sentiment"], f"contextual:{name}", 0.8
        return None, "", 0.0

    # -------------------- SENTIMENT ANALYSIS --------------------

    def analyze_snippet_sentiment(self, snippet, keyword=""):
        """Analyze one snippet."""
        language = self.detect_language(snippet)

        # Check contextual
        ctx_sent, ctx_word, ctx_conf = self.check_contextual_patterns(snippet, keyword)
        if ctx_sent:
            return dict(sentiment=ctx_sent, sentiment_word=ctx_word, confidence=ctx_conf,
                        language=language, method="contextual")

        # Try AI
        ai_sent, ai_method, ai_conf = self.analyze_with_ai(snippet, keyword)
        if ai_sent:
            return dict(sentiment=ai_sent, sentiment_word=ai_method, confidence=ai_conf,
                        language=language, method="ai")

        # Lexicon fallback
        scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
        matched = []

        if language in ["Thai", "Mixed"]:
            for w, s in self.thai_sentiment_words.items():
                if w in snippet:
                    matched.append(w)
                    scores[s] += 1

        if language in ["English", "Mixed"]:
            for w in re.findall(r"\b\w+\b", snippet.lower()):
                if w in self.english_sentiment_words:
                    s = self.english_sentiment_words[w]
                    matched.append(w)
                    scores[s] += 1

        total = sum(scores.values())
        if not total:
            sentiment, conf = DEFAULT_SENTIMENT, 0.0
        else:
            sentiment = max(scores, key=scores.get)
            conf = scores[sentiment] / total

        return dict(sentiment=sentiment,
                    sentiment_word=", ".join(set(matched)),
                    confidence=conf,
                    language=language,
                    method="lexicon")

    # -------------------- FILE PROCESSING --------------------

    def collect_json_files(self):
        """Locate all *_sentences.json files."""
        json_files = []
        json_root = os.path.join(OUTPUT_FOLDER, JSON_FOLDER)
        if not os.path.exists(json_root):
            self.logger.error(f"JSON folder not found: {json_root}")
            return []
        for root, _, files in os.walk(json_root):
            for f in files:
                if f.endswith("_sentences.json"):
                    json_files.append(os.path.join(root, f))
        return json_files

    def process_json_file(self, filepath):
        """Analyze one JSON file and return summary/detail."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta, keywords = data.get("metadata", {}), data.get("keywords", {})
            year, sector, industry, business = (
                meta.get("year", ""), meta.get("sector", ""),
                meta.get("industry", ""), meta.get("business", "")
            )
            if not all([year, sector, industry, business]):
                return False, {"error": "Missing metadata"}

            ai_queue = [
                (s, k) for k, snippets in keywords.items()
                for s in snippets if s and 5 < len(s.strip()) < 500
                and any("\u0E00" <= c <= "\u0E7F" for c in s)
            ]

            ai_results = self.analyze_with_ai_batch(ai_queue) if ai_queue else {}
            details, summaries = [], defaultdict(Counter)

            for keyword, snippets in keywords.items():
                for snip in snippets or []:
                    if not snip.strip():
                        continue
                    ck = f"{keyword}:{hashlib.md5(snip.encode()).hexdigest()[:16]}"
                    sent, meth, conf, lang = "", "", 0.0, ""
                    if ck in ai_results:
                        sent, meth, conf = ai_results[ck]
                        lang = self.detect_language(snip)
                    else:
                        res = self.analyze_snippet_sentiment(snip, keyword)
                        sent, meth, conf, lang = (
                            res["sentiment"], res["sentiment_word"],
                            res["confidence"], res["language"]
                        )

                    details.append(dict(
                        year=year, sector=sector, industry=industry,
                        business=business, filename=f"{business}.pdf",
                        keyword=keyword, sentence=snip,
                        sentiment=sent, sentiment_word=meth,
                        confidence=round(conf, 3), language=lang, method=res.get("method", "lexicon")
                    ))
                    summaries[keyword][sent] += 1

            summary_list = []
            for kw, c in summaries.items():
                total = sum(c.values())
                if total:
                    summary_list.append(dict(
                        year=year, sector=sector, industry=industry,
                        business=business, filename=f"{business}.pdf",
                        keyword=kw, total_sentences=total,
                        positive_count=c.get("Positive", 0),
                        negative_count=c.get("Negative", 0),
                        neutral_count=c.get("Neutral", 0),
                        positive_ratio=round(c.get("Positive", 0) / total, 3),
                        negative_ratio=round(c.get("Negative", 0) / total, 3),
                        neutral_ratio=round(c.get("Neutral", 0) / total, 3),
                    ))

            return True, {"detail_entries": details, "summary_entries": summary_list}
        except Exception as e:
            return False, {"error": str(e)}

    # -------------------- OUTPUT --------------------

    def process_results(self, results):
        with self._lock:
            self.sentiment_details.extend(results["detail_entries"])
            self.sentiment_summaries.extend(results["summary_entries"])

    def save_sentiment_summary(self):
        if not self.sentiment_summaries:
            self.logger.warning("No summaries to save")
            return
        df = pd.DataFrame(self.sentiment_summaries).sort_values(
            ["year", "sector", "industry", "business", "total_sentences"],
            ascending=[True, True, True, True, False],
        )
        out_path = os.path.join(OUTPUT_FOLDER, SENTIMENT_SUMMARY_FILE)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Saved {len(df)} sentiment summary entries to {out_path}")

    def save_sentiment_details(self):
        if not self.sentiment_details:
            self.logger.warning("No details to save")
            return
        df = pd.DataFrame(self.sentiment_details).sort_values(
            ["year", "sector", "industry", "business", "keyword", "confidence"],
            ascending=[True, True, True, True, True, False],
        )
        out_path = os.path.join(OUTPUT_FOLDER, SENTIMENT_DETAIL_FILE)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Saved {len(df)} sentiment detail entries to {out_path}")

    # -------------------- MAIN --------------------

    def run(self):
        json_files = self.collect_json_files()
        if not json_files:
            self.logger.warning("No JSON files found")
            return

        self.logger.info(f"Processing {len(json_files)} JSON files...")
        if ENABLE_PARALLEL_PROCESSING and MAX_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(self.process_json_file, f) for f in json_files]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON"):
                    ok, res = fut.result()
                    if ok:
                        self.process_results(res)
                    else:
                        self.logger.warning(f"Failed: {res.get('error', 'Unknown')}")
        else:
            for f in tqdm(json_files, desc="Processing JSON"):
                ok, res = self.process_json_file(f)
                if ok:
                    self.process_results(res)
                else:
                    self.logger.warning(f"Failed: {res.get('error', 'Unknown')}")

        self.save_sentiment_summary()
        self.save_sentiment_details()


def main():
    analyzer = SentimentAnalyzer()
    try:
        analyzer.run()
    except Exception as e:
        print(f"Sentiment analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
