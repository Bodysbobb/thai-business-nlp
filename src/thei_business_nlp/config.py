"""
Thai-English PDF Keyword & Sentiment Analysis Pipeline Configuration

Author: Pattawee Puangchit
Version: 0.1.0
Date: 2025-10-19
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# ===================== PATHS & FILES =====================
# Input and output folder paths (user should edit these before running)
INPUT_FOLDER = r"D:\NLP_Sentiment\in"                       # Folder containing PDF files
OUTPUT_FOLDER = r"D:\NLP_Sentiment\out"                     # Folder for output CSV/JSON/log files
MAP_FOLDER = r"D:\NLP_Sentiment\map"                        # Folder containing mapping and dictionary files
JSON_FOLDER = "keyword_sentences"                           # Folder name for saving extracted keyword sentences

# ===================== MAPPING FILE NAMES =====================
KEYWORD_FILE = os.path.join(MAP_FOLDER, "keywords.xlsx")              # Keyword-sector mapping Excel
SENTIMENT_LEXICON_FILE = os.path.join(MAP_FOLDER, "sentiment.json")   # Sentiment lexicon (positive/negative/neutral)
CONTEXTUAL_PATTERNS_FILE = os.path.join(MAP_FOLDER, "contextual_patterns.json")  # Regex-based sentiment patterns
STOPWORDS_FILE = os.path.join(MAP_FOLDER, "stopwords.json")           # Stopwords (Thai/English)
EXCLUDED_KEYWORDS_FILE = os.path.join(MAP_FOLDER, "excluded_keywords.json")      # Words to exclude

# ===================== MAPPING SHEET NAMES =====================
KEYWORD_SHEET = "SectorKeyword"                            # Sheet name in keyword Excel
KEYWORDS_COUNT_FILE = "keywords_count.csv"                 # Output file for keyword frequency summary
SENTIMENT_SUMMARY_FILE = "sentiment_summary.csv"           # Output file for sentiment summary
SENTIMENT_DETAIL_FILE = "sentiment_detail.csv"             # Output file for sentence-level sentiment details

# ===================== PROCESSING SETTINGS =====================
ENABLE_PARALLEL_PROCESSING = True                          # Enable multi-threaded processing for PDFs
MAX_WORKERS = 4                                            # Number of worker threads
LANGUAGE_MODE = "Both"                                     # Process "Thai", "Eng", or "Both"
MAX_AUTO_KEYWORDS = 0                                      # Max number of auto keywords (0 = unlimited)
MIN_WORD_FREQUENCY = 2                                     # Minimum frequency for a word to count as keyword
MIN_WORD_LENGTH = 3                                        # Minimum word length
MIN_TEXT_LENGTH = 10                                       # Minimum text length for valid content
MAX_SENTENCES_PER_KEYWORD = 20                             # Max sentences to save per keyword
WINDOW_WORDS_BEFORE = 10                                   # Context words before keyword (for snippet)
WINDOW_WORDS_AFTER = 10                                    # Context words after keyword (for snippet)
VALIDATE_SENTENCE_WITH_THAI_NLP = True                     # Use Thai NLP for sentence segmentation

# ===================== OCR SETTINGS =====================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"      # Path to Tesseract OCR executable
USE_OCR_FALLBACK = True                                     # Use OCR if text extraction fails
OCR_RESOLUTION = 300                                        # OCR image resolution (higher = slower)
OCR_LANGUAGES = "tha+eng"                                   # OCR languages (Thai + English)

# ===================== FILTERS =====================
FILTER_YEARS = []                                           # Filter by year (empty = all years)
FILTER_SECTORS = []                                         # Filter by sector (empty = all)
FILTER_INDUSTRIES = []                                      # Filter by industry (empty = all)

# ===================== WORD VALIDATION =====================
ENABLE_WORD_VALIDATION = False                              # Validate Thai words using dictionary (slower)

# ===================== SENTIMENT SETTINGS =====================
DEFAULT_SENTIMENT = "Neutral"                              # Default sentiment if none detected
SENTIMENT_CONFIDENCE_THRESHOLD = 0.6                       # Minimum confidence threshold for sentiment classification

# ===================== AI SENTIMENT ANALYSIS =====================
ENABLE_AI_SENTIMENT = True                                  # Enable AI-powered sentiment analysis
AI_MODEL_URL = "http://localhost:11434/api/generate"        # Ollama API endpoint
AI_MODEL_NAME = "llama3.2:1b"                              # Model name (:1b = fast, :3b = balanced, :8b = accurate)
AI_TIMEOUT = 3                                              # Timeout (seconds) per request
AI_TEMPERATURE = 0.0                                        # Randomness in AI response (0.0 = deterministic)
AI_MAX_CACHE_SIZE = 10000                                   # Max cached responses
AI_MIN_TEXT_LENGTH = 20                                     # Min text length to send to AI
AI_USE_PARALLEL = True                                      # Run AI calls in parallel (faster)
AI_BATCH_SIZE = 10                                          # Sentences per batch for AI analysis
AI_BATCH_TIMEOUT = 15                                       # Timeout per batch request
AI_MAX_TEXT_LENGTH = 500                                    # Max text length allowed for AI
AI_FOCUS_THAI_ONLY = True                                   # Use AI only for Thai text
AI_NUM_PREDICT = 50                                         # Max tokens to generate in AI response
AI_TOP_P = 0.1                                              # Sampling focus (lower = more precise)

# ===================== LOGGING =====================
LOG_LEVEL = "INFO"                                          # Logging level (DEBUG/INFO/WARNING/ERROR)

# ===================== PIPELINE CONTROL =====================
RUN_PIPELINE = True                                         # Run full pipeline when executing config.py

# ===================== FUNCTIONS =====================
def ensure_directories():
    """Create output directories if they do not exist."""
    dirs = [
        OUTPUT_FOLDER,
        os.path.join(OUTPUT_FOLDER, "logs"),
        os.path.join(OUTPUT_FOLDER, JSON_FOLDER)
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_log_file():
    """Return timestamped log file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_FOLDER, "logs", f"keyword_analysis_{timestamp}.log")


# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    if RUN_PIPELINE:
        print("🚀 Starting Thai-English PDF Keyword & Sentiment Analysis Pipeline")

        # Add current directory to system path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        try:
            import keyword_extract
            import sentiment_analysis

            # --- Step 1: Keyword extraction ---
            print("📄 Running keyword extraction...")
            keyword_extract.main()

            # --- Step 2: Sentiment analysis ---
            print("🎭 Running sentiment analysis...")
            sentiment_analysis.main()

            print("✅ Pipeline completed successfully!")

        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            sys.exit(1)
