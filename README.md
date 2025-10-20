````markdown
# Thai-Business-NLP: Python Package for Thai–English Keyword & Sentiment Analysis

[![Author](https://img.shields.io/badge/Pattawee.P-blue?label=Author)](https://bodysbobb.github.io/)
![Last Updated](https://img.shields.io/github/last-commit/Bodysbobb/thai-business-nlp?label=Last%20Updated&color=blue)
[![Stars](https://img.shields.io/github/stars/Bodysbobb/thai-business-nlp?style=social)](https://github.com/Bodysbobb/thai-business-nlp)

[![GitHub Version](https://img.shields.io/github/v/tag/Bodysbobb/thai-business-nlp?label=GitHub%20Version&color=3CB371&sort=semver)](https://github.com/Bodysbobb/thai-business-nlp/releases/latest)
![License](https://img.shields.io/github/license/Bodysbobb/thai-business-nlp?color=3CB371)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

## **Overview**

**Thai-Business-NLP** is a Python package for automatic **Thai–English keyword extraction** and **sentiment analysis** from business-related PDF reports.  
It combines rule-based and AI-assisted natural language processing, designed for researchers, analysts, and corporate users working with bilingual business texts.

---

## **Key Features**

* Extracts keywords and contextual sentences in both Thai and English.
* Performs sentiment classification using:
  * Lexicon-based rules
  * Regex contextual patterns
  * Optional AI models via [Ollama](https://ollama.ai)
* Supports parallel PDF processing for speed and scalability.
* Includes OCR fallback using Tesseract for scanned PDFs.
* Outputs clean CSV and JSON summaries for downstream analysis.
* Designed for easy setup via a single `config.py` configuration file.

---

## **Installation**

### Install directly from GitHub

```bash
pip install git+https://github.com/Bodysbobb/thai-business-nlp.git
````

### For local editable installation (recommended for customization)

```bash
git clone https://github.com/Bodysbobb/thai-business-nlp.git
cd thai-business-nlp
pip install -e .
```

---

## **Project Structure**

After installation or cloning, the default project structure is:

```
thai-business-nlp/
 └── src/
      ├── thai_business_nlp/
      │   ├── config.py
      │   ├── keyword_extract.py
      │   ├── sentiment_analysis.py
      │   └── data/
      │        ├── keywords.xlsx
      │        ├── sentiment.json
      │        ├── stopwords.json
      │        ├── contextual_patterns.json
      │        ├── excluded_keywords.json
      │        └── samples/
      │             ├── in/    ← Place your input PDF files here
      │             └── out/   ← Output CSV and JSON files will be generated here
```

**Folder levels explained:**

| Folder Level         | Path Example                              | Purpose                                                                                               |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Level 1**          | `src/thai_business_nlp/data/`             | Stores all mapping and dictionary files (e.g., keywords.xlsx, sentiment.json).                        |
| **Level 2**          | `src/thai_business_nlp/data/samples/`     | Contains subfolders for input and output processing.                                                  |
| **Level 3 (Input)**  | `src/thai_business_nlp/data/samples/in/`  | Place PDF files here. Each PDF will be processed for keyword extraction and sentiment classification. |
| **Level 3 (Output)** | `src/thai_business_nlp/data/samples/out/` | The program automatically saves output results here (CSV and JSON).                                   |

---

## **Quick Start**

Run the full pipeline with a single command:

```bash
python -m thai_business_nlp.config
```

This command will:

1. Extract Thai–English keywords from PDF files under the **input folder**.
2. Perform sentiment analysis using both rule-based and AI methods (if enabled).
3. Export summarized results to the **output folder**.

### Example output structure

```
out/
├─ keywords_count.csv           # Keyword frequency summary
├─ sentiment_summary.csv        # Aggregate sentiment per keyword
├─ sentiment_detail.csv         # Sentence-level sentiment breakdown
└─ keyword_sentences/           # Individual JSON files by document
```

---

## **Configuration Guide**

All parameters are stored in `config.py`.
Users can modify them to control folder locations, processing modes, and model behavior.

| Category                | Key Variables                                          | Description                                                               |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| **Paths**               | `INPUT_FOLDER`, `OUTPUT_FOLDER`, `MAP_FOLDER`          | Define where input PDFs, output files, and mapping resources are located. |
| **OCR**                 | `TESSERACT_PATH`, `USE_OCR_FALLBACK`                   | Set path to Tesseract and enable OCR fallback for scanned PDFs.           |
| **AI Model**            | `AI_MODEL_URL`, `AI_MODEL_NAME`, `ENABLE_AI_SENTIMENT` | Enable or disable AI-based sentiment classification (via Ollama).         |
| **Parallel Processing** | `ENABLE_PARALLEL_PROCESSING`, `MAX_WORKERS`            | Control multi-threaded document processing for performance.               |
| **Sentiment**           | `DEFAULT_SENTIMENT`, `SENTIMENT_CONFIDENCE_THRESHOLD`  | Adjust sentiment classification defaults and thresholds.                  |
| **Logging**             | `LOG_LEVEL`                                            | Control verbosity of logs (DEBUG, INFO, WARNING, ERROR).                  |

---

## **AI Sentiment Analysis (Optional)**

To enable AI-powered sentiment analysis via **Ollama**:

1. Install and start Ollama locally:

   ```bash
   ollama run llama3.2:1b
   ```
2. Update your configuration in `config.py`:

   ```python
   ENABLE_AI_SENTIMENT = True
   AI_MODEL_NAME = "llama3.2:1b"
   AI_MODEL_URL = "http://localhost:11434/api/generate"
   ```

If disabled, the package automatically uses lexicon-based and contextual rule sentiment analysis.

---

## **Dependencies**

* `pandas`, `numpy`, `openpyxl`
* `pythainlp`, `tqdm`
* `pymupdf`, `pillow`, `opencv-python`
* `requests`
* Optional: `Tesseract OCR`, `Ollama` (for AI models)

---

## **Local Testing**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m thai_business_nlp.config
```

Expected output:

```bash
🚀 Starting Thai-English PDF Keyword & Sentiment Analysis Pipeline
📄 Running keyword extraction...
🎭 Running sentiment analysis...
✅ Pipeline completed successfully!
```

---

## **License & Author**

**License:** MIT License
**Author:** Pattawee Puangchit
Ph.D. Candidate, Agricultural Economics
Purdue University | Research Assistant at GTAP

🔗 [Website](https://bodysbobb.github.io/) • [LinkedIn](https://www.linkedin.com/in/pattawee-puangchit/)

---

## **Related Projects**

* [HARplus](https://github.com/Bodysbobb/HARplus) – Enhanced R Package for GEMPACK `.har` Files
* [GTAPViz](https://github.com/Bodysbobb/GTAPViz) – Visualization Toolkit for GTAP Simulation Results

---

✅ **Usage Tip:**
- Modify only the folder paths in `config.py` to match your local directory.
- Place your input PDFs under `data/samples/in`, and the pipeline will handle extraction and sentiment analysis automatically.
