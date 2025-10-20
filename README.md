# Thai-Business-NLP: Python Package for Thai–English Keyword & Sentiment Analysis

[![Author](https://img.shields.io/badge/Pattawee.P-blue?label=Author)](https://bodysbobb.github.io/)
![Last Updated](https://img.shields.io/github/last-commit/Bodysbobb/thai-business-nlp?label=Last%20Updated&color=blue)
[![Stars](https://img.shields.io/github/stars/Bodysbobb/thai-business-nlp?style=social)](https://github.com/Bodysbobb/thai-business-nlp)

[![GitHub Version](https://img.shields.io/github/v/tag/Bodysbobb/thai-business-nlp?label=GitHub%20Version&color=3CB371&sort=semver)](https://github.com/Bodysbobb/thai-business-nlp/releases/latest)
![License](https://img.shields.io/github/license/Bodysbobb/thai-business-nlp?color=3CB371)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

## **Overview**

**Thai-Business-NLP** is a Python package for automated **Thai–English keyword extraction** and **sentiment analysis** from business-related PDF reports.  
The sample dataset is based on annual company reports from the **Stock Exchange of Thailand (SET)** — for example, [PTT’s Annual Report](https://www.set.or.th/en/market/product/stock/quote/PTT/company-profile/information).  
This toolkit combines rule-based and AI-assisted natural language processing, designed for researchers, analysts, and corporate users working with bilingual Thai–English business documents.

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

## **Input Folder levels (samples/in):**

* All input PDFs **must** be placed under `samples/in/` following this hierarchy.  
The program will automatically **scan and analyze every PDF** in the input folder and its subdirectories in a single run:

| Level | Folder Name | Purpose |
|--------|--------------|----------|
| **1. Year** | Example: `2023/`, `2024/` | Organizes PDFs by publication or reporting year. Used for time-based analysis. |
| **2. Sector** | Example: `Energy/`, `Finance/`, `Manufacturing/` | Defines the main business sector. Must match the sector names listed in `keywords.xlsx`. |
| **3. Industry** | Example: `OilGas/`, `Banking/`, `Textile/` | Sub-industry or category under each sector. Allows finer grouping of PDFs. |
| **File** | Example: `PTT.pdf`, `SCB.pdf` | Each file represents one company or report to be analyzed. |

---

## **Mapping Files (data/):**

These are the core reference files that control how the program interprets, filters, and classifies content during keyword extraction and sentiment analysis.  
You can modify these files to tailor the analysis to your own dataset, sectors, or business context.

| File | Description |
|------|--------------|
| **keywords.xlsx** | Core mapping file linking **sectors**, **Thai keywords**, **English keywords**, and **ESG categories**. Used to identify predefined terms in each business sector during keyword extraction (see detail below). |
| **sentiment.json** | Dictionary of **positive**, **negative**, and **neutral** words in both Thai and English. Used for lexicon-based sentiment classification. |
| **stopwords.json** | Common Thai and English stopwords (e.g., “และ”, “the”, “of”) automatically filtered out from text to prevent false keyword detection. |
| **contextual_patterns.json** | Regex-based rules that detect sentiment from contextual expressions (e.g., “ลดต้นทุน” → Positive, “เพิ่มต้นทุน” → Negative). Enhances sentiment accuracy beyond simple word matching. |
| **excluded_keywords.json** | List of generic or non-informative words (e.g., “บริษัท”, “manager”) that are **explicitly excluded** from keyword extraction to keep results relevant. |

### **keywords.xlsx**

This Excel file defines the **sector-specific keyword mapping** used for keyword extraction and sentiment analysis.  
Each row links an English and Thai keyword to its **ESG pillar** (Environmental, Social, or Governance) and assigns it to a specific **Sector**.

| Column | Description |
|---------|--------------|
| **English** | The English form of the keyword to be detected in PDF text. |
| **Thai** | The Thai equivalent of the keyword. Both English and Thai forms are recognized during extraction. |
| **ESG** | ESG pillar category for the keyword — use `E` (Environmental), `S` (Social), or `G` (Governance). |
| **Sector** | The business sector to which this keyword belongs. **This value must exactly match** the *Sector folder name* in the input directory (`samples/in/<Year>/<Sector>/`). Only keywords assigned to that sector will be used when analyzing PDFs within that sector. |

In summary, `keywords.xlsx` determines **which keywords are applied to which sector**, ensuring that the extraction process only uses relevant terms for each industry group.

---

## **Output Structure**

After running the pipeline, all results are automatically saved in the **output folder** (`samples/out/`).  
Each output file represents a different level of processed information derived from the folder hierarchy of your input PDFs (`samples/in/`).

```
out/
├─ keywords_count.csv           # Keyword frequency summary
├─ sentiment_summary.csv        # Aggregate sentiment per keyword
├─ sentiment_detail.csv         # Sentence-level sentiment breakdown
└─ keyword_sentences/           # Individual JSON files by document
```

### **Explanation of Output Files**

| File | Description | Input-Level Dependency |
|------|---------------------------|----------------------|
| **keywords_count.csv** | Contains all extracted keywords (predefined and auto-detected) with frequency counts across company reports. | Tagged with **Year**, **Sector**, **Industry**, and **Business**, derived from folder structure (`/in/<Year>/<Sector>/<Industry>/<File.pdf>`). |
| **sentiment_summary.csv** | Aggregated sentiment ratios (**Positive**, **Negative**, **Neutral**) for each keyword within a report. Useful for comparing sentiment across sectors or years. | Uses the same hierarchy (**Year → Sector → Industry → Business**). |
| **sentiment_detail.csv** | Sentence-level results showing which sentences triggered each keyword and its sentiment classification. | Directly linked to the same metadata for traceability to each PDF. |
| **keyword_sentences/** | Folder of individual `.json` files (one per company/report) containing all detected keywords and related text snippets. | Mirrors input structure: `/out/keyword_sentences/<Year>/<Sector>/<Industry>/<Business>_sentences.json`. |


### **How Input Levels Define Output**

The system **automatically reads the folder names** in your input directory to assign metadata to each PDF:

```
in/<Year>/<Sector>/<Industry>/<File.pdf>
```

From this structure:
- `Year` → used for time-based grouping and analysis.  
- `Sector` → matched with the `Sector` column in `keywords.xlsx` to load relevant predefined keywords.  
- `Industry` → used to further categorize results within each sector.  
- `File.pdf` → defines the **Business** or company name, carried over to all output files.

This ensures that every entry in the output (`csv` and `json`) is properly identified by its **source year**, **sector**, **industry**, and **company**, maintaining a consistent analytical hierarchy between input and output data.

---

## **Quick Run**

Run the full pipeline with a single command:

```bash
python -m thai_business_nlp.config
```

This command will:

1. Extract Thai–English keywords from PDF files under the **input folder**.
2. Perform sentiment analysis using both rule-based and AI methods (if enabled).
3. Export summarized results to the **output folder**.

---

## **Configuration Guide**

All parameters are defined in `config.py`, which **you are strongly encouraged to download and modify** to suit your setup.  
Users can adjust these settings to control folder paths, processing behavior, and model configurations.

| Category                | Key Variables                                          | Description                                                               |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| **Paths**               | `INPUT_FOLDER`, `OUTPUT_FOLDER`, `MAP_FOLDER`          | Define where input PDFs, output files, and mapping resources are located. |
| **OCR**                 | `TESSERACT_PATH`, `USE_OCR_FALLBACK`                   | Set path to Tesseract and enable OCR fallback for scanned PDFs.           |
| **AI Model**            | `AI_MODEL_URL`, `AI_MODEL_NAME`, `ENABLE_AI_SENTIMENT` | Enable or disable AI-based sentiment classification (via Ollama).         |
| **Parallel Processing** | `ENABLE_PARALLEL_PROCESSING`, `MAX_WORKERS`            | Control multi-threaded document processing for performance.               |
| **Sentiment**           | `DEFAULT_SENTIMENT`, `SENTIMENT_CONFIDENCE_THRESHOLD`  | Adjust sentiment classification defaults and thresholds.                  |
| **Logging**             | `LOG_LEVEL`                                            | Control verbosity of logs (DEBUG, INFO, WARNING, ERROR).                  |

---

## **AI Sentiment Analysis (Optional — but Recommended)**

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
