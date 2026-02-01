#  Shiken Jirei - AI Test Case Generator

An intelligent, RAG-powered Streamlit application that generates comprehensive test cases from user stories in multiple languages using advanced LLMs and vector embeddings.

##  Features

-  **Generate test cases** from user stories using AI
-  **Multi-language support**: English, Hindi, Tamil, Telugu, Malayalam, Kannada
-  **RAG (Retrieval-Augmented Generation)**: Context-aware generation using stored embeddings
-  **LanceDB integration**: Vector store for semantic search and retrieval
-  **Professional UI**: Built with Streamlit for ease of use
-  **Export options**: Download test cases as JSON or Excel (XLSX)
-  **Batch processing**: Configurable batch sizes for test case generation
-  **Data ingestion pipeline**: Extract text from PDFs, DOCX, images, CSV, XLSX files

##  Project Structure

```
.
 streamlit/
    streamlit_app.py          # Main Streamlit web application
 llm/
    run_app.py                # Startup script with environment checks
    datapipeline.py           # Data ingestion & LanceDB embedding pipeline
    check_database.py          # Utility to inspect LanceDB table contents
 lancedb_data/
    testcases.lance/          # LanceDB vector index (RAG data)
 requirements.txt               # Python package dependencies
 run_app.py                     # Convenience wrapper to launch the app
 .env.example                   # Environment variables template
 .gitignore                     # Git ignore rules
 README.md                      # This file
```

##  File Descriptions

### Core Application

**`streamlit/streamlit_app.py`**
- **Purpose**: Main web UI for the test case generator
- **Key functions**:
  - `initialize_components()`: Load SentenceTransformer and connect to LanceDB
  - `retrieve_context()`: RAG function to fetch relevant examples from LanceDB
  - `generate_testcases_rag()`: Generate test cases using LLM with RAG context
  - `create_excel_file()`: Export test cases to XLSX format
  - `parse_structured_testcase()`: Parse raw LLM output into structured test cases
- **Dependencies**: streamlit, lancedb, sentence_transformers, requests, pandas, openpyxl

### Backend & Utilities

**`llm/run_app.py`**
- **Purpose**: Application launcher with dependency and environment checks
- **Checks**:
  - Verifies API keys (`HF_TOKEN1`, `opcode`) are set
  - Validates Python packages are installed
  - Locates and runs `streamlit/streamlit_app.py`
- **Usage**: `python run_app.py` from repo root

**`llm/datapipeline.py`**
- **Purpose**: Data ingestion pipeline for building the RAG database
- **Processes**:
  - Extracts text from multiple file types (PDF, DOCX, TXT, images, CSV, XLSX)
  - Chunks text into semantic segments (500-word chunks with 50-word overlap)
  - Generates embeddings using `SentenceTransformer` (all-mpnet-base-v2)
  - Stores embeddings in LanceDB `testcases` table
- **Folder handling**: input/, success/, failure/ for file organization
- **Usage**: `python llm/datapipeline.py` (requires files in `input/` folder)

**`llm/check_database.py`**
- **Purpose**: Quick diagnostics for LanceDB
- **Shows**:
  - Number of rows in `testcases` table
  - Column names and data types
  - NaN and empty value counts
- **Usage**: `python llm/check_database.py`

### Configuration & Dependencies

**`requirements.txt`**
- Lists all Python package dependencies
- Key packages:
  - `streamlit`: Web UI framework
  - `lancedb`: Vector database
  - `sentence-transformers`: Embeddings model
  - `transformers`: LLM model libraries
  - `requests`: HTTP client for API calls
  - `pandas`, `openpyxl`: Data processing & Excel export
  - `PyPDF2`, `python-docx`, `Pillow`: File parsing

**`run_app.py`**
- Convenience wrapper that calls `llm/run_app.main()`
- Simplifies startup: `python run_app.py`

**`.env.example`**
- Template for required environment variables
- Must be copied to `.env` and filled with real values:
  ```
  HF_TOKEN1=your_huggingface_api_key
  opcode=your_openrouter_api_key
  ```

**`lancedb_data/testcases.lance/`**
- LanceDB vector index containing stored test case embeddings
- Used by RAG retrieval to find contextually relevant examples
- Pre-populated with sample test cases for immediate use
- Can be extended by running `llm/datapipeline.py`

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Carol-here/Testcase-Generation-using-RAG-based-AI.git
cd Testcase-Generation-using-RAG-based-AI
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```powershell
copy .env.example .env
# Edit .env and add your API keys:
#   HF_TOKEN1 = Your Hugging Face API token (https://huggingface.co/settings/tokens)
#   opcode = Your OpenRouter API key (https://openrouter.ai/keys)
```

### 5. Run the Application

**Option A (Recommended):**
```powershell
python run_app.py
```

**Option B (Direct):**
```powershell
streamlit run streamlit/streamlit_app.py
```

The app will open at `http://localhost:8501`

##  How to Use

1. **Select Language**: Choose from English, Hindi, Tamil, Telugu, Malayalam, Kannada
2. **Configure Generation**: Set number of test cases (5-50) and batch size (1-10)
3. **Enter User Story**: Paste a user story or feature description in the text area
4. **Generate**: Click "Generate Test Cases" to start AI-powered generation
5. **Export**: Download results as JSON or Excel

##  RAG (Retrieval-Augmented Generation) Workflow

1. User enters a user story
2. `retrieve_context()` encodes the story and searches LanceDB for similar test cases
3. Top-5 relevant test cases are retrieved from the vector index
4. LLM is prompted with both the user story AND the retrieved context
5. LLM generates new, contextually-aware test cases
6. Results are parsed, formatted, and displayed

##  Tech Stack

### Core Framework
- **Streamlit** (v1.28.1): Web application framework
- **Python 3.10+**: Programming language

### Machine Learning & NLP
- **Transformers** (v4.56.0): Large language models (HuggingFace)
- **Sentence Transformers** (v5.1.0): Text embeddings (`all-mpnet-base-v2`)
- **PyTorch** (v2.8.0): Deep learning backend
- **Scikit-learn** (v1.7.1): ML utilities
- **NumPy** (v2.2.6): Numerical computing

### Vector Database
- **LanceDB** (v0.24.3): Vector index for RAG

### Data Processing
- **Pandas** (v2.0.3): Data manipulation
- **PyArrow** (v21.0.0): Data serialization
- **OpenPyXL** (v3.1.2): Excel file handling
- **PyPDF2** (v3.0.1): PDF parsing
- **python-docx** (v1.2.0): DOCX parsing
- **Pillow** (v11.3.0): Image processing
- **pytesseract**: OCR for image text extraction

### APIs & External Services
- **Hugging Face API**: English test case generation (Llama 3.2)
- **OpenRouter API**: Multi-language LLM access

### Development & Utilities
- **python-dotenv** (v1.0.0): Environment variable management
- **Requests** (v2.32.5): HTTP client for API calls

##  Future Enhancements (Roadmap)

### Near-term (v1.1)
- [ ] **Custom embeddings model selection**: Allow users to choose embedding models
- [ ] **Database persistence UI**: Add admin panel to view/manage LanceDB
- [ ] **Test case templates**: Provide domain-specific templates (Web, Mobile, API, etc.)
- [ ] **Batch upload**: Support uploading multiple user stories at once
- [ ] **Test case evaluation**: Add scoring/rating of generated test cases

### Medium-term (v1.2)
- [ ] **Multi-user collaboration**: Shared workspaces and version control
- [ ] **Custom LLM models**: Support for local or self-hosted LLMs
- [ ] **Test case validation**: Automated checks for duplicate/conflicting test cases
- [ ] **Analytics dashboard**: Track generation metrics, success rates, language performance
- [ ] **API endpoint**: REST API for programmatic access

### Long-term (v2.0)
- [ ] **Test execution integration**: Connect to testing frameworks (Selenium, Cypress, pytest)
- [ ] **Automated test generation**: AI-driven test code generation (Gherkin/BDD)
- [ ] **Requirements traceability**: Link test cases to requirements documents
- [ ] **ML model fine-tuning**: Domain-specific model training for improved quality
- [ ] **Multi-modal support**: Accept diagrams, screenshots, and video as input

##  Configuration & Environment Variables

Required environment variables (set in `.env`):

```env
# Hugging Face API (for English test case generation)
HF_TOKEN1=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter API (for multi-language support)
opcode=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Optional environment variables:

```env
# LanceDB path (default: lancedb_data)
LANCEDB_PATH=lancedb_data

# Streamlit port (default: 8501)
STREAMLIT_SERVER_PORT=8501
```

##  Troubleshooting

### App won't start
-  Check all dependencies: `pip install -r requirements.txt`
-  Verify `.env` file exists with valid API keys
-  Ensure you're running from the repo root directory

### LanceDB connection errors
-  Verify `lancedb_data/testcases.lance/` directory exists
-  Run `python llm/check_database.py` to diagnose
-  If corrupted, run `python llm/datapipeline.py` to rebuild

### API errors (Hugging Face / OpenRouter)
-  Verify API keys in `.env` are correct and active
-  Check internet connection
-  Review API quota/rate limits

### Test case generation is slow
-  Reduce batch size to 1-3 to make API calls smaller
-  Reduce number of test cases to 5-10
-  Check internet connection speed

##  License

This project is provided as-is for educational and development purposes.

##  Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the maintainer.

---

**Built with  for intelligent test case generation**
