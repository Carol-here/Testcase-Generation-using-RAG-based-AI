# AI Test Case Generator

A professional Streamlit application that generates comprehensive test cases from user stories using AI and RAG (Retrieval-Augmented Generation).

## Features

- 🎯 Generate test cases from user stories
- 🌍 Support for multiple languages (English, Hindi, Tamil, Telugu, Malayalam, Kannada)
- 🤖 AI-powered using Hugging Face and OpenRouter APIs
- 📊 RAG-based context retrieval from LanceDB
- 💻 Professional Streamlit web interface
- 📋 Beautiful table display of test cases
- 📥 Download generated test cases as JSON
- ⚙️ Configurable parameters (batch size, number of cases)

## Project Structure

```
├── streamlit_app.py       # Main Streamlit application
├── run_app.py            # Application startup script
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── RAG.ipynb           # Original Jupyter notebook (DO NOT MODIFY)
└── lancedb_data/       # LanceDB vector database
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables

```bash
cp env.example .env
```

Edit `.env` and add your API keys:
```
HF_TOKEN1=your_huggingface_token
opcode=your_openrouter_key
```

### 3. Run the Application

```bash
python run_app.py
```

Or directly:
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Open the application** in your browser at `http://localhost:8501`

2. **Configure settings** in the sidebar:
   - Select language for test case generation
   - Set number of test cases (5-50)
   - Adjust batch size (1-10)

3. **Enter a user story** in the main text area

4. **Click "Generate Test Cases"** to create test cases using AI

5. **View results** in both expandable cards and table format

6. **Download the results** as a JSON file

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and table display
- **LanceDB** - Vector database for RAG
- **Sentence Transformers** - Text embeddings
- **Hugging Face API** - English test case generation
- **OpenRouter API** - Multi-language test case generation

## Important Notes

- **DO NOT MODIFY** the `RAG.ipynb` file as requested
- The backend uses the same LLM logic from the Jupyter notebook
- LanceDB connection is required for RAG functionality
- API keys are required for both Hugging Face and OpenRouter services

## Troubleshooting

1. **App won't start:**
   - Check if all dependencies are installed: `pip install -r requirements.txt`
   - Verify environment variables are set correctly
   - Ensure LanceDB data directory exists

2. **Test case generation fails:**
   - Check API key validity in .env file
   - Verify internet connection
   - Check Streamlit logs for detailed error messages
   - Ensure both HF_TOKEN1 and opcode are set

3. **LanceDB connection issues:**
   - Verify lancedb_data directory exists
   - Check if testcases table is properly initialized
   - Ensure SentenceTransformer model can load
