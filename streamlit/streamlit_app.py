import streamlit as st
import os
import time
import requests
import json
import re
import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Shiken Jirei - AI Test Case Generator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with dark/light theme support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: var(--text-color);
        opacity: 0.9;
        letter-spacing: 0.05em;
    }
    .metric-card {
        background-color: var(--background-color);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .test-case-card {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stTextArea > div > div > textarea {
        min-height: 150px;
        border-radius: 6px;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    .stSlider > div > div {
        border-radius: 6px;
    }
    .professional-section {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected {
        background-color: #10b981;
    }
    .status-disconnected {
        background-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []
if 'generation_in_progress' not in st.session_state:
    st.session_state.generation_in_progress = False

# Environment variables
HF_TOKEN = os.environ.get("HF_TOKEN1")
OPENROUTER_KEY = os.environ.get("opcode")

# API configurations
HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct:novita"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {"Authorization": f"Bearer {OPENROUTER_KEY}"}

# Initialize LanceDB and model
@st.cache_resource
def initialize_components():
    """Initialize LanceDB and SentenceTransformer model."""
    try:
        # Load SentenceTransformer model first
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Try to connect to LanceDB
        try:
            db = lancedb.connect('lancedb_data')
            table = db.open_table('testcases')
            return db, table, model
        except Exception as db_error:
            st.warning(f"LanceDB connection failed: {db_error}. RAG functionality will be limited.")
            return None, None, model
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None

def retrieve_context(user_story, top_k=5):
    """Retrieve relevant context from LanceDB for RAG."""
    db, table, model = initialize_components()
    
    if not model or not table:
        return ""
    
    try:
        query_embedding = model.encode(user_story).tolist()
        df_results = table.search(query_embedding).limit(top_k).to_pandas()
        
        if df_results.empty:
            return ""

        text_column = df_results.columns[0]
        context = "\n".join(df_results[text_column].tolist())
        return context
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return ""

def parse_structured_testcase(raw_text):
    """Extract structured fields from raw test case text."""
    title_match = re.search(r"Title[:\-]\s*(.*)", raw_text, re.IGNORECASE)
    precond_match = re.search(r"Preconditions[:\-]\s*(.*)", raw_text, re.IGNORECASE)
    expected_match = re.search(r"Expected Result[:\-]\s*(.*)", raw_text, re.IGNORECASE)
    steps = re.findall(r"^\s*\d+\.\s*(.*)", raw_text, re.MULTILINE)

    return {
        "Title": title_match.group(1).strip() if title_match else "",
        "Preconditions": precond_match.group(1).strip() if precond_match else "",
        "Steps": steps,
        "ExpectedResult": expected_match.group(1).strip() if expected_match else ""
    }

def create_excel_file(test_cases, language):
    """Create an Excel file from test cases data."""
    # Create a DataFrame with expanded structure for Excel
    excel_data = []
    
    for test_case in test_cases:
        # Create a row for each test case with all details
        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(test_case['Steps'])])
        
        excel_data.append({
            "Test Case ID": test_case['TestCaseID'],
            "Title": test_case['Title'],
            "Preconditions": test_case.get('Preconditions', ''),
            "Steps": steps_text,
            "Expected Result": test_case['ExpectedResult'],
            "Language": language.title()
        })
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Test Cases', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Test Cases']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.getvalue()

def generate_testcases_rag(user_story, language="english", total_cases=20, batch_size=5, delay=3):
    """Generate test cases using RAG approach."""
    all_testcases = []
    num_batches = (total_cases + batch_size - 1) // batch_size

    progress_bar = st.progress(0)
    status_text = st.empty()

    for batch_num in range(num_batches):
        status_text.text(f"Generating batch {batch_num+1}/{num_batches} ({language})...")
        progress_bar.progress((batch_num + 1) / num_batches)

        # RAG: retrieve context dynamically
        rag_context = retrieve_context(user_story, top_k=5)

        prompt = f"""
User Story: {user_story}

Relevant Context:
{rag_context}

Generate exactly {batch_size} unique test cases in {language}.

Each test case must follow this format:

Title: <short descriptive title>
Steps:
1. <step one>
2. <step two>
Expected Result: <expected outcome>

Start each test case with "Title:" on a new line.
Ensure exactly {batch_size} test cases per batch.
Do not include Test Case IDs in the content.
"""

        # Choose API based on language
        if language.strip().lower() == "english":
            endpoint = HF_ENDPOINT
            headers = HF_HEADERS
            model_id = HF_MODEL
        else:
            endpoint = OPENROUTER_ENDPOINT
            headers = OPENROUTER_HEADERS
            model_id = "google/gemma-3-12b-it:free"

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates structured software test cases."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600
        }

        # API request with retry
        output_text = ""
        for attempt in range(3):
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    st.warning(f"API error {response.status_code}: {response.text}")
                    time.sleep(2 ** attempt)
                    continue
                output_json = response.json()
                output_text = output_json["choices"][0]["message"]["content"].strip()
                if output_text:
                    break
            except Exception as e:
                st.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not output_text:
            st.warning(f"Batch {batch_num+1} failed. Skipping...")
            continue

        # Split raw test cases by Title keyword
        raw_cases = [c.strip() for c in re.split(r"\bTitle\b[:\-]", output_text) if c.strip()]

        # Ensure exactly batch_size test cases per batch
        if len(raw_cases) < batch_size:
            raw_cases += ["Title:Placeholder test case"] * (batch_size - len(raw_cases))
        raw_cases = raw_cases[:batch_size]

        # Parse and assign structured IDs
        for raw_case in raw_cases:
            if len(all_testcases) >= total_cases:
                break
            structured = parse_structured_testcase("Title:" + raw_case)
            structured["TestCaseID"] = f"TC-{len(all_testcases)+1}"
            all_testcases.append(structured)

        if batch_num < num_batches - 1:
            time.sleep(delay)

    progress_bar.empty()
    status_text.empty()
    return all_testcases

# Main app
def main():
    # Header with professional branding
    st.markdown('<div class="app-title">Shiken Jirei</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">AI Test Case Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["english", "hindi", "tamil", "telugu", "malayalam", "kannada"],
            index=0
        )
        
        # Number of test cases
        total_cases = st.slider("Number of Test Cases", 5, 50, 20)
        
        # Batch size
        batch_size = st.slider("Batch Size", 1, 10, 5)
        
        # API status
        st.header("API Status")
        hf_status = "Connected" if HF_TOKEN else "Missing Token"
        or_status = "Connected" if OPENROUTER_KEY else "Missing Key"
        
        hf_color = "status-connected" if HF_TOKEN else "status-disconnected"
        or_color = "status-connected" if OPENROUTER_KEY else "status-disconnected"
        
        st.markdown(f'<span class="status-indicator {hf_color}"></span>**Hugging Face:** {hf_status}', unsafe_allow_html=True)
        st.markdown(f'<span class="status-indicator {or_color}"></span>**OpenRouter:** {or_status}', unsafe_allow_html=True)
        
        if not HF_TOKEN or not OPENROUTER_KEY:
            st.error("Please configure your API keys in the .env file")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("User Story Input")
        user_story = st.text_area(
            "Enter your user story:",
            placeholder="As a registered student of an online e-learning platform, I want to browse courses, enroll in them, track my progress, complete quizzes, interact with instructors, and download course materials so that I can effectively learn new skills at my own pace...",
            height=150
        )
        
        # Generate button
        if st.button("Generate Test Cases", type="primary", disabled=st.session_state.generation_in_progress):
            if not user_story.strip():
                st.error("Please enter a user story")
            elif not HF_TOKEN or not OPENROUTER_KEY:
                st.error("Please configure your API keys in the .env file")
            else:
                st.session_state.generation_in_progress = True
                try:
                    with st.spinner("Generating test cases..."):
                        test_cases = generate_testcases_rag(
                            user_story=user_story,
                            language=language,
                            total_cases=total_cases,
                            batch_size=batch_size
                        )
                        st.session_state.test_cases = test_cases
                    st.success(f"Generated {len(test_cases)} test cases successfully!")
                except Exception as e:
                    st.error(f"Error generating test cases: {e}")
                finally:
                    st.session_state.generation_in_progress = False

    with col2:
        st.header("Statistics")
        if st.session_state.test_cases:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Total Test Cases", len(st.session_state.test_cases))
            with col2_2:
                st.metric("Language", language.title())
        else:
            st.info("Generate test cases to see statistics")

    # Results section
    if st.session_state.test_cases:
        st.markdown("---")
        st.header("Generated Test Cases")
        
        # Download buttons
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            if st.button("Download as Excel"):
                try:
                    excel_data = create_excel_file(st.session_state.test_cases, language)
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"Shiken_Jirei_TestCases_{language}_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error creating Excel file: {e}")
        
        with col_download2:
            json_data = json.dumps(st.session_state.test_cases, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name=f"Shiken_Jirei_TestCases_{language}_{int(time.time())}.json",
                mime="application/json"
            )
        
        # Display test cases in a table format
        for i, test_case in enumerate(st.session_state.test_cases):
            with st.expander(f"**{test_case['TestCaseID']}**: {test_case['Title']}", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Steps")
                    if test_case['Steps']:
                        for j, step in enumerate(test_case['Steps'], 1):
                            st.write(f"{j}. {step}")
                    else:
                        st.write("No steps provided")
                
                with col2:
                    st.subheader("Expected Result")
                    st.write(test_case['ExpectedResult'] if test_case['ExpectedResult'] else "No expected result provided")
                    
                    if test_case.get('Preconditions'):
                        st.subheader("Preconditions")
                        st.write(test_case['Preconditions'])

        # Alternative table view
        st.markdown("---")
        st.header("Table View")
        
        # Create DataFrame for table display
        table_data = []
        for test_case in st.session_state.test_cases:
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(test_case['Steps'])])
            table_data.append({
                "Test Case ID": test_case['TestCaseID'],
                "Title": test_case['Title'],
                "Steps": steps_text,
                "Expected Result": test_case['ExpectedResult']
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', height=400)

if __name__ == "__main__":
    main()
