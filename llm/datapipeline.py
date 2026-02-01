import os
import shutil
import uuid
import pytesseract
from PIL import Image
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import lancedb
import openpyxl
import pyarrow as pa
import pandas as pd

# ====== CONFIG ======
INPUT_FOLDER = "input"
SUCCESS_FOLDER = "success"
FAILURE_FOLDER = "failure"
LANCE_DB_PATH = "lancedb_data"
TABLE_NAME = "testcases"

# Choose your model
model = SentenceTransformer('all-mpnet-base-v2')
# ====================
# ----------------------------
# Utility: Chunk text
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of words with overlap for context.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
# ----------------------------
# Extract text from files
# ----------------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        # split per page
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # also split by paragraphs
                chunks.extend([p.strip() for p in text.split("\n") if p.strip()])
        return chunks

    elif ext == ".docx":
        doc = Document(file_path)
        return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return [line.strip() for line in text.split("\n") if line.strip()]
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        chunks = []
        for row in df.itertuples(index=False):
            row_text = [str(cell) for cell in row if pd.notna(cell)]
            if row_text:
                chunks.append(" ".join(row_text))
        return chunks

    elif ext == ".xlsx":
        wb = openpyxl.load_workbook(file_path, data_only=True)
        chunks = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = [str(cell) for cell in row if cell is not None]
                if row_text:
                    chunks.append(" ".join(row_text))
        return chunks

    else:
        raise ValueError(f"Unsupported file type: {ext}")
# ----------------------------
# Embed text
# ----------------------------
def embed_text(text):
    return model.encode(text).tolist()

# ----------------------------
# Connect to LanceDB
# ----------------------------
def connect_to_lancedb():
    db = lancedb.connect(LANCE_DB_PATH)
    
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    
    schema = pa.schema([
        ("id", pa.string()),
        ("text", pa.string()),
        ("embedding", pa.list_(pa.float32(), model.get_sentence_embedding_dimension()))
    ])
    
    return db.create_table(TABLE_NAME, schema=schema)

# ----------------------------
# Move processed files
# ----------------------------
def move_file(src_path, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    shutil.move(src_path, os.path.join(dest_folder, os.path.basename(src_path)))

# ----------------------------
# Process each file
# ----------------------------
def process_file(file_path, table):
    try:
        texts = extract_text(file_path)  # now returns list
        rows = []
        for text in texts:
            embedding = embed_text(text)
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "embedding": embedding
            })
        table.add(rows)
        move_file(file_path, SUCCESS_FOLDER)
        print(f"✅ {os.path.basename(file_path)} processed and stored {len(rows)} chunks.")
    except Exception as e:
        move_file(file_path, FAILURE_FOLDER)
        print(f"❌ Failed to process {os.path.basename(file_path)}: {e}")

# ----------------------------
# Main loop
# ----------------------------
def main():
    # Create folders if not exist
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(SUCCESS_FOLDER, exist_ok=True)
    os.makedirs(FAILURE_FOLDER, exist_ok=True)

    table = connect_to_lancedb()

    # Process all files in input folder
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)
        if os.path.isfile(file_path):
            process_file(file_path, table)

# ----------------------------
if __name__ == "__main__":
    main()

