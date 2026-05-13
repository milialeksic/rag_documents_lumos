# 🔦 Lumos Knowledge Agent

A RAG-based AI agent that gives Lumos members instant, conversational access to internal documents — meeting notes, project reports, retrospectives, and onboarding docs.

---

## What it does

Ask a question in natural language, get a grounded answer with source references. No manual digging through Notion required.

---

## Requirements

- Python 3.11+
- OpenAI API key (with credits)
- Tesseract OCR (for presentation PDFs)
- Poppler (for PDF to image conversion)

### Install Tesseract

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-deu poppler-utils
```

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/milialeksic/rag_documents_lumos.git
cd rag_documents_lumos
```

**2. Create a virtual environment**
```bash
python -m venv env
source env/bin/activate        # Linux/Mac
env\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file**
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**5. Add documents**

Put your Lumos documents (PDFs, PPTX, DOCX, XLSX, CSV, MD) into the `data/` folder.

Supported formats:
- `.pdf` — regular documents (pdfplumber) and presentations (OCR)
- `.pptx` — PowerPoint presentations
- `.docx` — Word documents
- `.xlsx` / `.csv` — spreadsheets
- `.md` / `.txt` — Notion exports and plain text

**6. Ingest documents**
```bash
python ingest.py
```

This extracts text, chunks documents, creates embeddings and stores them in the vector database. Run this every time you add new documents.

**7. Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project structure

```
lumos-agent/
│
├── data/               ← put your documents here
├── vectorstore/        ← auto-generated vector database
├── app.py              ← Streamlit UI
├── ingest.py           ← document ingestion pipeline
├── rag.py              ← retrieval and answer logic
├── evaluate.py         ← RAGAS evaluation pipeline
├── notion_export.py    ← export pages from Notion API
├── .env                ← API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How it works

```
Documents (PDF, PPTX, DOCX, XLSX, MD)
        ↓
   Text extraction
   (pdfplumber / OCR / python-pptx / python-docx)
        ↓
   Smart chunking
   (by document type — slides kept whole, prose split)
        ↓
   Embeddings + Vector store (Chroma + OpenAI)
        ↓
   User question
        ↓
   Hybrid retrieval (BM25 + vector search)
        ↓
   Neighbor expansion (±2 pages for context)
        ↓
   LLM answer (GPT-4o-mini)
        ↓
   Answer + source references
```

---

## Running from the terminal

To ask a single question without the UI:
```bash
python rag.py
```

---

## Evaluating quality

To run RAGAS evaluation:
```bash
python evaluate.py
```

This generates test questions from your documents and scores the pipeline on faithfulness, context precision and context recall.

---

## Adding new documents

1. Drop new files into the `data/` folder
2. Delete the old vector store: `rm -rf vectorstore/`
3. Re-run ingestion: `python ingest.py`

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |
| `NOTION_TOKEN` | Notion API token (optional, for `notion_export.py`) |

---

## Tech stack

| Component | Technology |
|-----------|------------|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-ada-002 (OpenAI) |
| Vector store | Chroma (local) |
| Retrieval | Hybrid BM25 + vector search |
| Framework | LangChain |
| UI | Streamlit |
| PDF extraction | pdfplumber + Tesseract OCR |

---

## Point of contact

Kirill Medovshchikov — Head of IT, Lumos Student Organization
