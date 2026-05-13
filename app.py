import streamlit as st
from dotenv import load_dotenv
from rag import ask_with_sources, get_cached_retriever, get_cached_documents
from ingest import ingest_single_file
from collections import defaultdict
import re
import os
import tempfile



load_dotenv()


st.set_page_config(page_title="Lumos Knowledge Agent", page_icon="🔦", layout="centered")
st.title("🔦 Lumos Knowledge Agent")
st.markdown("Ask anything about Lumos projects, events, and documents.")
@st.cache_resource
def initialize_retriever():
    from rag import get_cached_retriever, get_cached_documents
    get_cached_documents()  # preload documents
    get_cached_retriever()  # preload retriever
    return True
# ── Sidebar — file upload ─────────────────────────────────────────────────────
initialize_retriever()
st.sidebar.title("📁 Add Documents")
st.sidebar.markdown("Upload a file to add it to the knowledge base.")

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


if "upload_message" not in st.session_state:
    st.session_state.upload_message = None

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["pdf", "pptx", "docx", "xlsx", "csv", "md", "txt"],
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_file and st.sidebar.button("Add to knowledge base"):
    # Save to data/ folder
    filepath = os.path.join("data", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.sidebar.status(f"Adding {uploaded_file.name}..."):
        try:
            num_chunks = ingest_single_file(filepath)
            # Clear cache so retriever picks up new docs
            get_cached_retriever.cache_clear()
            get_cached_documents.cache_clear()
            initialize_retriever.clear()
            st.session_state.upload_message = f"✅ Added {uploaded_file.name} ({num_chunks} chunks)"
            st.session_state.upload_key += 1
            st.rerun()
            # st.sidebar.success(f"✅ Added {uploaded_file.name} ({num_chunks} chunks)")
        except Exception as e:
            st.session_state.upload_message = f"❌ Error: {e}"
if st.session_state.upload_message:
    if st.session_state.upload_message.startswith("✅"):
        st.sidebar.success(st.session_state.upload_message)
    else:
        st.sidebar.error(st.session_state.upload_message)          

# ── Helper functions ──────────────────────────────────────────────────────────

def clean_answer(text):
    text = re.sub(r'\s*\(Source:[^)]*\)', '', text)
    text = re.sub(r'\s*\(Document:[^)]*\)', '', text)
    text = re.sub(r'Source:.*?(\n|$)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\[\s*\n\s*\]', '', text) 
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    # st.write("Cleaned answer:", text)
    return text.strip()

def clean_sources(sources):
    file_pages = defaultdict(list)
    for source in sources:
        clean = source.replace("data/", "").replace("data\\", "")
        if " (page " in clean:
            filename, page_part = clean.rsplit(" (page ", 1)
            page_num = page_part.rstrip(")")
            try:
                file_pages[filename].append(int(page_num))
            except:
                pass  # ← just skip if page number can't be parsed
        else:
            if clean.strip():
                file_pages[clean].append(-1)  # Append a default page number if none is found
    result = []
    for filename, pages in file_pages.items():
        if not filename.strip():
            continue
        if pages:
            pages = sorted(set(pages))
            page_str = ", ".join(str(p) for p in pages)
            result.append(f"{filename} — pages {page_str}")
        else:
            result.append(filename)
    result = [r for r in result if r and r.strip()]
    return result

# ── Main chat interface ───────────────────────────────────────────────────────

with st.form(key="qa_form"):
    question = st.text_input(
        "Your question:",
        placeholder="e.g. Who presented the Claude workshop?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Searching documents..."):
        answer, sources = ask_with_sources(question)
    
    
    
    st.markdown("### Answer")
    st.write(clean_answer(str(answer)))
    # Debug — remove after fixing
    # st.code(repr(sources))
    # st.markdown("### Answer")
    # # st.write(clean_answer(answer))
    # st.markdown(clean_answer(str(answer)))

    cleaned = clean_sources(sources)
    cleaned = [s for s in cleaned if s and s.strip()]
    if cleaned:
        st.markdown("### Sources")
        for s in cleaned:
            st.markdown(f"- {s}")