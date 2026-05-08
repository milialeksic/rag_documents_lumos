import streamlit as st
from dotenv import load_dotenv
from rag import ask_with_sources

load_dotenv()

st.set_page_config(page_title="Lumos Knowledge Agent", page_icon="🔦")
st.title("🔦 Lumos Knowledge Agent")
st.markdown("Ask anything about Lumos projects, events, and documents.")

question = st.text_input("Your question:", placeholder="e.g. Who presented Cooking with Claude?")

if st.button("Ask") and question:
    with st.spinner("Searching documents..."):
        answer, sources = ask_with_sources(question)
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### Sources")
    for source in sources:
        st.markdown(f"- `{source}`")