import os
import fitz  # PyMuPDF for extracting text from PDFs
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

# Find Top N JD Matches
def get_top_jds(resume_pdf, jd_folder, top_n=5):
    """Matches resume with job descriptions and returns the top N results."""
    resume_text = extract_text_from_pdf(resume_pdf)
    jd_texts, jd_files = [], []
    
    for file in os.listdir(jd_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(jd_folder, file)
            jd_texts.append(extract_text_from_pdf(file_path))
            jd_files.append(file)

    # Compute SBERT embeddings
    resume_embedding = model.encode([resume_text], convert_to_numpy=True)
    jd_embeddings = model.encode(jd_texts, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = cosine_similarity(resume_embedding, jd_embeddings)[0]

    # Sort results by similarity score
    scores = sorted(zip(jd_files, similarities), key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("📄 Resume → Job Description Matching")

uploaded_resume = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
jd_folder = "JD_db"  # Folder containing job descriptions

if uploaded_resume:
    if not os.listdir(jd_folder):
        st.error("⚠️ No job descriptions found! Please add some PDFs to the 'JD_db' folder.")
    else:
        top_n = st.slider("🔢 Select Number of Job Matches:", 1, min(10, len(os.listdir(jd_folder))), 5)

        temp_resume_path = "temp_resume.pdf"
        with open(temp_resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

        matches = get_top_jds(temp_resume_path, jd_folder, top_n)

        st.write("### ✅ Top Matching Job Descriptions:")
        for i, (file, score) in enumerate(matches, 1):
            file_path = os.path.join(jd_folder, file)

            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{i}. 📄 **{file}** - 🏆 Score: {round(score * 100, 2)}%")
            with col2:
                st.download_button(f"⬇️ Download", pdf_data, file_name=file, mime="application/pdf")
            with col3:
                apply_link = f"https://www.example.com/jobs/{file.replace('.pdf', '')}"  # Example job application link
                st.markdown(f"[🚀 Apply]( {apply_link} )", unsafe_allow_html=True)

        os.remove(temp_resume_path)
