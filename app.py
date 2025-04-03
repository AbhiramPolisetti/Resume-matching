import os
import fitz 
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

def get_top_matches(input_pdf: str, comparison_folder: str, top_n: int = 5):
    """
    Find top N matching files for a given input PDF using TF-IDF.
    :param input_pdf: Path to the input PDF (Resume or JD).
    :param comparison_folder: Path to the folder containing comparison PDFs (JDs or Resumes).
    :param top_n: Number of top matches to return.
    """
    input_text = extract_text_from_pdf(input_pdf)
    comparison_texts = []
    comparison_files = []
    
    for file in os.listdir(comparison_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(comparison_folder, file)
            comparison_texts.append(extract_text_from_pdf(file_path))
            comparison_files.append(file)
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_text] + comparison_texts)
    input_vector = tfidf_matrix[0]
    comparison_vectors = tfidf_matrix[1:]
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, comparison_vectors)[0]
    
    # Sort by similarity and return top N
    scores = sorted(zip(comparison_files, similarities), key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("Resume & Job Description Matching ")
option = st.radio("Select Functionality:", ["Resume → JDs", "JD → Resumes"])

uploaded_file = st.file_uploader("Upload Resume or JD (PDF)", type=["pdf"])
resume_folder = "resume_db" 
jd_folder = "JD_db"  


if uploaded_file:
    # Determine comparison folder
    comparison_folder = jd_folder if option == "Resume → JDs" else resume_folder
    
    # Set slider after determining comparison folder
    top_n = st.slider("Number of top matches:", 1, len(os.listdir(comparison_folder)), min(5, len(os.listdir(comparison_folder))))
    # Save uploaded file temporarily
    temp_pdf_path = os.path.join("temp.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Determine comparison folder
    comparison_folder = jd_folder if option == "Resume → JDs" else resume_folder
    
    # Get top matches
    matches = get_top_matches(temp_pdf_path, comparison_folder, top_n)
    
    # Display results
    st.write("### Top Matches:")
    for i, (file, score) in enumerate(matches, 1):
        st.write(f"{i}. {file} - Matching Score: {round(score * 100, 2)}%")
    
    # Clean up temp file
    os.remove(temp_pdf_path)
