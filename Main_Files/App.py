import streamlit as st  # Used to create web application

 
st.set_page_config(
    page_title="Resume Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import pickle  # To import the 3 pickle files
from pdf2image import convert_from_bytes  # Helps to see pdf as image
import pytesseract  # Helps to do OCR on text
from PyPDF2 import PdfReader  # Helps handle pdf files

# Set the path for Tesseract executable 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the path to Poppler's bin folder
POPPLER_PATH = r"C:\Users\GODWIN\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# Load trained KNN model
with open(r'D:\Projects\Resume Classifer\KNN.pkl', 'rb') as f:
    model = pickle.load(f)

# Load vectorizer (like TfidfVectorizer)
with open(r'D:\Projects\Resume Classifer\Tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load LabelEncoder to convert numeric predictions to job role names
with open(r'D:\Projects\Resume Classifer\label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Adding custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #EFE9E1;
        font-family: "Arial", sans-serif;
        color: #333;
    }
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 1.5em;
        color: #333;
        text-align: center;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 12px 25px;
        border-radius: 10px;
        border: none;
    }
    .stFileUploader>div {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #fff;
    }
    .result {
        font-size: 1.3em;
        color: #FF5722;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Resume Classifier</div>', unsafe_allow_html=True)
st.write("Upload a resume (in .txt or .pdf format) and get the predicted job role based on the content provided.")

# File uploader 
uploaded_file = st.file_uploader("Select a file to upload:", type=["txt", "pdf"])

# Function to extract text from PDFs 
def extract_text(file):
    if file.type == "application/pdf":
        # Convert PDF pages to images (with Poppler path)
        images = convert_from_bytes(file.read(), poppler_path=POPPLER_PATH)

        # Extracting text from images using OCR
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

        # Also extract selectable text if any
        file.seek(0)
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""

        return text
    else:
        # Also handles .txt file
        return str(file.read(), "utf-8")

# If file is uploaded
if uploaded_file is not None:
    with st.spinner('Analyzing your resume...'):
        # Extract text from the uploaded resume
        text = extract_text(uploaded_file)

        # Convert to lowercase to standardize text
        text = text.lower()

        # Vectorize the text and predict the job role
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        predicted_role = le.inverse_transform([prediction])[0]

        # Display the predicted job role 
        st.markdown(f'<div class="result">Predicted Job Role: {predicted_role}</div>', unsafe_allow_html=True)
