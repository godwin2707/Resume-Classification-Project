The Resume Classification Project is a machine learning application designed to automatically classify resumes into specific job roles (e.g., Data Scientist, Web Developer, etc.) based on the content of the resume. Here's a detailed explanation of how the project works and what it includes:

🔍 Project Overview
The goal is to build a web-based app that accepts a resume (in .txt or .pdf format), processes its content, and predicts the most suitable job role using a trained machine learning model.

What problems does it solve?
It reduces the time and manual effort involved in screening resumes, helping recruiters and hiring platforms identify the right candidate fit more efficiently.

🧠 Key Components
Frontend/UI:
Built with Streamlit, a Python library for building web apps for ML/data science.
Users can upload resumes in .txt or .pdf formats.
Shows extracted resume text and displays predicted job role.

Text Extraction:
For PDF files, uses:
PyPDF2 for extracting text from standard PDFs.
pdf2image + pytesseract for OCR-based extraction from graphically designed PDFs (containing images or fancy fonts).
For .txt files, reads the text directly.

Preprocessing & Vectorization:
Text is converted to lowercase and then transformed using TF-IDF vectorization (TfidfVectorizer.pkl), which converts text to numerical format suitable for ML models.

Machine Learning Model:
Uses a K-Nearest Neighbors (KNN) model (KNN.pkl) trained on labeled resume data.

Model predicts a numerical label, which is decoded to a job title using a LabelEncoder (label_encoder.pkl).

🛠️ Tech Stack
Python
Streamlit – Web interface
scikit-learn – Model training and prediction
PyPDF2, pdf2image, pytesseract – PDF text extraction
pickle – Loading trained models and vectorizers
