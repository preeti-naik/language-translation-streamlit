# 🌍 Multilingual AI Translation System

A Transformer-based multilingual language translation system capable of translating text between 200+ languages using state-of-the-art deep learning models. This project is designed for academic evaluation as well as industry-level demonstration.



## 📌 Project Overview

Language barriers limit access to information and global communication. This project addresses the problem by building an AI-powered translation system that supports multiple languages, including low-resource languages like Odia.

The system uses a pretrained Transformer-based multilingual model to perform accurate and scalable translations in real time.



## 🎯 Objectives

- Build a multilingual translation system using Transformer models
- Support translation across 200+ languages
- Provide an easy-to-use web interface using Streamlit
- Demonstrate AI system design for academic and industry use



## 🧠 AI Technique Used

- **Model**: Facebook NLLB-200 (No Language Left Behind)
- **Architecture**: Transformer-based Seq2Seq model
- **Domain**: Natural Language Processing (NLP)
- **Frameworks**:
  - Hugging Face Transformers
  - PyTorch
  - Streamlit



## 🏗️ System Architecture

1. User inputs text in any language  
2. Source language is automatically detected  
3. Target language is selected from dropdown or entered manually  
4. Text is tokenized and passed to the Transformer model  
5. Model generates translated output  
6. Result is displayed to the user  



## 🖥️ Project Structure

language transalation/
│
├── app.py # Streamlit web application
├── notebook.ipynb # Detailed Jupyter notebook (analysis & explanation)
├── requirements.txt # Python dependencies
└── README.md # Project documentation



## ⚙️ Installation & Setup

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
Run Streamlit app
streamlit run app.py


The application will open automatically in  browser.


Evaluation & Results

Qualitative evaluation based on translation accuracy

Successfully translates between major and regional languages

Handles unseen input text effectively

Works in real-time with minimal latency



⚖️ Ethical Considerations

Possible bias due to pretrained model data

Translations may vary for low-resource languages

Responsible usage recommended for critical applications



🔮 Future Scope

 Add speech-to-text and text-to-speech support

Improve UI with language search

Add offline translation support

Fine-tune model for regional languages

