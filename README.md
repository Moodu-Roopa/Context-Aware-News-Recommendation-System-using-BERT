# 📰 Context-Aware News Recommendation System using BERT

This project implements a **context-aware content recommendation system** using BERT (SBERT variant). Given a user’s input (like a news headline or topic), it returns the most relevant news articles based on semantic similarity.

---

## 🚀 Features

- 🔍 **Semantic Search** powered by Sentence-BERT (`all-MiniLM-L6-v2`)
- 🧠 Uses **cosine similarity** over precomputed embeddings
- 📄 **Preprocessed news dataset** with headlines, content, categories, and dates
- 📊 Interactive **Streamlit UI**
- 🧹 Smart text preprocessing with stemming and stopword removal


## ⚙️ Setup Instructions


1. Create and Activate a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate    # On Linux/Mac

venv\Scripts\activate       # On Windows

2. Install Required Packages

pip install -r requirements.txt

  ✅ Make sure your Python version is ≥ 3.8.

3. Download NLTK Resources

If running for the first time, the script will download:

stopwords

punkt

Alternatively, you can download manually in Python:

import nltk 

nltk.download('stopwords')

nltk.download('punkt')

4. Run the Streamlit App

streamlit run news_bert.py

🧪 How It Works

Input: User enters a query/headline.

Preprocessing: Text is cleaned and stemmed.

Embedding: Query is encoded using SBERT.

Similarity Matching: Compares query with article embeddings via cosine similarity.

Top Results: Top 5 most similar articles are displayed.

📦 Dataset & Embeddings

Dataset: books datasets/Articles.csv

Must contain: Heading, NewsType, Article, Date

Embeddings: embeddings.pkl

If not available, generate using the notebook (.ipynb) provided.


👨‍💻 Author
Moodu Roopa
Feel free to connect or contribute!


