# Medinsight-RAG-Chatbot
Medinsight is an advanced Retrieval-Augmented Generation (RAG) chatbot designed to provide fact-checked answers to complex medical queries by leveraging the comprehensive MedQuad knowledge base.
# 🩺 MedInsight RAG Chatbot  
**AI-Powered Healthcare Assistant using RAG + LangChain**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python">
  <img src="https://img.shields.io/badge/LangChain-RAG-brightgreen?logo=chainlink">
  <img src="https://img.shields.io/badge/Google%20Generative%20AI-Powered-orange?logo=googlecloud">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## 🧭 Overview  
**MedInsight RAG Chatbot** is an AI-driven health assistant designed to help users get **instant, evidence-based medical insights**.  
Built with **Retrieval-Augmented Generation (RAG)** architecture, the system retrieves relevant data from a **global medical research dataset**, then generates precise, user-friendly explanations.  

💡 Unlike generic chatbots, MedInsight focuses on **disease recognition**, **symptom analysis**, and **treatment guidance**, making healthcare more accessible — without replacing professional medical advice.

---

## ⚙️ Tech Stack  

| Category | Technologies |
|-----------|---------------|
| 💻 Programming | Python, LangChain, Streamlit |
| 🧠 AI & NLP | Google Generative AI, Sentence Transformers, FAISS |
| 📦 Vector Storage | FAISS |
| 🧩 Integration | RAG Architecture, LLM Prompt Templates |

---

## 📊 Dataset Information  

The dataset used for training and retrieval is a **global English-language medical corpus**, compiled from publicly available research articles, healthcare documentation, and academic sources.  

| Metric | Details |
|--------|----------|
| 🌍 Source | Global Medical Publications (Open Access) |
| 📁 Total Records | 24,612 Documents |
| 📚 Topics | Diseases, Symptoms, Treatments, Anatomy, Diagnostics |
| 🗣️ Language | English |
| 🔍 Purpose | RAG knowledge base for medical QA & reasoning |

---

## How It Works  

1. **User Query** → A health-related question (e.g., *“What are the symptoms of iron deficiency?”*)  
2. **Retriever** → Searches similar text chunks from FAISS vector DB  
3. **Generator (LLM)** → Uses context to craft a reliable, human-like answer  
4. **Response** → Delivered instantly via Streamlit chat interface  

---

## 🖥️ Live Demo  

**Try it here:**  
👉 [https://medinsight-rag.streamlit.app](https://medinsight-rag.streamlit.app)

<p align="center">
  <img src="https://github.com/berkay-shn/Medinsight-RAG-Chatbot/assets/demo.gif" width="700" alt="MedInsight Chatbot Demo">
</p>

## 🧩 Setup Instructions  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/berkay-shn/Medinsight-RAG-Chatbot.git

# 2️⃣ Navigate to project folder
cd Medinsight-RAG-Chatbot

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run app.py
