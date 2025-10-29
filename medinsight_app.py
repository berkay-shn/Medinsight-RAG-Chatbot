import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
# YENİ MIMARIYE UYUMLU IMPORTLAR
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document # LangChain Document nesnesini kullanmak kritik
import os
# --- SAYFA VE TEMA AYARLARI ---
st.set_page_config(
    page_title="Medinsight RAG Bot",
    layout="wide", # Sayfanın tüm genişliğini kullanır
    initial_sidebar_state="collapsed"
)


# --- 1. ENV & API KEY SETUP ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# --- 2. RAG PIPELINE SETUP ---
@st.cache_resource
def setup_rag_pipeline():
    # ... (st.spinner ve data = load_dataset kısmı aynı kalır)

    # VERİ YÜKLEME VE DÖNÜŞTÜRME 
    with st.spinner("Loading the knowledge base... This may take a few minutes on the first run."):
        
        data = load_dataset("Laurent1/MedQuad-MedicalQnADataset_128tokens_max", split="train")

        docs = [] 
        
        # Sütun isimleri Mevcut veri setine göre düzeltildi: 'text' ve 'question'
        for article in data:
            # DÜZELTME: Ana içerik sütunu 'text' olarak değiştirildi.
            page_content = article.get("text", None) 
            
            if page_content and len(page_content) > 10:
                doc = Document(
                    page_content=page_content, 
                    metadata={
                        # DÜZELTME: Soru sütunu 'question' olarak değiştirildi.
                        "title": article.get("question", "Başlık Yok"), 
                        "source": article.get("url", "Bilinmiyor"),
                        "qtype": article.get("qtype", "Genel")
                    }
                )
                docs.append(doc)
    
    # Hata Kontrolü: Eğer hiç belge yüklenmezse (örneğin sadece 10 karakterden kısa cevaplar varsa)
    if not docs:
        raise ValueError("No valid documents were created from the dataset. Check your data or filtering logic.")


    # EMBEDDINGS OLUŞTURMA
    with st.spinner("Loading embedding model..."):
      embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS VEKTÖR VERİTABANI OLUŞTURMA
    with st.spinner("Creating vector database..."):
        # Doğrudan LangChain Document listesini kullanmak için .from_documents kullanılır, 
        # bu metin ve metadata'yı otomatik olarak ayırır.
        vector_store = FAISS.from_documents(docs, embeddings) 

    # RAG ZINCIRI KURULUMU 
    with st.spinner("Initializing language model and RAG chain..."):
        # LLM initialization
        llm = GoogleGenerativeAI(model="gemini-2.5-flash") # gemini-pro-latest yerine gemini-2.5-flash kullanıldı (daha hızlı ve güncel)
        retriever = vector_store.as_retriever()
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True # Kaynak belgeleri döndürmek için
        )
        return rag_chain # Zinciri döndür

# Initialize pipeline
try:
    rag_chain = setup_rag_pipeline()
    st.success("MedInsight is ready to answer your questions!")
except Exception as e:
    # Hata durumunda uygulama durdurulur ve hata mesajı gösterilir.
    st.error(f"An error occurred during setup: {e}")
    st.stop()

# --- ÖZEL CSS STİLİ (KLİNİK GÜVEN TEMASI) ---
st.markdown("""
<style>
/* 1. Başlık ve Genel Stil */
h1 {
    text-align: center;
    color: #007bff; /* Koyu Mavi */
    border-bottom: 2px solid #007bff;
    padding-bottom: 10px;
    font-weight: 600;
}

/* 2. Ana Uygulama Arka Planı (Hafif gri) */
.main {
    background-color: #f8f9fa;
}

/* 3. Sohbet Mesajı Baloncukları */
[data-testid="stChatMessage"] {
    background-color: white; /* Temiz Beyaz Arka Plan */
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* Hafif Gölgelendirme */
}

/* 4. Asistan (Bot) Cevabı Vurgusu */
/* Bu CSS sınıfı Streamlit'in cevabı çevreleyen div'ine özeldir */
[data-testid="stChatMessage"]:nth-child(even) { /* Çift sayılı mesajlar (genellikle asistan) */
    background-color: #e8f4ff; /* Açık Mavi arka plan */
    border-left: 5px solid #007bff; /* Yan çizgi ile vurgu */
}

/* 5. Kullanıcı Giriş Alanı */
.stChatInputContainer {
    padding-top: 20px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)
# -----------------------

# STREAMLIT WEB INTERFACE
st.title("MedInsight 🩺")
st.markdown(
    "MedInsight is a specialized **RAG (Retrieval-Augmented Generation) Chatbot** designed to answer complex medical questions." 
    "It provides reliable and comprehensive, fact-checked answers derived strictly from the structured medical knowledge base."
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new user question
user_question = st.chat_input("Ask a question about the medical articles...")

if user_question:
    # Kullanıcı mesajını kaydet ve göster
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Generating response..."):
        try:
            # LangChain'in yeni mimarisinde invoke kullanılıyor.
            response_dict = rag_chain.invoke({"query": user_question}) 
            response = response_dict["result"]
            
        except Exception as e:
            response = f"An error occurred while generating the response: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})