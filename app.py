import os
import pickle
import streamlit as st
import time
import requests
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from transformers import pipeline

load_dotenv()

st.set_page_config(page_title="AI-Powered News Intelligence System", layout="wide")
st.title("📰 AI-Powered News Intelligence System")
st.sidebar.title("🔗 News Sources & Settings")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🔄 Process News Articles")
file_path = "faiss_vector_index.pkl"

main_placefolder = st.empty()

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9,
    max_tokens=700
)

# Initialize Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

if process_url_clicked:
    st.sidebar.success("Fetching & processing articles...")
    
    # Load URLs data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("🔄 Fetching News Data... ✅")
    data = loader.load()
    
    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placefolder.text("🔄 Splitting & Structuring Articles... ✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save to FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_index = FAISS.from_documents(docs, embedding_model)
    main_placefolder.text("🔄 Generating Embeddings & Indexing... ✅")
    time.sleep(2)
    
    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)
    
    st.sidebar.success("✅ News Processing Complete!")

# User Query
query = st.text_input("💡 Ask about the news:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
            
            retriever = vector_index.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query})
            
            st.header("📝 Answer")
            st.subheader(result.get("answer", "No answer found."))
            
            # Sentiment Analysis
            sentiment = sentiment_analyzer(result.get("answer", ""))[0]
            st.markdown(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
            
            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("📌 Sources")
                sources_list = sources.split("\n") if isinstance(sources, str) else sources
                for source in sources_list:
                    st.write(f"🔗 {source}")
            else:
                st.write("No sources available.")
    else:
        st.error("❌ No processed news found. Please process articles first.")
