import os
import streamlit as st
import pickle
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQAWithSourcesChain
from bs4 import BeautifulSoup
import requests
from langchain.schema import Document
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("OPENAI_API_KEY"))

# Custom Gemini LLM
class GeminiLLM(LLM):
    """Custom Gemini model wrapper for LangChain"""

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text  # Return the generated response as a string
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return "Error generating response"

    @property
    def _llm_type(self) -> str:
        return "Gemini"

# Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "vector_index.pkl"  # Update with your preferred path
llm = GeminiLLM()

if process_url_clicked:
    def fetch_and_parse(url):
        """Fetch and parse the content of a URL."""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()

    # Fetch and parse the URLs
    data = [(url, fetch_and_parse(url)) for url in urls if url]

    # Create documents with 'source' metadata
    documents = [Document(page_content=text, metadata={"source": url}) for url, text in data]

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and FAISS vector index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    st.success("URLs processed and vector index created!")

query = st.text_input("Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            for source in sources.split(", "):
                st.write(f"- {source}")
