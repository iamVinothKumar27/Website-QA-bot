import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load env vars and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Extract readable text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n")
    except Exception as e:
        st.error(f"Failed to fetch content: {e}")
        return ""

# Chunk text for embedding
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Store embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("vectors_storage")

# QA chain using Gemini
def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant that reads the context and answers the question.
    Be detailed. If unsure, say "I couldn't find that in the website content."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Answering logic
def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.load_local("vectors_storage", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query)
    chain = get_conversational_chain()
    result = chain.invoke({"input_documents": docs, "question": query})
    return result["output_text"]

# Streamlit UI
st.set_page_config(page_title="🌐 Website Chatbot", layout="centered")
st.title("🌐 Chat about a Website")
st.markdown("Enter a URL and ask questions about the website content.")

url = st.text_input("🔗 Enter website URL (e.g., https://en.wikipedia.org/wiki/Artificial_intelligence)")

if url:
    st.info("📄 Extracting content...")
    text_data = extract_text_from_url(url)
    if text_data.strip():
        chunks = get_text_chunks(text_data)
        get_vector_store(chunks)
        st.success("✅ Website content processed and vector store created!")

# Chat interface
if os.path.exists("vectors_storage/index.faiss"):
    question = st.text_input("💬 Ask a question about the website")
    if question:
        with st.spinner("Thinking..."):
            try:
                answer = user_input(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please enter a valid website URL first.")
