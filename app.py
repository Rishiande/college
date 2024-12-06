import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.cloud import storage
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Access credentials from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Check for missing credentials
if not GOOGLE_API_KEY or not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("Google API key or Application Credentials not found. Check your .env file.")

# Configure environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Function to download a file from GCS
def download_file_from_bucket(bucket_name, source_blob_name, destination_file_name):
    """Download a file from the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "Answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    normalized_question = user_question.lower()
    keyword_mapping = {
        "hod": "head",
        "cse": "computer science and engineering",
        "ece": "electrical and communication engineering",
        "eee": "electrical and electronics engineering",
        "ai": "artificial intelligence",
        "ds": "data science",
    }

    for key, value in keyword_mapping.items():
        normalized_question = normalized_question.replace(key, value)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(normalized_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": normalized_question}, return_only_outputs=True)

    return response["output_text"]

def main():
    st.set_page_config(page_title="🎓 RCEE College Chatbot", layout="wide")
    
    # Sidebar navigation options
    st.sidebar.title("Navigation Bar")
    app_mode = st.sidebar.selectbox("Choose an option:", ["College Info", "Student Marks"])
    
    # Title of the Streamlit app
    st.title("🎓 RCEE College Chatbot")

    # Initialize session state for chat messages and input
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input" not in st.session_state:
        st.session_state.input = ""  # Initialize input session state

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Specify bucket details and file paths
    bucket_name = "rcee"
    source_blob_name = "RCEE.pdf"
    local_file_name = "RCEE.pdf"

    # Download the PDF file from GCS
    pdf_file_path = download_file_from_bucket(bucket_name, source_blob_name, local_file_name)

    # Process the downloaded PDF file
    raw_text = get_pdf_text([pdf_file_path])
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Define suggestions
    suggestions = [
        "🎓 What programs are offered?",
        "👨‍🏫 Who is the principal?",
        "🌟 What is the vision of RCEE?",
        "🏛 Tell me about the college?",
        "📚 What academic accreditations does RCEE hold?"
    ]

    # Display suggestion buttons in a horizontal layout
    st.subheader("Quick Suggestions:")
    cols = st.columns(len(suggestions))
    
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion):
            st.session_state["input"] = suggestion

    # Input box for user query at the bottom right, pre-filled with suggestion if clicked
    prompt = st.chat_input("Ask me anything about RCEE (e.g., 'What programs are offered?', 'Who is the principal?')") or st.session_state.input

    if prompt:
        # Clear the input session state after question is asked
        st.session_state.input = ""
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in the chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate content based on user input
        response = user_input(prompt)
        
        # Display assistant response in the chat
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8080)
