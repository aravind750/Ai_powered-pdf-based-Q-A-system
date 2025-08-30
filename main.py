# main.py
# Import necessary libraries
import streamlit as st
import os
import fitz  # PyMuPDF
import time # Added for retry logic
import json # Added for flashcard parsing
import re   # Added for robust JSON parsing
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Load variables from the .env file
load_dotenv()

# --- Configuration ---
# Set up the Streamlit page
st.set_page_config(
    page_title="StudyMate: AI PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Hugging Face API Token Configuration ---
# Function to check for environment variables
def check_credentials():
    """
    Checks if the necessary environment variable is set.
    """
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("🚨 HUGGINGFACEHUB_API_TOKEN not found.")
        st.info("Please create a .env file and add your Hugging Face API token to it.")
        st.stop()

# --- Model Configuration ---
def get_model_parameters():
    """
    Returns a dictionary of parameters for the model's text generation.
    """
    return {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.03
    }

# --- Model Initialization ---
@st.cache_resource
def initialize_model():
    """
    Initializes and returns the Hugging Face chat model instance using LangChain.
    """
    try:
        # Connect to a cloud-based Hugging Face model
        llm = HuggingFaceEndpoint(
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            **get_model_parameters()
        )
        # Wrap the LLM in the ChatHuggingFace adapter for conversation
        model = ChatHuggingFace(llm=llm)
        return model
    except Exception as e:
        st.error(f"🔥 Failed to initialize the model: {e}")
        st.stop()

# --- PDF and Text Processing Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def generate_summary(model, text):
    """
    Generates a summary for the given text using a Map-Reduce approach with retries.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    retries = 3

    map_summaries = []
    for i, doc in enumerate(docs):
        map_prompt = f"Please provide a concise summary of the following text chunk:\n\n---\n{doc.page_content}\n---"
        for attempt in range(retries):
            try:
                response = model.invoke([HumanMessage(content=map_prompt)])
                map_summaries.append(response.content)
                break 
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(f"Attempt {attempt + 1} failed for chunk {i + 1}. Retrying...")
                    time.sleep(2)
                else:
                    st.error(f"Could not summarize chunk {i + 1} after {retries} attempts. Error: {e}")
    
    if not map_summaries:
        st.error("Could not generate any summaries from the document chunks. Aborting.")
        return None

    combined_summary = "\n".join(map_summaries)
    reduce_prompt = f"Please create a single, cohesive summary from the following collection of summaries:\n\n---\n{combined_summary}\n---"
    try:
        final_response = model.invoke([HumanMessage(content=reduce_prompt)])
        return final_response.content
    except Exception as e:
        st.error(f"Error generating final summary: {e}")
        return None

def generate_flashcards(model, summary):
    """Generates flashcards from the summary text."""
    prompt = f"""
    Based on the following summary, create a set of flashcards. Each flashcard must have a 'topic' and a detailed 'explanation'. Return the output as a valid JSON array of objects only, with no other text before or after the array.

    Example format:
    ```json
    [
        {{"topic": "Key Concept 1", "explanation": "Detailed explanation of the first key concept."}},
        {{"topic": "Important Figure", "explanation": "Detailed explanation about the important figure."}}
    ]
    ```

    Summary:
    ---
    {summary}
    ---
    """
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        response_content = response.content
        
        json_match = re.search(r'```json\s*(\[.*\])\s*```', response_content, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(1)
            flashcards = json.loads(json_string)
            return flashcards
        else:
            st.error("No valid JSON code block found in the model's response for flashcards.")
            st.code(response_content)
            return []
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode the flashcards from the model's response. Error: {e}")
        st.code(response_content)
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while generating flashcards: {e}")
        return []

# --- Chat Logic ---
def generate_chat_response(model, chat_history, pdf_context=""):
    """
    Generates a response from the model, with optional PDF context.
    """
    try:
        messages = []
        for message in chat_history:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))

        if pdf_context and messages and isinstance(messages[-1], HumanMessage):
            last_user_message = messages[-1].content
            augmented_prompt = (
                f"Using the context below, answer the following question.\n\n"
                f"Context: {pdf_context}\n\n"
                f"Question: {last_user_message}"
            )
            messages[-1] = HumanMessage(content=augmented_prompt)

        response = model.invoke(messages)
        return response.content
    except Exception as e:
        return f"An error occurred during response generation: {e}"

# --- Streamlit UI ---
st.title("📚 StudyMate: AI PDF Assistant")
st.caption("Powered by Streamlit and Hugging Face. Provide an API key to get started.")

check_credentials()
model = initialize_model()

# --- Sidebar for File Upload and Controls ---
with st.sidebar:
    st.header("Upload & Control")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf", key="pdf_uploader")

    if st.button("Clear Chat & Document"):
        st.session_state.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
        st.rerun()
    
    st.divider()
    st.info("This app requires a Hugging Face API token. Please add it to your .env file.")


# --- Main Logic for Processing PDF ---
if uploaded_file and "pdf_text" not in st.session_state:
    with st.spinner("Analyzing your document... This may take a moment."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.session_state.pdf_text = pdf_text
            summary = generate_summary(model, pdf_text)
            if summary:
                st.session_state.summary = summary
                flashcards = generate_flashcards(model, summary)
                st.session_state.flashcards = flashcards

# --- Main Area for Displaying Content ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("💬 Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document to get started."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pdf_context = st.session_state.get("pdf_text", "")
                if not pdf_context:
                    response_content = "Please upload a PDF document first so I can answer your questions about it."
                else:
                    response_content = generate_chat_response(model, st.session_state.messages, pdf_context)
                st.markdown(response_content)
        
        st.session_state.messages.append({"role": "assistant", "content": response_content})

with col2:
    if "summary" in st.session_state:
        st.header("📄 Summary")
        st.info(st.session_state.summary)

    if "flashcards" in st.session_state and st.session_state.flashcards:
        st.header("🃏 Flashcards")
        for card in st.session_state.flashcards:
            if "topic" in card and "explanation" in card:
                with st.expander(f"**{card['topic']}**"):
                    st.write(card['explanation'])
    elif "pdf_text" in st.session_state:
        st.info("Flashcards are being generated or were not created successfully.")
