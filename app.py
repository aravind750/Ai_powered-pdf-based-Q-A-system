import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
import streamlit as st
import streamlit as st

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model():
    """
    Loads and caches the language model and tokenizer.
    This function is decorated with @st.cache_resource to ensure the model
    is loaded only once, improving performance significantly.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-3b-a800m-instruct", token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.1-3b-a800m-instruct", token=HF_TOKEN)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model. Please check your Hugging Face token and internet connection. Error: {e}")
        return None, None

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF file: {pdf.name}. Error: {e}")
    return text

def get_model_response(prompt, context, tokenizer, model):
    """
    Generates a response from the language model.
    """
    try:
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=4096, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        return answer
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

def generate_flashcards(context, tokenizer, model):
    """
    Generates flashcards (question-answer pairs) from the given text.
    """
    prompt = "Generate a list of question and answer pairs that can be used as flashcards based on the following text. Each pair should be on a new line, with the question and answer separated by a '|'. For example: 'What is the capital of France? | Paris'"
    response = get_model_response(prompt, context, tokenizer, model)
    
    flashcards = []
    for line in response.split('\n'):
        if '|' in line:
            parts = line.split('|', 1)
            if len(parts) == 2:
                question, answer = parts
                flashcards.append((question.strip(), answer.strip()))
    return flashcards

def main():
    """
    The main function for the Streamlit application.
    """
    st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
    st.title("📚 StudyMate: AI-Powered PDF Q&A and Flashcards")
    st.write("Upload your PDF, ask questions, and get summaries with flashcards!")

    # Load the model and tokenizer once with caching
    tokenizer, model = load_model()
    
    if not tokenizer or not model:
        st.stop()

    # File uploader in the sidebar
    with st.sidebar:
        st.header("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDF(s)..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.success("PDF(s) processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Check if the PDF has been processed
    is_processed = "raw_text" in st.session_state

    # Main content area
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question about the PDF:", disabled=not is_processed)
    if st.button("Get Answer", disabled=not is_processed):
        if user_question:
            with st.spinner("Generating answer..."):
                answer = get_model_response(user_question, st.session_state.raw_text, tokenizer, model)
                st.write("### Answer")
                st.info(answer)
        else:
            st.warning("Please enter a question.")

    st.header("Summary & Flashcards")
    if st.button("Generate Summary and Flashcards", disabled=not is_processed):
        with st.spinner("Generating..."):
            # Generate summary
            summary_prompt = "Provide a concise summary of the following text."
            summary = get_model_response(summary_prompt, st.session_state.raw_text, tokenizer, model)
            st.write("### Summary")
            st.success(summary)
            
            # Generate flashcards
            flashcards = generate_flashcards(st.session_state.raw_text, tokenizer, model)
            st.write("### Flashcards")
            if flashcards:
                for i, (question, answer) in enumerate(flashcards):
                    with st.expander(f"**Flashcard {i+1}: {question}**"):
                        st.write(answer)
            else:
                st.write("No flashcards could be generated from the text.")
    
    if not is_processed:
        st.info("Upload and process a PDF in the sidebar to get started!")

if __name__ == "__main__":
    main()
