import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama


def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# chunking
def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks


#summarize

def summarize_chunk(chunk):

    prompt = f"""
    Summarize the following text clearly and concisely:

    {chunk}
    """

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# summarize full pdf
def summarize_pdf(pdf_file):

    text = extract_text(pdf_file)

    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:

        summary = summarize_chunk(chunk)

        summaries.append(summary)

    final_summary = "\n\n".join(summaries)

    return final_summary


# streamlit
st.set_page_config(page_title="PDF Summarizer")

st.title("PDF Summarizer")

st.write("Upload a PDF and summarize it using local LLM.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully")

    if st.button("Generate Summary"):

        with st.spinner("Summarizing PDF"):

            result = summarize_pdf(uploaded_file)

        st.subheader("Summary")

        st.write(result)