
import os

import faiss
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
import time
from google.genai import errors


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF and ask questions about its contents."
)


# ==========================================
# 2. LOAD GEMINI API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please add it to your .env file."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ==========================================
# 4. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 6. CREATE FAISS VECTOR DATABASE
# ==========================================

def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ==========================================
# 7. RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=3
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    retrieved_chunks = []

    for index_number in indices[0]:

        if index_number >= 0:
            retrieved_chunks.append(
                chunks[index_number]
            )

    return retrieved_chunks


# ==========================================
# 8. GENERATE ANSWER USING GEMINI
# ==========================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present in the context, say:
"I could not find the answer in the provided PDF."

Context:
{context}

Question:
{question}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            return response.text

        except errors.ServerError as e:
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return (
                    "Gemini is temporarily unavailable because the "
                    "model is experiencing high demand. "
                    "Please try again in a few seconds."
                )

# ==========================================
# 9. PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ==========================================
# 10. PROCESS PDF
# ==========================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    with st.spinner(
        "Reading and processing the PDF..."
    ):

        text = extract_text_from_pdf(
            uploaded_file
        )

    if not text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        st.info(
            "Please upload a text-based PDF."
        )

        st.stop()

    # Create chunks
    chunks = create_chunks(text)

    # Create vector database
    index = create_vector_database(chunks)

    st.success(
        f"PDF processed successfully! "
        f"Created {len(chunks)} text chunks."
    )


    # ==========================================
    # 11. QUESTION INPUT
    # ==========================================

    question = st.text_input(
        "Ask a question about the PDF:"
    )


    # ==========================================
    # 12. GENERATE ANSWER
    # ==========================================

    if st.button("Get Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the PDF..."
            ):

                retrieved_chunks = retrieve_chunks(
                    question,
                    chunks,
                    index
                )


            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )


            # ==========================================
            # 13. DISPLAY ANSWER
            # ==========================================

            st.subheader("Answer")

            st.write(answer)


            # ==========================================
            # 14. SHOW RETRIEVED CONTEXT
            # ==========================================

            with st.expander(
                "🔎 View Retrieved Context"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.write(
                        f"**Retrieved Chunk {i}**"
                    )

                    st.write(chunk)

                    st.divider()