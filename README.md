# 📚 College Knowledge Assistant — PDF RAG Assistant

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a college-related PDF and ask questions about its contents.

The application extracts text from the uploaded PDF, splits it into chunks, creates semantic embeddings, retrieves the most relevant chunks using FAISS, and sends the retrieved context to Gemini to generate a grounded answer.

---

## 🚀 Features

- 📄 Upload any text-based PDF
- 🔍 Extract text from PDF documents
- ✂️ Split documents into smaller chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 🔎 Perform similarity search using FAISS
- 🤖 Generate answers using Google Gemini
- 💬 Simple Streamlit user interface
- 🔐 API key stored securely using `.env`
- ⚡ Lightweight RAG pipeline without LangChain

---

## 🏗️ Architecture

```text
                ┌─────────────────┐
                │   Upload PDF    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Extract Text   │
                │     PyPDF       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Chunking     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                │ SentenceTrans.  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │      FAISS      │
                │ Vector Search   │
                └────────┬────────┘
                         │
                    User Question
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieve Top-K  │
                │ Relevant Chunks │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Gemini LLM     │
                │ Answer from     │
                │ Retrieved Context│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Final Answer  │
                └─────────────────┘
