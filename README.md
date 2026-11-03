# Enterprise Knowledge Assistant

An AI-powered document question-answering system
built using Retrieval Augmented Generation (RAG).

## Features
- PDF document loading
- Text chunking
- Hugging Face embeddings
- ChromaDB vector storage
- Semantic search
- CrossEncoder reranking
- OpenAI answer generation
- Source citations
- RAGAS evaluation

## Tech Stack
- Python
- LangChain
- ChromaDB
- Hugging Face
- OpenAI
- Streamlit
- RAGAS

## Architecture

PDF → Chunking → Embeddings → ChromaDB
→ Retrieval → Reranking → LLM → Answer

## Installation

Install dependencies:

pip install -r requirements.txt

## API Configuration

Create a .env file:

OPENAI_API_KEY=your_api_key

## Run Application

streamlit run app.py