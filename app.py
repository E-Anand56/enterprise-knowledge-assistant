import streamlit as st
import os
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

# Configuration
load_dotenv()

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚"
)

st.title("Enterprise Knowledge Assistant")
st.write("Ask questions from your employee handbook.")

# Load PDF and create vector database
@st.cache_resource
def load_rag():

    loader = PyPDFLoader(
        "PROITBRIDGE_Employee_Handbook_2026_1.pdf"
    )

    documents = loader.load()

    for doc in documents:
        doc.page_content = re.sub(
            r"\s+", " ", doc.page_content
        ).strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="employee_handbook"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer using only the given context.
    Do not invent information.

    Context:
    {context}

    Question:
    {question}
    """)

    return vectorstore, reranker, llm, prompt


# Initialize RAG
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please configure OPENAI_API_KEY in your .env file.")
    st.stop()

if not os.path.exists("PROITBRIDGE_Employee_Handbook_2026_1.pdf"):
    st.error("Employee handbook PDF not found.")
    st.stop()

with st.spinner("Loading Knowledge Assistant..."):
    vectorstore, reranker, llm, prompt = load_rag()


# User question
question = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if question.strip():

        with st.spinner("Finding answer..."):

            results = vectorstore.similarity_search(
                question, k=3
            )

            pairs = [
                (question, doc.page_content)
                for doc in results
            ]

            scores = reranker.predict(pairs)

            ranked_results = sorted(
                zip(scores, results),
                key=lambda x: x[0],
                reverse=True
            )

            contexts = list(dict.fromkeys(
                doc.page_content
                for score, doc in ranked_results
            ))

            context = "\n\n".join(contexts)

            chain = prompt | llm

            response = chain.invoke({
                "context": context,
                "question": question
            })

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Sources")

        for score, doc in ranked_results:
            st.write(
                f"Page: {doc.metadata.get('page', 0) + 1}"
            )