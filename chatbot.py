import streamlit as st
from PyPDF2 import PdfReader
# newer version of LangChain, where text_splitter is no longer exposed at the top-level langchain package.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama  # updated import to avoid deprecation
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.base import Embeddings  # for proper dummy embeddings


# set API key in environment
os.environ["OPENAI_API_KEY"] = "sk-..."

st.header("Database Title")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Choose a pdf file", type=["pdf"])

    # select model
    model_choice = st.selectbox("Select LLM to use", ["OpenAI GPT-3.5", "phi3 Ollama Local"])

# read pdf
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # print(text)
        # st.write(text)

    # Split info
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", ". ", "? ", "! ", "\n", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # choose embeddings based on LLM selection
    if model_choice == "OpenAI GPT-3.5":
        embeddings = OpenAIEmbeddings()
    else:
        # local dummy embeddings for FAISS (phi3 Ollama)
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[1.0]*1536 for _ in texts]  # 1536 e typical OpenAI vector size
            def embed_query(self, text):
                return [1.0]*1536
        embeddings = DummyEmbeddings()

    # vector storage with FAISS
    vector_storage = FAISS.from_texts(chunks, embeddings)
    retriever = vector_storage.as_retriever()

    # inquire prompt
    question_inquire = st.text_input("Inquire question here: ")

    # search DB
    if question_inquire:
        # get relevant docs
        docs = retriever.invoke(question_inquire)

        # choose LLM based on selection
        # GhatGPT nu mai are API key
        if model_choice == "OpenAI GPT-3.5":
            llm = ChatOpenAI(
                temperature=0.1,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )
        else:
            # phi3 cu local Ollama
            llm = ChatOllama(
                model="phi3",
                temperature=0.1
            )

        prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below:\n\n{context}\n\nQuestion: {question}"
        )
        chain = (
                prompt
                | llm
                | StrOutputParser()
        )

        context = "\n\n".join(doc.page_content for doc in docs)
        response = chain.invoke({"context": context, "question": question_inquire})
        st.write(response)
