import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_Token")

st.title("welcome to retrieval augmented generation")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)

def create_database():
    if "vectors" not in st.session_state:
        st.session_state.embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFLoader('C:\\Users\\user\\OneDrive\\Desktop\\langchain\\temp.pdf')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.split=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
        st.session_state.document=st.session_state.split.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.document,st.session_state.embedding)


if st.button("click to create database"):
    create_database()
    st.write("database is created")

prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based on the given context
the answer should be precise and accurate
<context>
{context}
<context>
Question:{input} 
""")

input_prompt=st.text_input("enter the question")

if input_prompt:
    vectordb=st.session_state.vectors.as_retriever()
    retrieval=create_stuff_documents_chain(llm=llm,prompt=prompt)
    chain=create_retrieval_chain(vectordb,retrieval)
    response=chain.invoke({"input":input_prompt})
    st.write(response['answer'])

    with st.expander("similearity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)


