import os
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

st.title("Retrieval augmented generation using NVIDIA nim")

llm=ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

def create_database():
    if "vectors" not in st.session_state:
        st.session_state.embedding=NVIDIAEmbeddings()
        st.session_state.loading=PyPDFLoader("C:\\Users\\user\\OneDrive\\Desktop\\langchain\\temp.pdf")
        st.session_state.docs=st.session_state.loading.load()
        st.session_state.split=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
        st.session_state.document=st.session_state.split.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.document,st.session_state.embedding)



if st.button("click here to create database"):
    create_database()
    st.write("database is created")



prompt=ChatPromptTemplate.from_template(
"""
Answer the following question based on the given context
and the answer should be precise and accurate
<context>
{context}
<context>
question:{input}                                                                                                                                                                                                                                                                                                     
"""
)

input_text=st.text_input("enter your question based on the given context")
if input_text:
    db=st.session_state.vectors.as_retriever()
    chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retrieval=create_retrieval_chain(db,chain)
    response=retrieval.invoke({"input":input_text})
    st.success(response['answer'])

    with st.expander("similearity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)

