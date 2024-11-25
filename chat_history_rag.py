import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import MessagesPlaceholder

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

st.title("enter your groq api")
groq_api_key=st.sidebar.text_input("enter your groq",type="password")

file_uploaders=st.file_uploader("upload your file",type=['pdf'],accept_multiple_files=True)

if groq_api_key:
    llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    for file_uploader in file_uploaders:
        document=[]
        gen=f"./gen.pdf"
        with open(gen,'wb') as file:
            file.write(file_uploader.getvalue())
            file_name=file_uploader.name

        loader=PyPDFLoader(gen)
        docs=loader.load()
        document.extend(docs)

    split=CharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    splitter=split.split_documents(document)
    vectordb=FAISS.from_documents(splitter,embedding=embedding)
    db=vectordb.as_retriever()

    contextual_q_system=(
    """
     you should answer the question precise and accurately and
     without chat history you don't answer the question
                         
    """)

    contextual_query=ChatPromptTemplate.from_messages([
           ("system",contextual_q_system),
           MessagesPlaceholder("chat_history"),
           ("human","{input}")
    ])

    history=create_history_aware_retriever(llm,db,contextual_query)


    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
    )

    prompt2=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
    ])

    chain=create_stuff_documents_chain(llm=llm,prompt=prompt2)
    retrieval=create_retrieval_chain(history,chain)

    store={}

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id]=ChatMessageHistory()
        return store[session_id]
    
    with_message_history=RunnableWithMessageHistory(
                           retrieval,
                           get_session_history,
                           input_messages_key="input",
                           history_messages_key="chat_history",
                           output_messages_key="answer")
    
    input_text=st.text_input("enter your question")

    if input_text:
        response=with_message_history.invoke({"input":input_text},
                            config={"configurable":{"session_id":"abc"}})
    
        st.success(response["answer"])

else:
    st.warning("enter the groq api key")