#machine translation from one langauge to another langauge
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")


model=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)

chat=ChatPromptTemplate.from_messages([
    ("system","As you are an AI assisstant please translate the given text into following langauge {language}"),
    ("user","{text}")
])

parser=StrOutputParser()

st.title("machine translation using groq")

lang=st.text_input("enter the language to tanslate")

text_input=st.text_input("enter the text to translate")


#lce=>langchain expression Language 
combine=chat|model|parser

if text_input:
    st.write(combine.invoke({"language":lang,"text":text_input}))