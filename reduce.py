#creation of text summarization in refine method using gqoq api key
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)

loader=PyPDFLoader("C:\\Users\\user\\OneDrive\\Desktop\\langchain\\temp.pdf")
docs=loader.load()

split=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
document=split.split_documents(docs)

chain=load_summarize_chain(llm,chain_type="refine",verbose=True)

response=chain.run(document)
