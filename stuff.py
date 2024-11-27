#creation of text summarization in stuff method using groq api key 
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)

loader=PyPDFLoader("C:\\Users\\user\\OneDrive\\Desktop\\langchain\\temp.pdf")
docs=loader.load()
print(docs)

template="""
As you are an ai expert in text summarization 
please provide me a summary of the context
text:{text}
summary:
"""

prompt=PromptTemplate(input_variables=["text"],template=template)

chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt,verbose=True)

response=chain.run(docs)

print(response)