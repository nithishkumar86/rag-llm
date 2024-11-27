#creation of text summarization in map_reduce method using groq api key 
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


template1="""
you are an Ai assisstant please
provide me a summary of the following text
text:{text}
summary1: 
"""

prompt1=PromptTemplate(input_variables=["text"],template=template1)

template2="""
provide the final summary of the following text
and add some title,start with precise summary of an introduction
and provide the summmary in number points for the text
text:{text}
summary2:
"""

prompt2=PromptTemplate(input_variables=["text"],template=template2)

chain=load_summarize_chain(llm=llm,chain_type="map_reduce",map_prompt=prompt1,combine_prompt=prompt2,verbose=True)

response=chain.run(document)

print(response)
