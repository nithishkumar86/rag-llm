import os
from dotenv import load_dotenv
load_dotenv()
import validators,streamlit as st
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

st.set_page_config(page_title="urlloader")
st.title("url_loader using groq api and map_reduce method")

with st.sidebar:
    groq_api=st.text_input("enter the your groq api key",type="password")


llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api)


template1="""
you are an Ai assisstant please
provide me a summary of the following text
text:{text}
summary1: 
"""


template2="""
provide the final summary of the following text
and add some title,start with precise summary of an introduction
and provide the summmary in number points for the text
text:{text}
summary2:
"""

urlloader=st.text_input("enter the url")

if st.button("enter to process"):
    if not groq_api and not urlloader:
        print("enter groq api key and url")

    elif not validators.url(urlloader):
        print("enter valid url")

    else:
        try:
                #https://en.wikipedia.org/wiki/Deep_learning
                import re
                #mass="https://en.wikipedia.org/wiki/Deep_learning"
                if re.findall(".org",urlloader):
                    loader=UnstructuredURLLoader(urls=[urlloader])
                else:
                    st.warning("please enter a url in website in org file")

                docs=loader.load()

                split=CharacterTextSplitter(chunk_size=400,chunk_overlap=50)
                document=split.split_documents(docs)

                prompt1=PromptTemplate(input_variables=["text"],template=template1)

                prompt2=PromptTemplate(input_variables=["text"],template=template2)
                

                chain=load_summarize_chain(llm=llm,chain_type="map_reduce",map_prompt=prompt1,combine_prompt=prompt2)
                response=chain.run(document[:50])

                st.success(response)

        except Exception as e:
            print(f"exception:{e}")