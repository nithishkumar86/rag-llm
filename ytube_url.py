import os
from dotenv import load_dotenv
load_dotenv()
import validators,streamlit as st
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

st.set_page_config(page_title="yt and urlloader")
st.title("yt_loader or url_loader using groq api and stuff method")

with st.sidebar:
    groq_api=st.text_input("enter the your groq api key",type="password")


llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api)

template="""
As you are an expert in summarizing the text 
please provide me a summary of the context
context:{text}
"""

yt_url=st.text_input("enter the url")

if st.button("please enter summarize"):
    if not groq_api and not yt_url:
        st.warning("enter groq api key and url")

    elif not validators.url(yt_url):
        st.warning("enter valid url")

    else:
        try:
            with st.spinner("loading"):
                if "youtube.com" in yt_url:
                    loader=YoutubeLoader.from_youtube_url(yt_url)
                else:
                    st.warning("please enter a youtube url")

                docs=loader.load()

                prompt=PromptTemplate(input_varaibles=["text"],template=template)

                chain=load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
                response=chain.run(docs)

                st.success(response)

        except Exception as e:
            st.exception(f"exception:{e}")




