import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import Ollama
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import initialize_agent,AgentType
st.title("model project for creating tools and agent in langchain")

st.sidebar.title("settings")
groq_api_key=st.sidebar.text_input("enter your groq api key",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assisstance","content":"hey come and chat with me"}]

#reading the chat messages using for loop
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt:=st.chat_input("if any question you can ask me"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)


llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key,streaming=True)

arxivwrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxivwrapper)

wikipediawrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wikipediawrapper)


search=DuckDuckGoSearchRun(name="Search")

tools=[search,arxiv,wiki]

"""
ZERO_SHOT_REACT_DESCRIPTION is a type of agent that performs a reasoning step before 
taking action. It does not rely(nambi) on any chat(arattai) history, meaning it makes decisions 
based solely on the current input. This makes it suitable for scenarios where an
immediate response is required without the need for context from previous interactions.

On the other hand, CHAT_ZERO_SHOT_REACT_DESCRIPTION also performs a reasoning step
before acting, but unlike `ZERO_SHOT_REACT_DESCRIPTION`, it uses a chat history 
variable in the prompt. This means that the final prompt will include the chat history,
allowing the agent to remember the context of the chat and the history of the 
conversation. This agent type is designed for multi-turn tasks that require maintaining
the context of a conversation.
"""
search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=False)

if st.button("enter to show your answer"):
    st_ip=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
    response=search_agent.run(st.session_state.messages,callbacks=[st_ip])
    st.session_state.messages.append({"role":"assisstant","content":response})
    st.success(response)


st.write(st.session_state["messages"])