from langchain.chains.llm_math.base import LLMMathChain,LLMChain
from langchain import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentType,initialize_agent,Tool
from langchain_community.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="langchain using mathematic function")
st.title("Mathematic solving problems")

st.sidebar.title("settings")
groq_api_key=st.sidebar.text_input(label="enter the Groq api key",type="password")

if not groq_api_key:
    st.info("please enter your grog api key")
    st.stop()

llm=ChatGroq(model="gemma2-9b-it",api_key=groq_api_key)

wikipedia=WikipediaAPIWrapper()
wiki=Tool(name="wikipedia",
     func=wikipedia.run,
     description="wikipedia documentation")

#combine LLMMathChain with llm to ceate a tool to solve with math problem
math=LLMMathChain.from_llm(llm=llm)
maths=Tool(
    name="llmmath",
    func=math.run,
    description="llmath to solve math problems"
)

templates="""
As you are an ai expert in mathematics please provide the answer for following math question
question:{text}
Answer:
"""

prompt=PromptTemplate(input_variables=["text"],
              template=templates)

chain=LLMChain(llm=llm,prompt=prompt)
chains=Tool(
    name="chain",
    func=chain.run,
    description="i am ready to answer you question"
)

agent=initialize_agent(tools=[wiki,maths,chains],llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

if 'messages' not in st.session_state:
    st.session_state.messages=[{"role":"assisstant","content":"what can i assisst you"}]

#TypeError: string indices must be integers, not 'str'[] forgot to give some list 

for mes in st.session_state.messages:
    st.chat_message(mes['role']).write(mes['content'])

question=st.text_input("enter the maths question to solve")

if st.button("find my answer"):
    if question:
        with st.spinner("loading..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            sq_it=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response=agent.run(st.session_state.messages,callbacks=[sq_it])
            st.session_state.messages.append({"role":"assisstance","content":response})
            st.success(response)
else:
    st.error("enter the question")

