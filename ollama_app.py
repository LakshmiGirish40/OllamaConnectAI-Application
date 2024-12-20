import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_Tracing_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "Hello, how are you?"),
        ("ai", "I'm doing well, thanks!"),
        ("user", "That's good to hear."),
        ("user", "Question: {question}")
    ]
)

# Streamlit Application
st.title("OllamaConnectAI Application")

# Dropdown for model selection
model_choice = st.selectbox(
    "Choose a model:",
    options=["gemma2:2b", "llama3.2","llama3.2:1b","gemma2"]
)

# Initialize the selected model
llm = Ollama(model=model_choice)  
output_parser = StrOutputParser()

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)

input_text = st.text_input("What question do you have in mind?")

if input_text:
    try:
        result = chain.run({"question": input_text})  # Execute the chain
        st.write(result)  # Display the result in Streamlit
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
