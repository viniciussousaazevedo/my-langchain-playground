# This is a simple hello for the LLM using chains

from ..setup import *
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{input}")

print(llm_chain(prompt).invoke({"input": "hello!"}))
