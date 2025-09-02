# This is a simple hello for the LLM using chains

from ..llm import *

prompt = ChatPromptTemplate.from_template("{input}")

print(llm.get_chain(prompt).invoke({"input": "hello!"}))
