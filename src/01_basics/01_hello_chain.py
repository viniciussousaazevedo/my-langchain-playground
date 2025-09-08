# This is a simple hello for the LLM using chains
from ..util import *

prompt = ChatPromptTemplate.from_template("{input}")

print(instruct_model.get_chain(prompt).invoke({"input": "hello!"}))
