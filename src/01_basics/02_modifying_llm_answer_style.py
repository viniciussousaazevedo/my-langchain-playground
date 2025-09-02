# This is an example about how we can dinamically change the context provided in chains

from ..setup import *
from langchain_core.prompts import ChatPromptTemplate

contexts = [
    "you are almost freezing",
    "rhymes only",
    "you are Goku"
]

system_message = "you must answer the user shortly and following the context provided: {context}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "{input}")
])

for context in contexts:
    print(llm_chain(prompt).invoke({"input": "What do you like to do?", "context": context}), end=divisor)
