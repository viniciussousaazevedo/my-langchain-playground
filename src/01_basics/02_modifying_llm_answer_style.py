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

# Here we had the same idea as the 01_hello_chain file, but now we add the context parameter to the chain,
# which is a game changer for the way the LLM answer. Run this file to see how different the outputs are!

# We can define as many parameters for the chains as we want. The only thing we must do is to have all the parameters
# that the chain is going to use set before it operates

# There are ways in which we can add more parameters dinamically (i.e. while the chain is running), I'll show this feature later on