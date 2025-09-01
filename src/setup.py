from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import *

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
)
divisor = '\n' + ('-'*20) + '\n'

def make_dict(v, k="input"):
    if isinstance(v, dict):
        return v
    return {k : v}

def llm_chain(prompt):
    return prompt | llm | {'answer': StrOutputParser()}

def chain_value_print(x):
    print("Current data in the chain: " + str(x), end=divisor)
    return make_dict(x)

# def chain_state_print():
    