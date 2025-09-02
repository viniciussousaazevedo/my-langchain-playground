from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import *

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
)
divisor = '\n' + ('-'*20) + '\n'

def llm_chain(prompt):
    return prompt | llm | {'answer': StrOutputParser()}

def show_chain_data(x):
    print(str(x), end=divisor)
    return x
    