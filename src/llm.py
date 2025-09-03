from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import *
from operator import itemgetter

class LLM:
    def __init__(self, name="openai/gpt-oss-120b"):
        self.llm = ChatGroq(
        model=name,
        temperature=0.0,
    )

    def get_chain(self, prompt):
        return prompt | self.llm | {'answer': StrOutputParser()}

llm = LLM()
divisor = '\n' + ('-'*20) + '\n'
def show_chain_data(x):
    print(str(x), end=divisor)
    return x
        