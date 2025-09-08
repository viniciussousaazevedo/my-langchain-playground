from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import *
from langchain_huggingface import HuggingFaceEmbeddings

class InstructModel:
    def __init__(self, name="openai/gpt-oss-120b"):
        self.model = ChatGroq(
        model=name,
        temperature=0.0,
    )

    def get_chain(self, prompt):
        return prompt | self.model | StrOutputParser()

class EmbeddingModel:
    def __init__(self, name="BAAI/bge-large-en"):
        self.model = HuggingFaceEmbeddings(model_name=name)
    
    def embed(self, text):
        return self.model.embed_query(text)


instruct_model = InstructModel()
embedding_model = EmbeddingModel()
divisor = '\n' + ('-'*20) + '\n'

def show_chain_data(x):
    print(str(x), end=divisor)
    return x
        