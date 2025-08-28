from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
)

def prompt_chain(system_message: str = ""):
    prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "{input}")
    ])
    return prompt | llm | StrOutputParser()


