# This file will perform just like the 02.01 file (running states),
# but this version will use embeddings with FAISS vector store to retrieve
# and update the Knowledge Base Pydantic object

# 1. User send a message
# 2. Embedding model brings relevant pieces of the Knowledge Base from FAISS DB (pair: attribute name - value)
# 2. LLM answers user based on the contextualization provided by topic 2
# 4. LLM creates pairs of information to update the DB based the interaction that just happened and Knowledge Base template
# 5. Embedding model vectorizes the the output of topic 4 and updates FAISS DB content

from ..util import *
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict, Union
from langchain.vectorstores import FAISS

# -------------------- Setting up the Knowledge Base Object and FAISS database --------------------
class KnowledgeBase(BaseModel):
    topic: str = Field('general', description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field({}, description="User preferences and choices")
    session_notes: list = Field([], description="Notes on the ongoing session")
    unresolved_queries: list = Field([], description="Unresolved user queries")
    action_items: list = Field([], description="Actionable items identified during the conversation")

kb_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)
knowledge_base = KnowledgeBase()

def (kb_obj):
    str_obj = str(kb_obj)


convstore = FAISS.from_texts(, embedding=embedding_model.model)
retriever = convstore.as_retriever()

# -------------------- Answering user based on Knowledge Base snippets --------------------
answer_sys_msg = """
    Your task is to answer the user. Only if appropriate, use the pieces of information provided below to enhance your answer contextualization.
    INFORMATION: {relevant_data}
"""
answer_prompt = ChatPromptTemplate.from_messages([
    ('system', answer_sys_msg),
    ('user', '{input}')
])

# -------------------- Updating Knowledge Base --------------------
kb_update_sys_msg = """
    Your task is to update your Knowledge Base based on the latest conversation you had with the user.
    Your output must only be the Knowledge Base updated, in JSON format.
    OLD KNOWLEDGE BASE: {knowledge_base}
    FORMAT INSTRUCTIONS: {format_instructions}
    USER LAST INPUT: {input}
    YOUR ANSWER: {answer}
"""
kb_update_prompt = ChatPromptTemplate.from_messages([
    ('system', kb_update_sys_msg)
])

# -------------------- Vectorizing and storing the updated Knowledge Base --------------------
# TODO

# -------------------- Setting up the chain --------------------
main_chain = (
    RunnableAssign({'answer': instruct_model.get_chain(answer_prompt)})
    | RunnableAssign({'knowledge_base': kb_update_prompt | instruct_model.model | kb_parser})
)

# -------------------- User Interaction --------------------
while True:
    user_input = input("User: ")
    if user_input == 'exit':
        break

    chain_exec = main_chain.invoke({
        'input': user_input,
        'knowledge_base': knowledge_base,
        'format_instructions': kb_parser.get_format_instructions()
    })
    knowledge_base = chain_exec['knowledge_base']
    print('Answer: ' + chain_exec['answer'], end=divisor)
