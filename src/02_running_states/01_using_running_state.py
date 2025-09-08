# This file implements a simple Knowledge Base class (powered by Pydantic) and updates it as the conversation with the user flows,
# creating a simple type of memory
# 1. User send a message
# 2. LLM take a look at its Knowledge Base
# 3. LLM answers the user based on topics 1 and 2
# 4. LLM updates its Knowledge Base based on topics 1 and 3

from ..llm import *
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict, Union

# -------------------- Setting up the Knowledge Base Object --------------------
class KnowledgeBase(BaseModel):
    topic: str = Field('general', description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field({}, description="User preferences and choices")
    session_notes: list = Field([], description="Notes on the ongoing session")
    unresolved_queries: list = Field([], description="Unresolved user queries")
    action_items: list = Field([], description="Actionable items identified during the conversation")

kb_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)
knowledge_base = KnowledgeBase()

# -------------------- Answering user based on Knowledge Base --------------------
answer_sys_msg = """
    Your task is to answer the user based on the Knowlodge Base below:
    KNOWLEDGE BASE: {knowledge_base}
"""
answer_prompt = ChatPromptTemplate.from_messages([
    ('system', answer_sys_msg),
    ('user', '{input}')
])

# -------------------- Knowledge Base Update --------------------
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

# -------------------- Setting up the chain --------------------
main_chain = (
    RunnableAssign({'answer': llm.get_chain(answer_prompt)})
    | RunnableAssign({'knowledge_base': kb_update_prompt | llm.llm | kb_parser})
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
