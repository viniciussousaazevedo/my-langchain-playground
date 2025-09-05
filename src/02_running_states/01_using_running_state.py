# This file implements a simple Knowledge Base class (powered by Pydantic) and updates it as the conversation with the user flows,
# creating a simple type of memory

from ..llm import *
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict, Union

class KnowledgeBase(BaseModel):
    topic: str = Field('general', description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field({}, description="User preferences and choices")
    session_notes: list = Field([], description="Notes on the ongoing session")
    unresolved_queries: list = Field([], description="Unresolved user queries")
    action_items: list = Field([], description="Actionable items identified during the conversation")

kb_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)

kb_update_sys_msg = """
    Your task is to update your Knowledge Base based on the latest conversation you had with the user.
    Your output must only be the Knowledge Base updated, in JSON format.
    OLD KNOWLEDGE BASE: {format_instructions}
    USER LAST INPUT: {input}
    YOUR ANSWER: {answer}
"""


# 1. usuário envia sua mensagem
# 2. modelo observa a base de conhecimento
# 3. modelo responde o usuário com base nos pontos 1 e 2
# 4. modelo atualiza a base de conhecimento com base nos pontos 1 e 3