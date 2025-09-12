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

def split_knowledge_base(kb_obj):
    for attribute, value in kb_obj.dict().items():
        print(attribute, value)
    
print(kb_parser.get_format_instructions())
split_knowledge_base(knowledge_base)