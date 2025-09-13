# This file will perform just like the 02.01 file (running states),
# but this version will use embeddings with FAISS vector store to retrieve
# and update the Knowledge Base snippets

# 1. User send a message
# 2. Embedding model brings relevant pieces of the Knowledge Base from FAISS DB (pair: attribute name - value)
# 2. LLM answers user based on the contextualization provided by topic 2
# 4. LLM creates pairs of information to update the DB based the interaction that just happened and Knowledge Base template
# 5. Embedding model vectorizes the the output of topic 4 and updates FAISS DB content

from ..util import *
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

# -------------------- Knowledge Base description --------------------
knowledge_base_description = {
    "topic": "Current conversation topic",
    "user_preferences": "User preferences and choices",
    "session_notes": "Notes on the ongoing session",
    "unresolved_queries": "Unresolved user queries",
    "action_items": "Actionable items identified during the conversation"
}

# -------------------- Initialize empty FAISS convstore --------------------
embedding_dim = 1024  # BGE-Large-EN
convstore = FAISS(
    embedding_function=embedding_model.model,
    index=faiss.IndexFlatL2(embedding_dim),
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

def get_retriever():
    return convstore.as_retriever(search_kwargs={"k": 4})

# -------------------- Answering user --------------------
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
Your task is to update your Knowledge Base pairs based on the latest conversation you had with the user.
Your output must be the complete old knowledge base updated with new pieces of information based on the last conversation you had with the user. Each pair must be a separated key-value and each pair separated by " | ".

Fell free to update the old knowledge base (i.e. edit or remove content that is already there).
Output example: session_notes: the name of the user is Sara | user_preferences: User likes walking | user_preferences: User likes gaming

OLD KNOWLEDGE BASE: {knowledge_base_snippets}
FORMAT INSTRUCTIONS: {format_instructions}
USER LAST INPUT: {input}
YOUR ANSWER: {answer}
"""
kb_update_prompt = ChatPromptTemplate.from_messages([
    ('system', kb_update_sys_msg)
])

# -------------------- Knowledge Base retrieval and storage --------------------
def knowledge_base_update(x):
    """Reset the FAISS store and add the new knowledge base snippets safely."""
    convstore.docstore._dict.clear()
    convstore.index.reset()
    convstore.index_to_docstore_id.clear()

    snippets = x['new_data'].split(' | ')
    convstore.add_texts(snippets)

    return x

def get_whole_knowledge_base(_):
    """Return all snippets currently in the convstore."""
    return [doc.page_content for doc in convstore.docstore._dict.values()]

def retrieve_texts(x):
    """Retrieve top_k relevant snippets using the retriever."""
    retriever = get_retriever()
    docs = retriever.invoke(x['input'])
    return [d.page_content for d in docs]

# -------------------- Setting up the chain --------------------
main_chain = (
      RunnableAssign({'knowledge_base_snippets': get_whole_knowledge_base})
    | RunnableAssign({'relevant_data': retrieve_texts})
    | RunnableAssign({'answer': instruct_model.get_chain(answer_prompt)})
    | RunnableAssign({'new_data': instruct_model.get_chain(kb_update_prompt)})
    | RunnableLambda(knowledge_base_update)
)

# -------------------- User Interaction --------------------
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    chain_exec = main_chain.invoke({
        'input': user_input,
        'format_instructions': knowledge_base_description
    })

    print('Answer: ' + chain_exec['answer'], end=divisor)
