# This sets up a simple document loader and a simple QA agent,
# which works with a primitive iteration-based query engine over document chunks of data.
# This is not effective, but I wanted to build this in order to show how document chunks and loaders work.

from ..llm import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader

## Unstructured File Loader: Good for arbitrary "probably good enough" loader
# documents = UnstructuredFileLoader("llama2_paper.pdf").load() # This loads local docs
documents = ArxivLoader(query="2404.16130").load()  ## GraphRAG paper extracted direcly from Arxiv API

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

docs_split = text_splitter.split_documents(documents)

def look_over_documents():
    for chunk in docs_split:
        prompt = ChatPromptTemplate.from_messages([
            ('system', "Does this document snippet answers the question below? If so, answer it normally. If not, just say 'NO', and nothing more")
            ('user', chunk)
        ])
        answer = llm.get_chain(prompt).invoke()

        if 'NO' in answer:
            continue
        return answer
