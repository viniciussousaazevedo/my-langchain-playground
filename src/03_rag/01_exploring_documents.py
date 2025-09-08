# This sets up a simple document loader and a simple QA agent,
# which works with a primitive iteration-based query engine over document chunks of data.
# This is not effective, but I wanted to build this in order to show how document chunks and loaders work.

from ..llm import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader

## Unstructured File Loader: Good for arbitrary "probably good enough" loader
# documents = UnstructuredFileLoader("llama2_paper.pdf").load() # This loads local docs
paper = ArxivLoader(query="2404.16130").load()  ## GraphRAG paper extracted direcly from Arxiv API

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

splitted_paper = text_splitter.split_documents(paper)

def look_over_documents(x):
    for chunk in x['document']:
        safe_content = chunk.page_content.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ('system', f"Does this document snippet answers the question below? If so, answer it normally. If not, just say 'NO', and nothing more\n\n{x['input']}"),
            ('user', safe_content)
        ])
        answer = llm.get_chain(prompt).invoke(x)

        if 'NO' not in answer:
            return answer
    return 'Question could not be answered based on the document loaded'

dict_input = {
    'document': splitted_paper,
    'input': "When was this paper published?"
}

print(RunnableLambda(look_over_documents).invoke(dict_input))
