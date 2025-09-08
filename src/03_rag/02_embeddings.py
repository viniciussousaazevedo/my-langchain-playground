from ..util import *

text = "Hugging Face embeddings with LangChain are easy to use!"
vector = embedding_model.embed(text)

print("Vector length:", len(vector))
print("First 5 values:", vector[:5])
