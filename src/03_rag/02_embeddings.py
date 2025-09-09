
# This does not use any vector store for now, comparing cosine similarity manually
# This file shows a simple query between vectorized data and queries using an embedding model to find the most relevant samples.
from ..util import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = [
    "I can't believe they did this to Gustave!",
    "Maelle was Verso's sister this whole time",
    "Verso is older than the whole expedition",
    "Verso wanted to be real all this time, just like his sister"
]

query = "Who is Verso's sister?"

embedded_data = np.array([embedding_model.embed(d) for d in data])
embedded_query = np.array(embedding_model.embed(query)).reshape(1, -1)

scores = cosine_similarity(embedded_query, embedded_data)[0]

ranked_results = sorted(zip(data, scores), key=lambda x: x[1], reverse=True)

print("Query:", query)
print("\nTop results:")
for text, score in ranked_results:
    print(f"({score:.4f}) {text}")
