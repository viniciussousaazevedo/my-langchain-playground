# This file shows a simple query between vectorized data and queries using an embedding model to find the most relevant samples.
from ..util import *

data = [
    "I can't believe they did this to Gustave!",
    "Maelle was Verso's sister this whole time",
    "Verso is older than the whole expedition",
    "Verso wanted to be real all this time, just like his sister"
]

query = "Who is Verso's sister?"

embedded_data = [embedding_model.embed(d) for d in data]
embedded_query = embedding_model.embed(query)


