# Whats up! Let's talk about LangChain's core feature: Chains (actually, these are Runnables, but the concept is the same)
# Chains are just like jigsaw pieces: It does not make sense alone, but together they form a beautyfull art!

# You define how things should work in every chain, connect them and you use it later with the values you want to!
# In this file we have a simple use case of chains: invoking an LLM and getting it's answer


# I'm gonna use this import in every .py file. It does everything I've explained above
from util.model_initalization import *

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

print(llm_chain(prompt).invoke({"input": "hello!"}))

# Let's break this down:

# the 'llm_chain' function is called (therefore, we have a chain 'prompt | llm | StrOutputParser()'
# the 'prompt' chain is filled with the 'input' value declared at the invoke function
# this means that the prompt chain is formed with a user message 'hello!'

# now, we have this pipe ('|') thing, which is passing the output of the previous chain as an input to the next chain
# so, the 'llm' chain is now being invoked with the prompt chain output to perform
# the 'ChatGroq' object already know what to do with the 'ChatPromptTemplate.from_messages' function, so it just gives an answer right away

# finally, we have another pipe that brings the output of the LLM to the 'StrOutputParser()'
# This is a simple function that extracts the string answer from the LLM chain answer object, as simple as that

# Now, our chain is finally over, and we can see the answer directly! Magical, isn't it? 