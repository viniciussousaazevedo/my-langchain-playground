# This file will show some of the existing Runnables and how to operate with them properly.
# This example will show the following:
# 1. Receives a text.
# 2. Cleans and normalizes the text.
# 3. Classifies the text's mood (happy, neutral, sad).
# 4. Observes the text's length, determining which model will respond based on it.
# 5. Creates a personalized message based on the mood.
# 6. Formats the output as a JSON object.

from ..llm import *
limited_model = LLM("llama-3.1-8b-instant")

def normalize(text: str) -> str:
    # Simple normalization function
    return ' '.join(text.strip().split()).lower()
normalize_runnable = RunnableLambda(lambda x: {"input": normalize(x["input"]), **x})

mood_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "you must classify the user input in only one word contained in these options: {options}"),
    ("user", "{input}")
])

llm_instance = RunnableBranch(
    (lambda x: len(x["input"].split()) < 10, limited_model),
    llm
)

main_chain = (
    normalize_runnable
    mood_classifier_prompt
    | show_chain_data
    | # TODO
    
)

main_chain.invoke({"options": ['sad', 'angry', 'happy', 'confused'], "input": "I just got promoted in my job!"})
