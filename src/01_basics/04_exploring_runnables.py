# This file will show some of the existing Runnables and how to operate with them properly.
# This example will show the following:
# 1. Receives a text.
# 2. Cleans and normalizes the text.
# 3. Classifies the text's mood (happy, neutral, sad).
# 4. Observes the text's length, determining which model will respond based on it.
# 5. Creates a personalized message based on the mood.
# 6. Formats the output as a JSON object.

from ..llm import *

def normalize(text: str) -> str:
    # Simple normalization function)
    return " ".join(text.strip().split()).lower()
normalize_runnable = RunnableAssign({'input': lambda x: normalize(x['input'])})

mood_classifier_prompt = ChatPromptTemplate.from_messages([
    ('system', "you must classify the user input in only one word contained in these options: {options}. You must also declare your model name at the end like the example provided, in parenthesis (e.g. Llama, GPT, Mistral, etc)."),
    ('user', "I just got fired :("),
    ('assistant', "sad ([Your Name])"),
    ('user', "{input}")
])

model_selector = RunnableBranch(
    (lambda x: len(x['input'].split()) < 10, lambda x: LLM("meta-llama/llama-4-scout-17b-16e-instruct")),
    lambda x: LLM(),
)

main_chain = (
    normalize_runnable
    | show_chain_data
    | RunnableAssign({'model': model_selector})
    | show_chain_data
    | RunnableLambda(lambda x: x['model'].get_chain(mood_classifier_prompt))
    | show_chain_data
)
main_chain.invoke({'options': ['sad', 'angry', 'happy', 'confused'], 'input': "I   just got promoted in my job!"})
