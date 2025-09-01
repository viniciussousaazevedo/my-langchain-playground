from ..setup import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import *
from random import choice

places2go_sys_message = """
You must suggest cities in different continents.
You must answer only the name of the cities separated by comma.
"""

places2go_prompt = ChatPromptTemplate.from_messages([
    ("system", places2go_sys_message),
    ("user", "{input}"),
])

itinerary_sys_message = """
You must answer only places to go separated by comma.
"""

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", itinerary_sys_message),
    ("user", "Give me an itinerary for this city you have mentioned: {itinerary_city}"),
])

trip_price_sys_message = """
You must answer the price in US Dollars, only.
"""

trip_price_prompt = ChatPromptTemplate.from_messages([
    ("system", trip_price_sys_message),
    ("user", "{trip_price_input}"),
])

main_chain = (
      llm_chain(places2go_prompt)
    | RunnableAssign({'itinerary_city': lambda state: choice(state['answer'].split(", "))})
    | chain_value_print
)

main_chain.invoke({"input": "tell me places to visit"})