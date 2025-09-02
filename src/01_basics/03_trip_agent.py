# This is a simple trip agent, which: # 1. Provides cities to go for the user
# # 2. Picks a random city from the output
# # 3. Tell some places to go on that city
# # 4. Think about how long it would be to visit some of these places on topic 3

from ..llm import *
from random import choice, randint, sample

def remove_random_elements(lst):
    """
    Removes random elements from a list and returns a new list.
    """
    count = randint(0, len(lst))
    indices_to_remove = set(sample(range(len(lst)), count))
    return [item for i, item in enumerate(lst) if i not in indices_to_remove]

def clean_and_split(text):
    """
    Clean and split comma-separated text into a clean list.
    """
    return [item.strip() for item in text.split(",") if item.strip()]


# -------------------- 1 - CHOOSING PLACES TO GO --------------------
places2go_sys_msg = """
You must suggest cities in different continents.
You must answer only the name of the cities separated by comma.
"""

places2go_prompt = ChatPromptTemplate.from_messages([
    ("system", places2go_sys_msg),
    ("user", "{input}"),
])

# -------------------- 2 - PLANNING AN ITINERARY --------------------
itinerary_sys_msg = """
You must answer only places to go separated by comma.
"""

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", itinerary_sys_msg),
    ("user", "Give me an itinerary for this city you have mentioned: {itinerary_city}"),
])

# -------------------- 3 - GETTING THE AMOUNT OF TIME --------------------
trip_duration_sys_msg = """
You only must say the total amount of hours necessary to visit all the places mentioned.
"""

trip_duration_prompt = ChatPromptTemplate.from_messages([
    ("system", trip_duration_sys_msg),
    ("user", "How long would it be to visit these places you have mentioned? {trip_duration_input}"),
])

# -------------------- MAIN CHAIN --------------------
main_chain = (
      llm.get_chain(places2go_prompt)
    | show_chain_data
    | {'itinerary_city': lambda x: choice(clean_and_split(x['answer']))}
    | show_chain_data
    | llm.get_chain(itinerary_prompt)
    | show_chain_data
    | {'trip_duration_input': lambda x: remove_random_elements(clean_and_split(x['answer']))}
    | show_chain_data
    | llm.get_chain(trip_duration_prompt)
    | show_chain_data
)

main_chain.invoke({"input": "tell me places to visit"})
