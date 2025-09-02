# This file will show some of the existing Runnables and how to operate with them properly.
# This example will show the following:
# 1. Receives a text.
# 2. Cleans and normalizes the text.
# 3. Classifies the text's mood (happy, neutral, sad).
# 4. Observes the text's length, determining which model will respond based on it.
# 5. Creates a personalized message based on the mood.
# 6. Formats the output as a JSON object.

from ..llm import *
