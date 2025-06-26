from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

# define schema
class Review(TypedDict):
    summary: str
    sentiment: str

# create new mode with structured output, behind the scene it generates prompt to return json output with these attributes
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but hte software feels bloated. There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)

print(result["summary"])
print(result["sentiment"])