from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# define output schema for model
class Review(BaseModel):

    keywords: list[str] = Field(description="List of Main keywords in the review")
    summary: str = Field(description="Short Summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="List of pros in the Reviews")
    cons: Optional[list[str]] = Field(default=None, description="List of cons in the Reviews")

# pass schema to a model
structured_model = model.with_structured_output(Review)

# invoke model to get the structured output
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by A Patel""")

print(result)

print(dict(result))

# print("*****")
# print(result.keywords)
# print(result.summary)
# print(result.sentiment)
# print(result.pros)
# print(result.cons)
