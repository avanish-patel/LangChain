# Feedback
# Analyze [positive/negative]
# if Positive -> give it to model to handle positive response
# if Negative -> give it to model to handle negative response

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


# define schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(default="Give the sentiment of the feedback text")


# define parser
string_parser = StrOutputParser()
feedback_parser = PydanticOutputParser(pydantic_object=Feedback)

# define template
sentiment_template = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": feedback_parser.get_format_instructions()}
)

positive_response_template = PromptTemplate(
    template="Write an appropriate response for positive feedback. \n {feedback}",
    input_variables=["feedback"]
)

negative_response_template = PromptTemplate(
    template="Write an appropriate response for negative feedback. \n {feedback}",
    input_variables=["feedback"]
)

# define model
llm = HuggingFaceEndpoint(
    # repo_id="google/gemma-2-2b-it",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# feedback_sentiment = classifier_chain.invoke({"feedback":" This is a terrific smartphone"})
# print(feedback_sentiment.sentiment)

branch_chain = RunnableBranch(
    # takes tuple, (condition, chain)
    # if positive run this chain
    (lambda x: x.sentiment == "positive", positive_response_template | model | string_parser),
    # if negative run this chain
    (lambda x: x.sentiment == "negative", negative_response_template | model | string_parser),
    # if neither above do this (else part)
    RunnableLambda(lambda x: "Could not find sentiment!")
)

classifier_chain = sentiment_template | model | feedback_parser

chain = classifier_chain | branch_chain

# try different type of feedback to see how model write response to it
print(chain.invoke({"feedback": "This is a terrible smartphone with awesome features"}))


# visualize chain
# chain.get_graph().print_ascii()