from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field

load_dotenv()

# define model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# define schema
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The person's city")

# define parser
parser = PydanticOutputParser(pydantic_object=Person)

# define template
template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person. \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

# print(template.invoke({"place":"indian"}).text)
result = chain.invoke({"place":"indian"})

print(result)

# print(dict(result)["name"])