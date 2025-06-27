from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# Define model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# define parser
parser = JsonOutputParser()

# define template
template = PromptTemplate(
    template="Give me the name, age, and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# print(template.format())

chain = template | model | parser

# Alternatively, I can use following if I don't want to use chain
# prompt = template.format()
# result = model.invoke(prompt)
# final_result=parser.parse(result.content)

result = chain.invoke({})

print(result)

