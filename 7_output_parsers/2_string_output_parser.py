from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# TinyLlama doesn't support structured output out the box so we need to use output parser : runs model locally
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# 1st prompt -> detailed report
detailed_report_template = PromptTemplate(
    template="Write a details report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary
summary_report_template = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

# define parser
parser = StrOutputParser()

# parser is going to parser result in form of string and will pass it to next item in a chain (Output of first item become input of next item in a chain)
chain = detailed_report_template | model | parser | summary_report_template | model | parser

result = chain.invoke({"topic":"Black Hole"})

print(result)