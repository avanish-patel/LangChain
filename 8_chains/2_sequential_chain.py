from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# define template
detailed_template = PromptTemplate(
    template="Generate detailed report on {topic}",
    input_variables=["topic"]
)

summary_template = PromptTemplate(
    template="Generate a 3 pointer summary from the following text. \n {report}",
    input_variables=["report"]
)

# define model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# define parser
parser = StrOutputParser()


# define chain
chain = detailed_template | model | parser | summary_template | model | parser

result = chain.invoke({"topic":"US Stock Market"})

print(result)


# visualize graph
chain.get_graph().print_ascii()

