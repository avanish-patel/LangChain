from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# define template
template = PromptTemplate(
    template="Generate 3 interesting facts about {topic}",
    input_variables=["topic"]
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
chain = template | model | parser

result = chain.invoke({"topic":"cricket"})
print(result)


# visualize chain
chain.get_graph().print_ascii()