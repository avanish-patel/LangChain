from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

hf_model = ChatHuggingFace(llm=llm)
openai_model = ChatOpenAI()

linkedin_template = PromptTemplate(
    template="Generate LinkedIn post about {topic}",
    input_variables=["topic"]
)

facebook_template = PromptTemplate(
    template="Generate Facebook post about {topic}",
    input_variables=["topic"]
)

string_parser = StrOutputParser()

# RunnableParallel takes dictionary as input
parallel_chain = RunnableParallel({
    "facebook": RunnableSequence(facebook_template, hf_model, string_parser),
    "linkedIn": RunnableSequence(linkedin_template, hf_model, string_parser),
})

result = parallel_chain.invoke({"topic": "LangChain"})

print("Facebook post: \n")
print(result["facebook"])

print("LinkedIn post: \n")
print(result["linkedIn"])
