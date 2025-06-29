from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

# following 3 are Runnable components prompt, model and parser
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# pass all 3 runnable into RunnableSequence
# we can also use | operator also called LangChain Expression Language Operation (since sequential chain is most widely used in Langchain framework)
chain = RunnableSequence(prompt, model, parser)

print(chain.invoke({"topic": "AI"}))
