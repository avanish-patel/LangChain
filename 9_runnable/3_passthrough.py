from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

# following 3 are Runnable components prompt, model and parser
generate_joke_prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

explain_joke_prompt = PromptTemplate(
    template="Explain the joke \n {joke}",
    input_variables=["joke"]
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# runnable_passthrough = RunnablePassthrough()
# print(runnable_passthrough.invoke(2))
# print(runnable_passthrough.invoke({"topic":"LangChain"}))

joke_generator_chain = RunnableSequence(generate_joke_prompt, model, parser)

# this parallel chain print the joke using RunnablePassthrough which came from joke_generator_chain and also RunnableSequence to generate explanation
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(explain_joke_prompt, model, parser)
})

chain = RunnableSequence(joke_generator_chain, parallel_chain)

print(chain.invoke({"topic": "Cricket Game"}))
