from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# def word_counter(text):
#     return len(text.split())
#
# runnable_word_counter = RunnableLambda(word_counter)
#
# print(runnable_word_counter.invoke("Hi there, How are you?"))

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Write me a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()


# function to count words in python, we will pass it to RunnableLambda to use it
def word_counter(text):
    return len(text.split())


joke_generator_chain = RunnableSequence(template, model, parser)

parallel_chain = RunnableParallel({
    # will print the joke which is part of 1st step
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_counter),
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({"topic": "AI"})

print("Joke: \n")
print(result["joke"])

print("Word Count: \n")
print(result["word_count"])
