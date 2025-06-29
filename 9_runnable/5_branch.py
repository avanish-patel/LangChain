from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

generate_report_template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

summarize_report_template = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=["text"]
)

report_generation_chain = RunnableSequence(generate_report_template, model, parser)

branch_chain = RunnableBranch(
    # takes tuple of condition and runnable to execute when condition met
    # check if output from report_generation_chain has > 200 words, then summarize the report into smaller report by summarizing it
    (lambda x: len(x.split()) > 200, RunnableSequence(summarize_report_template, model, parser)),
    # else part just print the report using RunnablePassthrough
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

print(final_chain.invoke({"topic": "Indian stock market"}))
