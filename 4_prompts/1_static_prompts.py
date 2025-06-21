from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.2,
#         max_new_tokens=50,
#     )
# )
#
# model = ChatHuggingFace(llm=llm)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
user_input = st.text_input("Enter your prompt!")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.text(result.content)

# to run app:  streamlit run <file_name>



