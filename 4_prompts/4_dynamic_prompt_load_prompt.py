from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool using Dynamic Prompt")

# Create 3 dropdowns to select values
paper_input = st.selectbox(
    "Select Research Paper Name",
    options=[
        "Attention is all you need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language models are few-shot learners",
        "Diffusion models beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    options=[
        "Beginner Friendly",
        "Technical",
        "Code Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    options = [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ]
)

# load prompt
template = load_prompt("./4_prompts/research-tool-template.json")

# fill placeholders
prompt = template.invoke({
    "paper_name": paper_input,
    "style": style_input,
    "length": length_input,
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)

# streamlit run 4_prompts/2_dynamic_prompt.py