from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

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

# create prompt template that takes variables
template = PromptTemplate(
    template="""
    Please summarize the research paper title "{paper_name}" with the following specification:
    Explanation style: {style}
    Explanation length: {length}
    1. Mathematical Details:
      - Include relevant mathematical equation if present in the paper.
      - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies
      - Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with: Insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=["paper_name", "style", "length"],
)

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