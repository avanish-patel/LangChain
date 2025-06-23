from langchain_core.prompts import PromptTemplate

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

template.save("research-tool-template.json")