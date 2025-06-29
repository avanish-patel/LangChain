from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# define splitter
text_splitter = SemanticChunker(
    # OpenAI for embedding
    OpenAIEmbeddings(),
    # provide threshold type to split when meaning changes in the text
    breakpoint_threshold_type="standard_deviation",
    # if standard deviation amount reaches 1 then it breaks int the chunk from above standard deviation
    breakpoint_threshold_amount=1,
)

# create semantic based chunked document based on meaning
chunks = text_splitter.create_documents([text])

print(len(chunks))
print(chunks)
