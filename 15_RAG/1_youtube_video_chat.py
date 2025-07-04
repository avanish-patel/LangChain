from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# a. Indexing (Document ingestion)
video_id = "__4Vbt9Hs50"

transcript = ""
try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(video_id=video_id, languages=["en"])

    # flatten it to plain text
    transcript = " ".join(snippets.text for snippets in transcript_list)

    # print(transcript)

except Exception as e:
    print("Failed to fetch transcript", e)

# b. Indexing (Document splitting)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

document_chunks = splitter.create_documents([transcript])

# print(document_chunks)


# c. Indexing (Embedding generation & storing into Vector store)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(document_chunks, embedding)

# Retriever ( retriever returns List of Documents objects)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# print(retriever.invoke("What is meaning to success?"))


# Augmentation


# question to ask to llm
question = "Is there a discussion about American businesses? if yes then what was discussed?"

# get docs that matches question from store using retriever
similar_docs = retriever.invoke(question)

# convert docs to string
context_text = "\n\n".join(doc.page_content for doc in similar_docs)

# print(context_text)


# Generation


template = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"],
)

prompt = template.invoke({"context": context_text, "question": question})

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

answer = model.invoke(prompt)

print(answer.content)
