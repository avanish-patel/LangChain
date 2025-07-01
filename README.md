1. Create new virtual environment

```shell
python3 -m venv <environment_name>
```

2. Activate new virtual environment

```shell
source <environment_name>/bin/activate
```

3. Install dependencies from requirements.txt

```shell
pip3 install -r requirements.txt
```

4. Deactivate the environment

```shell
deactivate
```

# Models

1. LLM's
> Focus on Text Generation

2. Chat Models
> Interactive Conversational Focus

> Open source: Meta's LLaMA, Mistral etc...
> 
> Close source: OpenAI's GPT, Google's Gemini, Anthropic's Claude etc...

3. Embedding Models
> Use to convert text into embedding vectors

# Prompts

> Prompts are the input instructions or queries given to a model to guide its output.

1. Static
> A static prompt is a fixed instruction given to a language model that doesn't change during interaction.
2. Dynamic
> A dynamic prompt is a template with variables that can be filled in with different values, allowing the prompt to change based on the situation, user input, or program logic.


A `PromptTemplate` in Langchain is a structured way to create prompts dynamically by inserting variable into a predefined template.

PromptTemplate allows you to define placeholder that can be filled in at runtime with different inputs.

This makes it reusable, flexible, and easy to manage especially when working with dynamic user inputs or automated workflows.

# Messages

Messages are the unit of communication in chat models. They are used to represent the input and output of a chat model, as well as any additional context or metadata that may be associated with a conversation.

Each message has a role (e.g., "user", "assistant") and content (e.g., text, multimodal data) with additional metadata that varies depending on the chat model provider.

LangChain provides a unified message format that can be used across chat models, allowing users to work with different chat models without worrying about the specific details of the message format used by each model provider.

# Structured output

For many applications, such as chatbots, models need to respond to users directly in natural language. However, there are scenarios where we need models to output in a structured format. For example, we might want to store the model output in a database and ensure that the output conforms to the database schema. This need motivates the concept of structured output, where models can be instructed to respond with a particular output structure.

#### Ways to get structured output

1. LLM's that can generate Structured Output on its own : use `with_structured_output` function
2. LLM's that can't generate Structured Output on its own: use output parsers

## Using `with_structured_output`

For LLM's that supports structured output

1. TypedDict
2. Pydantic
3. JSON Schema

## Using output parsers

Output parsers in LangChain help converts raw LLM responses into structured formats like JSON, CSV, Pydantic and more.
They ensure consistency, validation and ease of use in applications.

# Chains

Chain is a basic building block that combines a prompt template with a language model. 
It takes input variables, formats them using the prompt template, and then passes the formatted text to the language model for processing.

- Sequencing
- Parallel
- Conditional

# Runnable

1. Task Specific Runnable

There are the core LangChain components that have been converted into Runnable so they can be used in pipeline.

Perform task specific operations like LLM calls, prompting, retrieval etc.

- `ChatOpenAI` -> runs an LLM Model
- `PromptTemplate` -> Formats prompts dynamically
- `Retriever` -> Retrieves relevant documents

2. Runnable Primitives

There are fundamental building blocks for structuring execution logic in AI workflows.

They help orchestrate execution by defining how different Runnable interact (sequentially, in parallel, conditionally etc.)

- `RunnableSequence` -> Runs steps in order (using | operator)
- `RunnableParallel` -> Runs multiple steps simultaneously
- `RunnableMap` -> Maps the same input across multiple functions
- `RunnableBranch` -> Implements conditional execution (if-else logic)
- `RunnableLambda` -> Wraps custom python function into Runnable
- `RunnablePassthrough` -> Just forward input as output (act as placeholder)

# Loader

Documents loaders are components in LangChain used to load data from various sources into a standardized format (usually as Document Object), which can be used for chunking, embedding, retrival and generation.

* `load()` : 
  * Eager loading (loads everything at once)
  * Returns: A list of `Document` objects
  * Loads all documents immediately into the memory
  * Best when the number of documents is small and want everything loaded upfront
  
* `lazy_load()` :
  * Lazy loading loads on demand
  * Returns: A generator of `Document` objects
  * Documents are not loaded at once, they're fetched one at a time as needed
  * Best when dealing with large documents or lots of files and you want stream processing without using lots of memory

```shell
Document(
  page_content="The actual text content",
  metadata={"source":"filename.pdf",...}
)
```

* `TextLoader` : `TextLoader` is simple and commonly used document loader in LangChain that reads plain text .txt files nad converts them into LangChain Document Objects.
* `PyPDFLoader`: `PyPDFLoader` is a document loader in LangChain used to load content from PDF files and convert each page into a Document Object.
* `DirectoryLoader`: `DirectoryLoader` is a document loader that lets you load multiple documents from a directory of files.
* `WebBaseLoader` : `WebBaseLoader` is a document loader in LangChain used to load and extract content from web pages(URLs). It uses `BeautifulSoup` under the hood to parse HTML and extract visible context.

# Text Splitter

https://chunkviz.up.railway.app/

1. Length Based : `CharacterTextSplitter` divide text/document based on character length
2. Text Structured Based: Text can be divided based on its structure like paragraph, sentence, word, characters recursively using `RecursiveCharacterTextSplitter`
3. Document Structured Based: Documents which are not just plain text, such as markdown file, code document can be divided using `RecursiveCharacterTextSplitter`
4. Semantic Meaning: Sometime text hold tricky meaning in small part and dividing using above two methods can ruin the meaning of original text (if text holds different meaning in smaller text), In this scenario taking semantic meaning and splitting text would be best approach

# Vector Store

a vector store is a database optimized for storing and searching high-dimensional vectors, often generated from text or other data using embedding models. It allows for efficient similarity searches, finding documents or data points that are semantically similar to a given query.

Chroma is a lightweight, open source vector database that is specially friendly for local development and small to medium scale production needs.
DB -> Collection -> Docs

# Retrievers

A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user query.

There are multiple types of retrievers and they are runnables.

Can be divide based on...

- Data sources - Wikipedia, Vector store, Archive retrievers based on data source
- Retriever search strategy - Maximum Marginal Relevance (MMR), Multi Query, Contextual Compression retrieval etc.

> MMR : MMR is an information retrival algorithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query.
> In regular similarity search, we might get documents that are very similar to each other, repeating info and lack diverse perspective
> MMR avoids that by picking the most relevant info first

> Multi-Query Retrieval: Sometimes single query might not capture all the ways information is phrased in your documents.
> In this scenario, simple similarity search might miss documents that talk about those things but don't use the main word.
> 
> Query: How can I stay healthy?
> 
> Could mean:
> * What should I eat?
> * How often should I exercise?
> * How can I manage stress?
> 
> How it works:
> 1. Take your original query
> 2. Uses an LLM to generate multiple semantically different version of that query
> 3. Performs retrival for each sub query
> 4. Combines and deduplicates the results

# Projects

