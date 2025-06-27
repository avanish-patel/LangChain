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

# Projects

1. Document similarity search
> Have N numbers of documents, Have a query sentence and find documents that matches the highest similarity score with the query 

