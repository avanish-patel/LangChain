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


# Projects

1. Document similarity search
> Have N numbers of documents, Have a query sentence and find documents that matches the highest similarity score with the query 

