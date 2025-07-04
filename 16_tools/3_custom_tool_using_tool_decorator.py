from langchain.tools import tool


# step 1: create function
# step 2: add type hints and document comment
# step 3: add tool decorator

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# tool is runnable so we can call using invoke
result = multiply.invoke({"a":3, "b":4})

print(result)


print(multiply.name)
print(multiply.description)
print(multiply.args)

# tool schema is what LLM sees not just plain python function
print(multiply.args_schema.model_json_schema())