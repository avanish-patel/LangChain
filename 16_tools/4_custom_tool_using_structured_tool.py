from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# define argument schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(description="First input for multiply operation")
    b: int = Field(description="Second input for multiply operation")

def multiply(a, b):
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

print(multiply_tool.invoke({"a": 5, "b": 2}))

