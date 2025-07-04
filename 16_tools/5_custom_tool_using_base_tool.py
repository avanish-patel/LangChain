from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# All tool in LangChain are built on top of BaseTool

# define args schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(description="First input for multiply operation")
    b: int = Field(description="Second input for multiply operation")

# create class that extends BaseTool and define it's base properties
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: type[BaseTool] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()

print(multiply_tool.invoke({"a":3, "b":4}))

