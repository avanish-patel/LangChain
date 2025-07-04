from langchain_community.tools import ShellTool

shell_tool = ShellTool()

print(shell_tool.invoke("time"))

print(shell_tool.invoke("ls"))