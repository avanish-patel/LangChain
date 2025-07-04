from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

print(search_tool.invoke("Top news in business today"))