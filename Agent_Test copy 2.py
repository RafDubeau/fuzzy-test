from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

import asyncio

# Initialize LLM agent
llm = ChatOpenAI(temperature=0.7)

# Create sync browser
browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
tools = toolkit.get_tools()

# tools_by_name = {tool.name: tool for tool in tools}
# navigate_tool = tools_by_name["navigate_browser"]
# get_elements_tool = tools_by_name["get_elements"]


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

async def execute(query: str) -> asyncio.coroutine:
    await agent_chain.arun(query)

result = asyncio.run(execute("How old is Joe Biden?"))
print(result)