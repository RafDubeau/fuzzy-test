from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType


# Initialize LLM object
llm = ChatOpenAI(temperature=0)

tools = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.invoke({
    "input": "What is the distance between the capital of France and the capital of Nova Scotia? Is it possible to drive in a car between these two locations?"
})