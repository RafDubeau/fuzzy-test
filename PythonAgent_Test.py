from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.run("Generate the first 10 prime numbers")

# agent_executor.run("Generate the first 10 Mersenne prime numbers")

# agent_executor.run(
#     """Understand, write a single neuron neural network in PyTorch.
# Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
# Return prediction for x = 5"""
# )
