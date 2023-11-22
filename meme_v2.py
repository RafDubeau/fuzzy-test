from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from colorama import Fore


suggestions = []


# Initialize LLM object
llm = ChatOpenAI(temperature=0.5, model="gpt-4")


@StructuredTool.from_function
def create_attached_suggestion(item_id: str, suggestion_text: str) -> str:
    """Creates a document for a suggestion relating to a specific item."""
    suggestions.append(
        f"Suggestion attached to {Fore.BLUE}{item_id}{Fore.RESET}: {suggestion_text}"
    )
    return str(suggestions)


@StructuredTool.from_function
def create_tile_suggestion(item_id: str, suggestion_text: str) -> str:
    """Creates a document for a suggestion to be displayed following a specific item."""
    suggestions.append(
        f"Tile suggestion following {Fore.BLUE}{item_id}{Fore.RESET}: {suggestion_text}"
    )

    # write function debug_now(suggestions) to print out the suggestions geenrated to the console for debugging

    return str(suggestions)


@StructuredTool.from_function
def create_global_suggestion(suggestion_text: str) -> str:
    """Creates a document for a suggestion relating to the entire itinerary."""
    suggestions.append(f"{Fore.BLUE}Global{Fore.RESET}: {suggestion_text}")
    return str(suggestions)


agent = initialize_agent(
    [create_attached_suggestion, create_tile_suggestion, create_global_suggestion],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.invoke(
    {
        "input": """
You are tasked with optimizing a travel itinerary and creating suggestions to improve it. As you create suggestions, call the appropriate tool to add the suggestion to the list of suggestions.


Travel Itinerary:

Day 1: 2023-11-06
The start of day 1: Title: Exploring the Eternal City [ID: 01A]
Day 1 at 10:00: Flight from JFK to FCO booked. Next event is 30 km away and starts in 4 hours. [ID: 02A]
Day 1 at 14:00: Booked a Hotel at Hotel Roma in Rome for 2. Next event is 2 km away and starts in 5 hours. [ID: 03A]
Day 1 at 19:00: Dining for dinner at La Pergola in Rome. Described as: A Michelin-starred rooftop restaurant with panoramic views. [ID: 04A]
Day 2: 2023-11-07
The start of day 2: Title: The Heart of Ancient Rome [ID: 05A]
Day 2 at 10:00: Booked Colosseum Underground Tour in Rome for 2. Described as: An exclusive tour of the Colosseum, including its underground complex. Next event is 1 km away and starts in 3 hours. [ID: 06A]
Day 2 at 13:00: Dining for lunch at Roscioli in Rome. Described as: A historical bakery and gourmet deli near Campo de’ Fiori. Next event is 4 km away and starts in 2 hours. [ID: 07A]
Day 2 at 15:00: Booked Vespa rental for 2 from Roscioli to Villa Borghese. Next event is not listed. [ID: 08A]
Day 3: 2023-11-08
The start of day 3: Title: Vatican Wonders [ID: 09A]
Day 3 at 10:00: Booked Vatican Museums Skip-the-Line Ticket in Vatican City for 2. Described as: Skip-the-line access to the Vatican Museums and Sistine Chapel. Next event is 3 km away and starts in 3 hours. [ID: 10A]
Day 3 at 13:00: Dining for lunch at Pizzarium in Rome. Described as: Famous for its gourmet pizza by the slice. Next event is 25 km away and starts in 7 hours. [ID: 11A]
Day 3 at 20:00: Flight from FCO to JFK booked. [ID: 12A]

"""
    }
)

for sug in suggestions:
    print(f"- {sug}")
