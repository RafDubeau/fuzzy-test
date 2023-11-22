from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Initialize LLM object
llm = ChatOpenAI(temperature=0.5, model="gpt-4")

result = llm.invoke(
    """
You are tasked with optimizing a travel itinerary and incorporating suggested changes/concerns graphically on a timeline. You have access to a predetermined list of components that can be used to represent these suggestions.


Travel Itinerary:

Destination: Paris, France
Departure Date: May 10, 2023
Return Date: May 20, 2023
Day 1: Arrival in Paris, check-in at Hotel A.
Day 2: Explore the Louvre Museum.
Day 3: Visit the Eiffel Tower.
Day 4: Day trip to Versailles Palace.
Day 5: Explore Montmartre and Sacré-Cœur Basilica.
Day 6: Take a Seine River cruise.
Day 7: Free day for shopping and leisure.
Day 8: Explore Notre-Dame Cathedral.
Day 9: Visit Musée d'Orsay.
Day 10: Departure from Paris.


List of Suggestions/Concerns:

- Extend the stay in Paris by two days.
- Include a day trip to the Palace of Fontainebleau.
- Add a visit to the Champs-Élysées and Arc de Triomphe.
- Consider a cooking class experience.
- Make reservations at a renowned French restaurant for one evening.

List of Components:

1. Timeline Extension: Extend the timeline to represent an extended stay.
2. Additional Day: Add an extra day to the itinerary.
3. Day Trip Icon: Use an icon to represent a day trip.
4. Landmark Marker: Mark significant landmarks on the timeline.
5. Restaurant Reservation: Use a reservation badge to indicate a restaurant booking.
6. Cooking Class: Include a chef's hat symbol for a cooking class experience.
7. Museum Ticket: Display a ticket icon for museum visits.
8. Shopping Bag: Use a shopping bag symbol for shopping and leisure days.
9. Hotel Symbol: Represent hotel stays with a hotel building icon.
10. Transportation: Use an arrow symbol to indicate transportation between locations.
11. Free Day Label: Label free days as "Free Day."
12. Cruise Ship: Use a cruise ship icon for a river cruise.
13. Historical Site: Mark historical sites with an ancient monument symbol.
14. Day Number: Display day numbers to indicate the sequence of days.
15. Tour Guide Icon: Use a tour guide symbol for guided tours.


Your task is to decide:
a. Which components should be used to represent each suggestion/concern?
b. Where on the timeline should these components be placed to effectively communicate the suggestions?

Your response should include a description of:
- Which components you choose for each suggestion.
- The specific positions on the timeline where you place these components.
"""
)

print(result.content)
