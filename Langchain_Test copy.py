from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import SimpleMemory
from langchain.chains import LLMChain, SequentialChain, TransformChain
import re


initial_itinerary = """Day 1: Embarkation Day
Morning
- 8:00 AM - 11:00 AM: Arrive at the port, check-in, and board the ship.
- 11:00 AM - 12:30 PM: Explore the ship or relax in your cabin.
Afternoon
- 1:00 PM - 2:00 PM: Welcome buffet lunch.
- 2:00 PM - 3:30 PM: Mandatory safety drill.
- 3:30 PM - 5:00 PM: Option A: Kids and teens orientation at the kids' club and teen lounge. Option B: Spa tour and relaxation.
Evening
- 7:00 PM - 9:00 PM: Welcome dinner.
- 9:30 PM - 11:00 PM: Option A: Opening night show at the ship's theater. Option B: Casino night.
Day 2: At Sea
Morning
- 8:00 AM - 10:00 AM: Breakfast buffet.
- 10:00 AM - 11:30 AM: Option A: Yoga and fitness classes. Option B: Art gallery tour.
Afternoon
- 12:00 PM - 1:30 PM: Lunch buffet.
- 2:00 PM - 4:00 PM: Option A: Pool games and contests. Option B: Wine tasting.
Evening
- 7:00 PM - 9:00 PM: Themed dinner (e.g., Caribbean night).
- 9:30 PM - 11:00 PM: Option A: Broadway-style show. Option B: Dance party on the main deck.
Day 3: Port of Call - Tropical Island
Morning
- 8:00 AM - 9:00 AM: Breakfast onboard.
- 9:30 AM - 12:30 PM: Option A: Shore excursions (e.g., snorkeling). Option B: Island tour.
Afternoon
- 1:00 PM - 3:00 PM: Option A: Beach relaxation. Option B: Local shopping and exploration.
Evening
- 7:00 PM - 9:00 PM: Dinner onboard.
- 9:30 PM - 11:00 PM: Option A: Comedy night at the ship's theater. Option B: Live music at the ship's lounge.
Day 4: At Sea
Morning
- 8:00 AM - 10:00 AM: Breakfast buffet.
- 10:30 AM - 12:00 PM: Option A: Cooking class. Option B: Book club and relaxation.
Afternoon
- 12:30 PM - 2:00 PM: Lunch buffet.
- 2:30 PM - 4:30 PM: Option A: Bingo and trivia games. Option B: Spa treatments.
Evening
- 7:00 PM - 9:00 PM: Captain's gala dinner.
- 9:30 PM - 11:00 PM: Option A: Movie night under the stars. Option B: Karaoke night.
Day 5: Disembarkation Day
Morning
- 7:00 AM - 8:30 AM: Breakfast buffet.
- 9:00 AM - 11:00 AM: Disembarkation process.
"""

user_data = {
    "user_likes": "gambling, sports, nature",
    "user_dislikes": "theater, cooking, music",
    "accessibility": "wears glasses"
}


# Initialize LLM object
llm = ChatOpenAI(temperature=0.9)

# Create first chain
suggestion_sys_message = SystemMessage(content="""
Assume the role of an expert itinerary planner.
Based on the travel details and preferences, find and list any potential problems in the itinerary and pick the best options from the optional activities.
                                       
Pay very close attention the user's preferences, and accessibility needs.
""")

suggestion_human_template = HumanMessagePromptTemplate.from_template(
"""
Here is some context about the user's preferences:
Likes: {user_likes}
Dislikes: {user_dislikes}
Accessibility: {accessibility}

Here is the itinerary:
{initial_itinerary}
"""
)

suggestion_template = ChatPromptTemplate.from_messages([suggestion_sys_message, suggestion_human_template])


suggestion_chain = LLMChain(
    llm=llm,
    prompt=suggestion_template,
    output_key="suggestion",
    verbose=True
)


# response = suggestion_chain(initial_itinerary)
# print(response['suggestion'])

# Create second chain
itinerary_sys_message = SystemMessage(content="""
Assume the role of an expert itinerary planner.
Based on the given suggestions, update the itinerary by adding or removing activities.
For each time slot with multiple options, select the suggested choice.
                                      
At the end of your response, show the new revised itinerary along with a list of all the changes you made to the itinerary. Use the format:
                                      
Revised Itinerary: 
REVISED ITINERARY GOES HERE
                                      
Changes:
LIST OF CHANGES GOES HERE
""")

itinerary_human_message = HumanMessagePromptTemplate.from_template(
"""
Here is the itinerary:
{initial_itinerary}

Here are the suggestions:
{suggestion}
"""
)

itinerary_template = ChatPromptTemplate.from_messages([itinerary_sys_message, itinerary_human_message])


itinerary_chain = LLMChain(
    llm=llm,
    prompt=itinerary_template,
    output_key="final_itinerary",
    verbose=True
)


# Transformer Chain to parse output
def regex_parsing(input: dict):
    text = input["final_itinerary"]
    regex_str = r"Revised Itinerary:([\S\s]+)Changes:([\S\s]+)"
    m = re.match(regex_str, text)
    return {
        "itinerary": m.group(1).strip(),
        "changes": m.group(2).strip()
    }

transform_chain = TransformChain(
    input_variables=["final_itinerary"],
    output_variables=["itinerary", "changes"],
    transform=regex_parsing,
    verbose=True
)

from langchain.tools import Tool

tools = [
    Tool.from_function(
        func=suggestion_chain.run
    )
]


# Link the chains sequentially
full_chain = SequentialChain(
    memory=SimpleMemory(memories=user_data),
    input_variables=["initial_itinerary"],
    chains=[suggestion_chain, itinerary_chain, transform_chain],
    output_variables=["itinerary", "changes"],
    verbose=True
)

response = full_chain(initial_itinerary)

print("----------------------------------------")
print(f"Finalized Itinerary:\n{response['itinerary']}")
print("----------------------------------------")
print(f"Change List:\n{response['changes']}")