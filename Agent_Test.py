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
