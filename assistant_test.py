import time
from openai import OpenAI
from colorama import Fore
import json

suggestions = []


def add_attached_suggestion(item_id: str, suggestion_text: str) -> str:
    suggestions.append(
        f"Suggestion attached to {Fore.BLUE}{item_id}{Fore.RESET}: {suggestion_text}"
    )
    return "Suggestion added successfully."


client = OpenAI()
FUZZY_ID = "asst_ySH2fZjOyi94BDEfwdjAOjRT"

sample_itinerary = """Day 1: 2023-11-06
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
Day 3 at 20:00: Flight from FCO to JFK booked. [ID: 12A]"""

thread_id = None
run_id = None
if run_id is None or thread_id is None:
    thread = client.beta.threads.create()

    # Send initial message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Please add suggestions to the following itinerary:\n\n{sample_itinerary}",
    )

    # Begin running the assistant on the thread to generate a response
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=FUZZY_ID,
    )

    thread_id = thread.id
    run_id = run.id

    print(f"Thread ID: {thread_id}")
    print(f"Run ID: {run_id}\n\n")
else:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )

# Wait for the assistant to respond
while run.status != "completed":
    time.sleep(0.25)

    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )

    # Check if the run failed
    if run.status in ["cancelling", "cancelled", "failed", "expired"]:
        print(f"Run failed with status: {run.status}")
        exit()

    # Check if action is required
    if run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        tool_outputs = []

        for tool_call in tool_calls:
            tool_output_obj = {
                "tool_call_id": tool_call.id,
                "output": "Unknown Tool",  # Default response if the tool is not found
            }

            if tool_call.function.name == "add_attached_suggestion":
                args = json.loads(tool_call.function.arguments)
                print(args)
                item_id = args["item_id"]
                suggestion_text = args["suggestion_text"]

                tool_output_obj["output"] = add_attached_suggestion(
                    item_id, suggestion_text
                )

            tool_outputs.append(tool_output_obj)

        # Submit tool response
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
        )
