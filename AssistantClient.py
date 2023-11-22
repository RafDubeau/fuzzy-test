import os
import sys

from colorama import Fore

# Add the 'local_dependencies' directory to sys.path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_script_directory, "local_dependencies")
if path not in sys.path:
    sys.path.append(path)

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Coroutine
from openai import OpenAI
from openai.types.beta.thread import Thread
import asyncio
from PythonFuzzyTypes import validateTile, Tile, validateTimelineHeader

openai_client = OpenAI()


class ToolResponse(TypedDict):
    tool_call_id: str
    output: str | Coroutine[None, Exception, str]


class AssistantClient:
    def __init__(self, assistant_id: str):
        self.assistant_id = assistant_id
        self.functions: Dict[str, Callable[..., str]] = {}

        self.curr_thread_id: Optional[str] = None

    def add_function(self, func: Callable[..., str]) -> None:
        self.functions[func.__name__] = func

    def new_thread(self) -> None:
        thread = openai_client.beta.threads.create()
        self.curr_thread_id = thread.id

    def set_thread(self, thread_id: str) -> None:
        self.curr_thread_id = thread_id

    def get_thread_id(self) -> str:
        return self.curr_thread_id

    async def __call__(self, message: str, thread_id: Optional[str] = None) -> str:
        print(f"Calling assistant {self.assistant_id} with message: {message}")

        if thread_id is not None:
            thread = openai_client.beta.threads.retrieve(thread_id=thread_id)
        elif self.curr_thread_id is not None:
            thread = openai_client.beta.threads.retrieve(thread_id=self.curr_thread_id)
        else:
            thread = openai_client.beta.threads.create()
            self.curr_thread_id = thread.id

        # Send message
        message = openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message,
        )

        # Begin running the assistant on the thread to generate a response
        run = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
        )

        # Wait for the assistant to respond
        while run.status != "completed":
            await asyncio.sleep(0.25)
            run = openai_client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

            if run.status in ["cancelled", "failed", "expired"]:
                raise ValueError(
                    f"Assistant {self.assistant_id} run failed with status {run.status}{f'and error {run.error}' if run.error else ''}"
                )

            if run.status == "requires_action":
                print(
                    f"Assistant {self.assistant_id} requires action. Calling functions..."
                )
                tool_calls = run.required_action.submit_tool_outputs.tool_calls

                tool_outputs: List[ToolResponse] = []

                for tool_call in tool_calls:
                    tool_output_obj = {
                        "tool_call_id": tool_call.id,
                    }

                    if tool_call.function.name in self.functions:
                        args = json.loads(tool_call.function.arguments)
                        print(f"Calling {tool_call.function.name} with args {args}...")

                        tool_output_obj["output"] = self.functions[
                            tool_call.function.name
                        ](**args)

                    else:
                        raise ValueError(
                            f"Assistant {self.assistant_id} requested unknown function {tool_call.function.name}"
                        )

                    tool_outputs.append(tool_output_obj)

                # Resolve coroutine outputs
                coroutine_idxs = [
                    i
                    for i in range(len(tool_outputs))
                    if asyncio.iscoroutine(tool_outputs[i]["output"])
                ]
                coroutine_results = await asyncio.gather(
                    *[tool_outputs[i]["output"] for i in coroutine_idxs]
                )
                for i in range(len(coroutine_idxs)):
                    tool_outputs[coroutine_idxs[i]]["output"] = coroutine_results[i]

                # Submit tool outputs
                run = openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

                print(
                    f"Tool outputs submitted. Waiting for assistant {self.assistant_id} to respond..."
                )

        messages = openai_client.beta.threads.messages.list(thread_id=thread.id)

        return messages.data[0].content[0].text.value


if __name__ == "__main__":
    WISHTRIP_PLANNER_ASST = "asst_gyx28dTn6CQVyGulUslvGNsZ"
    DATA_GENERATOR_ASST = "asst_wDqMuuBKhle0LotCisKK6f5w"

    data_generator = AssistantClient(DATA_GENERATOR_ASST)

    timeline: list[Tile] = []

    async def create_timeline_item(item_type: str, description: str) -> str:
        data_generator.new_thread()
        str_response = await data_generator(
            f"Create a sample {item_type}.json that fits the description: {description}"
        )
        try:
            timeline_item = json.loads(str_response)
        except:
            print(f"Failed to parse JSON response. Asking GPT to fix it...")
            str_response = await data_generator(
                f"Your response was not a valid JSON. Please fix it."
            )
            timeline_item = json.loads(str_response)

        try:
            timeline_item = validateTile(timeline_item)
        except:
            print(
                f"Failed to validate JSON response as type {item_type}. Asking GPT to fix it..."
            )
            str_response = await data_generator(
                f"Your response did not match the {item_type}.json schema. Please fix it."
            )
            timeline_item = json.loads(str_response)
            timeline_item = validateTile(timeline_item)

        timeline.append(timeline_item)

        return f"{item_type}: {description}"

    wishtrip_planner = AssistantClient(WISHTRIP_PLANNER_ASST)
    wishtrip_planner.add_function(create_timeline_item)

    result = asyncio.run(
        wishtrip_planner(f"Create a trip to London with approximately 10 tiles.")
    )

    print(result)

    with open("AssistantClient_output.json", "w") as f:
        json.dump(timeline, f, indent=4)
