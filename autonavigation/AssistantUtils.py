import asyncio
import json
from typing import Coroutine, TypedDict, Dict, List, Optional, Callable, Iterable
from openai import OpenAI
from colorama import Fore


class ToolResponse(TypedDict):
    tool_call_id: str
    output: str | Coroutine[None, Exception, str]


class Assistant(TypedDict):
    id: str
    name: str
    instructions: str
    model: str
    functions: Dict[str, Dict]
    code_interpreter: bool
    retrieval: bool


class AssistantClient:
    """
    Client for interacting with OpenAI's assistant API.

    Each client manages a single thread with a single assistant. To use multiple assistants or threads concurrently, create multiple clients.
    Alternatively, you can use the `get_thread` and `set_thread` methods to manage multiple threads on a single client instance.

    """

    def __init__(self, assistant: str | Assistant) -> None:
        if isinstance(assistant, str):
            self.assistant_id = assistant
        else:
            self.assistant_id = assistant["id"]
        self.functions: Dict[str, Callable[..., str]] = {}
        self.curr_thread_id: Optional[str] = None

        self.openai_client = OpenAI()

    def add_function(self, func: Callable[..., str]) -> None:
        self.functions[func.__name__] = func

    def new_thread(self) -> None:
        thread = self.openai_client.beta.threads.create()
        self.curr_thread_id = thread.id

    def set_thread(self, thread_id: str) -> None:
        self.curr_thread_id = thread_id

    def get_thread(self) -> str:
        return self.curr_thread_id

    async def __call__(self, message: str) -> str:
        print(f"Calling assistant {self.assistant_id} with message: {message}")

        if self.curr_thread_id is not None:
            thread = self.openai_client.beta.threads.retrieve(
                thread_id=self.curr_thread_id
            )
        else:
            thread = self.openai_client.beta.threads.create()
            self.curr_thread_id = thread.id

        # Send message
        message = self.openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message,
        )

        # Begin running the assistant on the thread to generate a response
        run = self.openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
        )

        # Wait for the assistant to respond
        while run.status != "completed":
            await asyncio.sleep(0.25)
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

            if run.status in ["cancelled", "failed", "expired"]:
                raise Exception(
                    f"Assistant {self.assistant_id} run failed with status {run.status}{f'and error {run.last_error}' if run.last_error is not None else ''}"
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
                        print(
                            f"Calling {tool_call.function.name} with args {Fore.BLUE}{args}{Fore.RESET}..."
                        )

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
                run = self.openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

                print(
                    f"Tool outputs submitted. Waiting for assistant {self.assistant_id} to respond..."
                )

        messages = self.openai_client.beta.threads.messages.list(thread_id=thread.id)

        return messages.data[0].content[0].text.value
