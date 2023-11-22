import json
import time
from openai import OpenAI
from colorama import Fore


class Assistant:
    FUZZY_ID = "asst_ySH2fZjOyi94BDEfwdjAOjRT"

    def __init__(self, asst_id: str = FUZZY_ID):
        self.client = OpenAI()
        self.asst_id = asst_id

        self.thread = None
        self.run = None

    def call(self, first_message: str):
        self.thread = self.client.beta.threads.create()

        # Send initial message to thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=first_message,
        )

        # Begin running the assistant on the thread to generate a response
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.asst_id,
        )

        # Wait for the assistant to respond

        while run.status != "completed":
            time.sleep(0.25)

            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id,
            )

            # Check if the run failed
            if run.status in ["cancelling", "cancelled", "failed", "expired"]:
                print(f"Run failed with status: {run.status}")

                self.thread = None
                self.run = None
                return

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
