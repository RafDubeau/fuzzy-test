import json
from AssistantUtils import AssistantClient
import asyncio
from colorama import Fore


class App:
    def __init__(self):
        self.state = {
            "rCurrentScreen": "home screen",
            "rFocusName": False,
            "rFocusPassword": False,
            "rIsThinking": False,
            "rLoggedIn": False,
            "rName": "",
            "rPassword": "",
            "rStartAnimation": False,
        }

    def invalid_action(self, screen: str, action: str) -> str:
        return f"Invalid action {action} for screen {screen}"

    def execute_action(self, screen: str, action: str) -> str:
        if screen == "home screen":
            if action == "login screen":
                self.state["rCurrentScreen"] = "login screen"
            elif action == "settings screen":
                self.state["rCurrentScreen"] = "settings screen"
            else:
                return self.invalid_action(screen, action)
        elif screen == "login screen":
            if action == "Go back":
                self.state["rCurrentScreen"] = "home screen"
            elif action == "Login" and not self.state["rLoggedIn"]:
                self.state["rLoggedIn"] = True
            elif action == "Logout" and self.state["rLoggedIn"]:
                self.state["rLoggedIn"] = False
            else:
                return self.invalid_action(screen, action)
        elif screen == "settings screen":
            if action == "Go back":
                self.state["rCurrentScreen"] = "home screen"
                self.state["rFocusName"] = False
                self.state["rFocusPassword"] = False
            elif action == "Change username" and self.state["rLoggedIn"]:
                self.state["rFocusName"] = True
            elif action == "Change password" and self.state["rLoggedIn"]:
                self.state["rFocusPassword"] = True
            else:
                return self.invalid_action(screen, action)
        else:
            return self.invalid_action(screen, action)

        return json.dumps(self.state)


if __name__ == "__main__":
    app = App()

    fuzzy = AssistantClient("asst_wBJVX5tBztDMg1mGEssYLyow")

    def execute_action(screen: str, action: str) -> str:
        print(app.state)
        response = app.execute_action(screen, action)
        print(app.state)
        print(f"{Fore.GREEN if response[0] == '{' else Fore.RED}{response}{Fore.RESET}")
        return response

    fuzzy.add_function(execute_action)

    print(asyncio.run(fuzzy(f"I want to change my username\n {json.dumps(app.state)}")))
