import time
from colorama import Fore

method = "get"
endpoint = "video_groups"

print(f"\r{Fore.BLUE}{method.upper()}{Fore.RESET} {endpoint} ", end="")
time.sleep(1)
print(
    f"\r{Fore.BLUE}{method.upper()}{Fore.RESET} {endpoint} - {Fore.GREEN}200{Fore.RESET}"
)
