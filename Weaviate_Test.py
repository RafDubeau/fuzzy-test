import weaviate

import os

client = weaviate.Client(
    url="http://35.193.215.5:8080",  # Replace with your endpoint
    # auth_client_secret=weaviate.AuthApiKey(
    #     api_key="CUWrOUtRalQAQxiMfMv6y13TmulH7x0w4Xcd"
    # ),  # Replace w/ your Weaviate instance API key
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

print(client.schema.get())
