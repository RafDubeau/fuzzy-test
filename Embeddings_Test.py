import os
import openai
import numpy as np

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

# Toy data
items = [
    [
        "safari",
        "lions",
        "giraffes",
        "Enjoy this beautiful 3-day ride across Tanzania!\nYour satisfaction is guaranteed",
    ],
    [
        "2-day snowboarding adventure",
        "winter sports",
        "snowboarding",
        "skiing",
        "mountain",
    ],
    ["culture", "luxury", "all expenses paid shopping trip to New York City"],
]

user_initial_interests = [["athletics", "nature"], ["art", "literature", "fashion"]]


# Preprocessing
items_text = [" ".join(item) for item in items]
users_text = [" ".join(user) for user in user_initial_interests]


# Generate Embeddings


def get_embedding(text: str | list, model="text-embedding-ada-002"):
    # Clean string(s)
    if isinstance(text, str):
        text = [text]
    text = [subtext.replace("\n", " ") for subtext in text]

    embeddings = openai.Embedding.create(input=text, model=model)["data"]

    return np.array([embedding["embedding"] for embedding in embeddings])


item_embeddings = get_embedding(items_text)
users_embeddings = get_embedding(users_text)

print(f"item embeddings shape: {item_embeddings.shape}")
print(f"user embeddings shape: {users_embeddings.shape}")

print(f"Example Embedding: {users_embeddings[0]}")


# Compute cosine similarity matrix
cos_similarity = users_embeddings @ item_embeddings.T

print()
print(cos_similarity.shape)
print(cos_similarity)

# Maximum score in each row gives the vacation which is closest to the user's interests
recommendations = np.argmax(cos_similarity, axis=1)

print()
print("Results:")
for i in range(cos_similarity.shape[0]):
    print(f"{user_initial_interests[i]} <---> {items[recommendations[i]]}")
