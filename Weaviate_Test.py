import weaviate
import requests
import json

client = weaviate.Client(
    url="https://test-cluster-ljsb7fh8.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="CUWrOUtRalQAQxiMfMv6y13TmulH7x0w4Xcd"),  # Replace w/ your Weaviate instance API key
        additional_headers = {
        "X-OpenAI-Api-Key": "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"  # Replace with your inference API key
    }
)


# ============================= START ADD SCHEMA =============================
# class_obj = {
#     "class": "newItem",
#     "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
#     "moduleConfig": {
#         "text2vec-openai": {},
#         "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
#     }
# }

# client.schema.create_class(class_obj)
# ============================= END ADD SCHEMA =============================

# ============================= START IMPORT DATA =============================
# resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
# data = json.loads(resp.text)  # Load data

# client.batch.configure(batch_size=100)  # Configure batch
# with client.batch as batch:  # Initialize a batch process
#     for i, d in enumerate(data):  # Batch import data
#         print(f"importing question: {i+1}")
#         properties = {
#             "answer": d["Answer"],
#             "query": d["Question"],
#             "category": d["Category"],
#         }
#         batch.add_data_object(
#             uuid=d["Answer"],
#             data_object=properties,
#             class_name="Item"
#         )
# ========================== END IMPORT DATA =============================

# ============================= START QUERY DATA =============================

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_where({"path": "id", "operator": "Equal", "valueString": "028ae72b-f276-4196-aec9-3d5ac1f51cfe"})
    .do()
)

print(json.dumps(response, indent=4))

# ============================= END QUERY DATA =============================


class WeaviateAPI:

    def __init__(self, field_map, class_name, verbose=False):
        self.client = weaviate.Client(
            url="https://test-cluster-ljsb7fh8.weaviate.network",  # Replace with your endpoint
            auth_client_secret=weaviate.AuthApiKey(api_key="CUWrOUtRalQAQxiMfMv6y13TmulH7x0w4Xcd"),  # Replace w/ your Weaviate instance API key
            additional_headers = {
                "X-OpenAI-Api-Key": "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"  # Replace with your inference API key
            }
        )

        self.field_map = field_map
        self.class_name = class_name

        self.verbose = verbose

    
    def create_class(self, class_obj):
        self.client.schema.create_class(class_obj)

    def set_verbose(self, verbose):
        self.verbose = verbose
    

    def import_data(self, data, use_uuid=True, batch_size: int | None = 100):
        """
        Import data to the client in batches.

        Parameters:
        - data (list): The list of dictionaries containing data to be imported.
        - batch_size (int, optional): Size of the batch. Defaults to 100.

        """
        
        self._client.batch.configure(batch_size=batch_size)
        with self._client.batch as batch:
            for i, d in enumerate(data):
                if self.verbose:
                    print(f"importing item: {i+1}")
                properties = {key: d[value] for key, value in self.field_map.items()}
                
                # Here, assuming that UUID is based on the `id` field, but this can be customized as needed
                if use_uuid:
                    batch.add_data_object(
                        uuid=d[self.field_map["id"]],
                        data_object=properties,
                        class_name=self.class_name
                    )
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name
                )


    async def query_by_uuid(self, uuid: str, fields: list[str]):
        response = await (
            client.query
            .get(self.class_name, [self.field_map[field] for field in fields]).with_where({"path": "id", "operator": "Equal", "valueString": uuid})
            .do()
        )

        return response
    

    async def query_by_text(self, text: str | list[str], fields: list[str], limit: int = 10):
        if isinstance(text, str):
            text = [text]

        response = await (
            client.query
            .get(self.class_name, [self.field_map[field] for field in fields])
            .with_near_text({"concepts": text})
            .with_additional(["id"])
            .with_limit(limit)
            .do()
        )

        return response


