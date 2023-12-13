import asyncio
from itertools import chain
import json
from types import UnionType
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    TypedDict,
    Generic,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import os

import numpy as np
import weaviate
from weaviate.gql.get import GetBuilder

schema_types_map = {
    str: "text",
    int: "int",
    float: "number",
    bool: "boolean",
    dict: "object",
}


def is_typed_dict(x):
    return hasattr(x, "__annotations__") and any(
        issubclass(base, dict) for base in x.__bases__
    )


def python_type_to_wv_schema_type(python_type: Type):
    if get_origin(python_type) in (Union, UnionType):
        return list(
            chain.from_iterable(
                python_type_to_wv_schema_type(arg) for arg in get_args(python_type)
            )
        )
    elif get_origin(python_type) in (List, list):
        return [
            f"{t}[]" for t in python_type_to_wv_schema_type(get_args(python_type)[0])
        ]
    else:
        return [schema_types_map[python_type]]


def typed_dict_to_wv_schema_properties(
    typed_dict: Type[TypedDict], vectorized_properties: List[str] = []
):
    properties = []

    for name, t in get_type_hints(typed_dict).items():
        skip = (len(vectorized_properties) > 0) and (name not in vectorized_properties)
        if is_typed_dict(t):
            properties.append(
                {
                    "name": name,
                    "dataType": ["object"],
                    "nestedProperties": typed_dict_to_wv_schema_properties(t),
                    "moduleConfig": {
                        "text2vec-openai": {"skip": skip, "vectorizePropertyName": True}
                    },
                }
            )
        else:
            properties.append(
                {
                    "name": name,
                    "dataType": python_type_to_wv_schema_type(t),
                    "moduleConfig": {
                        "text2vec-openai": {"skip": skip, "vectorizePropertyName": True}
                    },
                }
            )

    return properties


def wv_auto_schema(
    type: Type[TypedDict], vectorized_properties: List[str] = []
) -> Dict:
    schema = {
        "class": type.__name__,
        "vectorize": "text2vec-openai",
        "properties": typed_dict_to_wv_schema_properties(type, vectorized_properties),
    }
    return schema


T = TypeVar("T", bound=TypedDict)


class WeaviateClient(Generic[T]):
    def __init__(
        self,
        weaviate_url: str,
        weaviate_api_key: str,
        openai_api_key: str,
        schema: Dict,
    ) -> None:
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.openai_api_key = openai_api_key

        self.wv_client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
            additional_headers={"X-OpenAI-Api-Key": openai_api_key},
        )

        # Auto-generate Weaviate schema if it isn't provided
        self.class_schema = schema if schema is not None else wv_auto_schema(T)
        self.class_name = self.class_schema["class"]

        if not self.wv_client.schema.exists(self.class_name):
            self.wv_client.schema.create_class(self.class_schema)

    def __get_async_wv_client(
        self,
        url: str,
        weaviate_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> Client:
        headers = {
            "Content-Type": "application/json",
        }
        if weaviate_api_key is not None:
            headers["Authorization"] = f"Bearer {weaviate_api_key}"
        if openai_api_key is not None:
            headers["X-Openai-Api-Key"] = openai_api_key

        transport = AIOHTTPTransport(url=f"{url}/v1/graphql", headers=headers)
        return Client(transport=transport, fetch_schema_from_transport=True)

    U = TypeVar("U")

    def __listify(self, u: U | Iterable[U]) -> list[U]:
        if isinstance(u, list):
            return u
        elif isinstance(u, Iterable) and not isinstance(u, str):
            return list(u)
        else:
            return [u]

    def __base_query(
        self,
        fields: Optional[str | list[str]] = None,
        additional: Optional[str | list[str]] = None,
        excluded_ids: Optional[str | list[str]] = None,
    ) -> GetBuilder:
        if fields is None:
            fields = [property["name"] for property in self.class_schema["properties"]]
        fields = self.__listify(fields)

        query = self.wv_client.query.get(self.class_name, fields)

        if additional is not None:
            additional = self.__listify(additional)
            query = query.with_additional(additional)

        if excluded_ids is not None:
            query = query.with_where(
                {
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["id"],
                            "operator": "NotEqual",
                            "valueString": excluded_id,
                        }
                        for excluded_id in self.__listify(excluded_ids)
                    ],
                }
            )

        return query

    def __execute_query(self, query: GetBuilder) -> dict:
        return query.do()

    async def __execute_query_async(self, query: GetBuilder) -> dict:
        async_client = self.__get_async_wv_client(
            url=self.weaviate_url,
            weaviate_api_key=self.weaviate_api_key,
            openai_api_key=self.openai_api_key,
        )

        query_str = query.build()
        query = gql(query_str)

        return await async_client.execute_async(query)

    def __build_query_by_uuid(
        self,
        uuid: str | Iterable[str],
        fields: Optional[str | Iterable[str]] = None,
        additional: Optional[str | Iterable[str]] = None,
    ) -> GetBuilder:
        query = self.__base_query(fields=fields, additional=additional)
        uuid = self.__listify(uuid)

        if len(uuid) == 1:
            query = query.with_where(
                {
                    "path": ["id"],
                    "operator": "Equal",
                    "valueString": uuid,
                }
            )
        else:
            query = query.with_where(
                {
                    "operator": "Or",
                    "operands": [
                        {
                            "path": ["id"],
                            "operator": "Equal",
                            "valueString": id,
                        }
                        for id in uuid
                    ],
                }
            )

        return query

    def __build_query_by_vector(
        self,
        vector: np.ndarray | list[float],
        limit: int = 10,
        excluded_ids: Optional[str | list[str]] = None,
        fields: Optional[str | list[str]] = None,
        additional: Optional[str | list[str]] = None,
    ) -> GetBuilder:
        query = self.__base_query(
            fields=fields, additional=additional, excluded_ids=excluded_ids
        )
        if not isinstance(vector, list):
            vector = vector.tolist()

        query = query.with_near_vector({"vector": vector}).with_limit(limit)

        return query

    def __build_query_by_text(
        self,
        text: str | list[str],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> GetBuilder:
        query = self.__base_query(
            fields=fields, additional=additional, excluded_ids=excluded_ids
        )

        text = self.__listify(text)

        query = query.with_near_text({"concepts": text}).with_limit(limit)

        return query

    def query_by_uuid(
        self,
        uuid: str | list[str],
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_uuid(
            uuid=uuid,
            fields=fields,
            additional=additional,
        )

        return self.__execute_query(query)

    async def query_by_uuid_async(
        self,
        uuid: str | list[str],
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_uuid(
            uuid=uuid,
            fields=fields,
            additional=additional,
        )

        return await self.__execute_query_async(query)

    def query_by_vector(
        self,
        vector: np.ndarray | list[float],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_vector(
            vector=vector,
            limit=limit,
            excluded_ids=excluded_ids,
            fields=fields,
            additional=additional,
        )

        return self.__execute_query(query)

    async def query_by_vector_async(
        self,
        vector: np.ndarray | list[float],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_vector(
            vector=vector,
            limit=limit,
            excluded_ids=excluded_ids,
            fields=fields,
            additional=additional,
        )

        return await self.__execute_query_async(query)

    def query_by_text(
        self,
        text: str | list[str],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_text(
            text=text,
            limit=limit,
            excluded_ids=excluded_ids,
            fields=fields,
            additional=additional,
        )

        return self.__execute_query(query)

    async def query_by_text_async(
        self,
        text: str | list[str],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> Dict:
        query = self.__build_query_by_text(
            text=text,
            limit=limit,
            excluded_ids=excluded_ids,
            fields=fields,
            additional=additional,
        )

        return await self.__execute_query_async(query)

    def import_data(
        self,
        data: List[T],
        uuids: list[str] | None = None,
        batch_size: int | None = 100,
    ):
        """
        Import data to the client in batches.

        Parameters:
        - data (list): The list of dictionaries containing data to be imported.
        - uuids (list[str], optional): The list of UUIDs to be used for each data item. Defaults to None.
        - batch_size (int, optional): Size of the batch. Defaults to 100.

        """

        self.wv_client.batch.configure(batch_size=batch_size)
        with self.wv_client.batch as batch:
            for i, d in enumerate(data):
                if uuids is not None:
                    batch.add_data_object(
                        uuid=uuids[i],
                        data_object=d,
                        class_name=self.class_name,
                    )
                else:
                    batch.add_data_object(data_object=d, class_name=self.class_name)

    def delete_data(self, ids: str | list[str]):
        if isinstance(ids, str):
            ids = [ids]

        assert len(ids) > 0, "At least one ID must be provided."

        for id in ids:
            self.wv_client.data_object.delete(
                uuid=id,
                class_name=self.class_name,
            )

    def get_id_list(self, limit: int = 100):
        query = self.__base_query(fields=[], additional=["id"]).with_limit(limit)
        response = self.__execute_query(query)

        id_list = [
            item["_additional"]["id"]
            for item in response["data"]["Get"][self.class_name]
        ]

        return id_list

    def get_embeddings(self, id_list: str | list[str]) -> dict[str, list[float]]:
        if isinstance(id_list, str):
            id_list = [id_list]

        embeddings: dict[str, list[float]] = {}
        response = self.query_by_uuid(id_list, fields=[], additional=["id", "vector"])
        for data in response["data"]["Get"][self.class_name]:
            embeddings[data["_additional"]["id"]] = data["_additional"]["vector"]

        return embeddings

    def update_embedding(self, id: str, embedding: list[float]) -> None:
        self.wv_client.data_object.update(
            data_object={}, uuid=id, class_name=self.class_name, vector=embedding
        )


experience_video_class_obj = {
    "class": "ExperienceVideo",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "sourceVideoDescription", "dataType": ["text"]},
        {"name": "videoTranscript", "dataType": ["text"]},
        {"name": "productTitle", "dataType": ["text"]},
        {"name": "productDescription", "dataType": ["text"]},
        {"name": "productLocation", "dataType": ["text"]},
        {"name": "productLanguage", "dataType": ["text"]},
        {"name": "productItineraryItems", "dataType": ["text[]"]},
        {"name": "productHighlights", "dataType": ["text[]"]},
        {"name": "videoIntelligenceDescription", "dataType": ["text"]},
        {"name": "videoIntelligenceObjects", "dataType": ["text[]"]},
        {"name": "videoTextOnScreen", "dataType": ["text[]"]},
    ],
}


async def main() -> None:
    client = WeaviateClient(
        weaviate_url="https://zkjd2oxtqbqsklhm48raww.c0.us-central1.gcp.weaviate.cloud",
        weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        schema=experience_video_class_obj,
    )

    results = await asyncio.gather(
        client.query_by_uuid_async(
            uuid=[
                "067ea483-f0aa-498f-a680-53b87d76178a",
                "39839692-ed06-48c9-8bc9-f9d4dd5f7bb3",
            ],
            fields=["productTitle", "productLocation"],
            additional=["id"],
        ),
        client.query_by_vector_async(
            vector=np.random.rand(1536),
            limit=3,
            fields=["productTitle", "productLocation"],
            additional=["id"],
        ),
        client.query_by_text_async(
            text="beach",
            limit=3,
            fields=["productTitle", "productLocation"],
            additional=["id"],
        ),
    )

    for result in results:
        print("-" * 100)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
