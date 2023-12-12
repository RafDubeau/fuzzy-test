import asyncio
import json
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from typing import Dict, Iterable, List, Optional, TypeVar, TypedDict, Generic
import os

import numpy as np


class QueryArgs(TypedDict):
    className: str
    properties: List[str]
    additional: List[str]
    where: List[str]
    limit: Optional[int]
    nearVector: Optional[str]
    nearText: Optional[str]


T = TypeVar("T", bound=TypedDict)
U = TypeVar("U")


class WeaviateAsyncClient(Generic[T]):
    def __init__(
        self,
        weaviate_url: str,
        weaviate_api_key: str,
        openai_api_key: str,
        schema: Dict,
        verbose=False,
    ) -> None:
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.openai_api_key = openai_api_key

        self.class_schema = schema
        self.class_name = schema["class"]

        self.verbose = verbose

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

        transport = AIOHTTPTransport(url=url, headers=headers)
        return Client(transport=transport, fetch_schema_from_transport=True)

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
    ) -> QueryArgs:
        query: QueryArgs = {
            "className": self.class_name,
            "properties": [],
            "additional": [],
            "limit": None,
            "where": [],
            "nearVector": None,
            "nearText": None,
        }

        if fields is None:
            fields = [property["name"] for property in self.class_schema["properties"]]
        query["properties"] = self.__listify(fields)

        if additional is not None:
            query["additional"] = self.__listify(additional)

        if excluded_ids is not None:
            query["where"] = [
                '{{ path: ["id"], operator: NotEqual, valueText: "{excluded_id}" }}'.format(
                    excluded_id=excluded_id
                )
                for excluded_id in self.__listify(excluded_ids)
            ]

        return query

    def __query_to_gql_str(self, query: QueryArgs) -> str:
        query_template = """
        {{
            Get {{
                {className}{operators} {{
                    {properties}
                }}
            }}
        }}
        """

        operators = []
        if len(query["where"]) > 0:
            if len(query["where"]) == 1:
                operators.append(f'where: {query["where"][0]}')
            else:
                operators.append(
                    'where: {{operator: "And", operands: [{conditions}]}}'.format(
                        conditions=", ".join(query["where"])
                    )
                )
        if query["limit"] is not None:
            operators.append(f"limit: {query['limit']}")
        if query["nearVector"] is not None:
            operators.append("nearVector: {vector}".format(vector=query["nearVector"]))
        if query["nearText"] is not None:
            operators.append("nearText: {text}".format(text=query["nearText"]))

        if len(operators) > 0:
            operators_str = f'({", ".join(operators)})'
        else:
            operators_str = ""

        if len(query["additional"]) > 0:
            properties_str = ", ".join(
                query["properties"]
                + [f"_additional{{ {', '.join(query['additional'])} }}"]
            )
        else:
            properties_str = ", ".join(query["properties"])

        query_str = query_template.format(
            operators=operators_str,
            className=query["className"],
            properties=properties_str,
        ).replace("'", '"')

        print(query_str)

        return query_str

    def __execute_query(self, query: QueryArgs) -> dict:
        client = self.__get_async_wv_client(
            url=self.weaviate_url,
            weaviate_api_key=self.weaviate_api_key,
            openai_api_key=self.openai_api_key,
        )

        query_str = self.__query_to_gql_str(query)
        query = gql(query_str)

        return client.execute(query)

    async def __execute_query_async(self, query: QueryArgs) -> dict:
        client = self.__get_async_wv_client(
            url=self.weaviate_url,
            weaviate_api_key=self.weaviate_api_key,
            openai_api_key=self.openai_api_key,
        )

        query_str = self.__query_to_gql_str(query)
        query = gql(query_str)

        return await client.execute_async(query)

    async def query_by_uuid(
        self,
        uuid: str | Iterable[str],
        fields: Optional[str | Iterable[str]] = None,
        additional: Optional[str | Iterable[str]] = None,
    ) -> dict:
        query = self.__base_query(fields=fields, additional=additional)
        uuid = self.__listify(uuid)

        id_query_format = '{{ path: ["id"], operator: Equal, valueText: "{uuid}" }}'
        if len(uuid) == 1:
            query["where"].append(id_query_format.format(uuid=uuid[0]))
        else:
            query["where"].append(
                "{{ operator: Or, operands: [{conditions}]}}".format(
                    conditions=", ".join(
                        id_query_format.format(uuid=_uuid) for _uuid in uuid
                    )
                )
            )

        return await self.__execute_query_async(query)

    async def query_by_vector(
        self,
        vector: np.ndarray | list[float],
        limit: int = 10,
        excluded_ids: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> dict:
        query = self.__base_query(
            fields=fields, additional=additional, excluded_ids=excluded_ids
        )
        if not isinstance(vector, list):
            vector = vector.tolist()

        query["nearVector"] = "{{ vector: {vector} }}".format(vector=vector)
        query["limit"] = limit

        return await self.__execute_query_async(query)

    async def query_by_text(
        self,
        text: str | list[str],
        limit: int = 10,
        id_list: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        additional: Optional[list[str]] = None,
    ) -> dict:
        query = self.__base_query(
            fields=fields, additional=additional, excluded_ids=id_list
        )

        text = self.__listify(text)

        query["nearText"] = "{{ concepts: {text} }}".format(text=text)
        query["limit"] = limit

        return await self.__execute_query_async(query)


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
    client = WeaviateAsyncClient(
        weaviate_url="https://zkjd2oxtqbqsklhm48raww.c0.us-central1.gcp.weaviate.cloud/v1/graphql",
        weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        schema=experience_video_class_obj,
    )

    results = await asyncio.gather(
        client.query_by_uuid(
            uuid=[
                "067ea483-f0aa-498f-a680-53b87d76178a",
                "39839692-ed06-48c9-8bc9-f9d4dd5f7bb3",
            ],
            fields=["productTitle", "productLocation"],
            additional=["id"],
        ),
        client.query_by_vector(
            vector=np.random.rand(1536),
            limit=3,
            fields=["productTitle", "productLocation"],
            additional=["id"],
        ),
        client.query_by_text(
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
