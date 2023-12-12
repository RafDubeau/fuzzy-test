import json
from types import UnionType
from typing import List, Union, get_args, get_type_hints, get_origin, TypedDict, Type
from itertools import chain


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


def auto_schema(type: Type[TypedDict], vectorized_properties: List[str] = []):
    schema = {
        "class": type.__name__,
        "vectorizer": "text2vec-openai",
        "properties": typed_dict_to_wv_schema_properties(type, vectorized_properties),
    }
    return schema


class Person(TypedDict):
    last_name: str
    address: "Address"


class Address(TypedDict):
    street: str
    city: str


print(
    json.dumps(
        auto_schema(
            Person,
            vectorized_properties=["address"],
        ),
        indent=2,
    )
)
