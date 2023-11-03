from __future__ import annotations
import inspect
from types import NoneType, UnionType
from typing import (
    Any,
    Callable,
    Dict,
    TypedDict,
    get_origin,
    get_args,
    Union,
    List,
    Tuple,
    Optional,
    Required,
    Set,
    get_type_hints,
)
import typing
from colorama import Fore


class GoodData(TypedDict):
    name: str
    age: int | float


class local_good_data(TypedDict):
    name: str
    age: int


def validator(data) -> GoodData:
    return data


def cool_function(data: local_good_data) -> int:
    return data["age"]


def cooler_function(name: str, age: int) -> int:
    return age


def is_typed_dict(x):
    return hasattr(x, "__annotations__") and any(
        issubclass(base, dict) for base in x.__bases__
    )


def is_subtype_of(subtype, supertype):
    print(
        f"'{subtype}'({get_origin(subtype)})  <:  '{supertype}'({get_origin(supertype)})"
    )

    if subtype is supertype:
        return True

    # Break down TypedDicts
    elif is_typed_dict(subtype) and is_typed_dict(supertype):
        subtype_hints = get_type_hints(subtype)
        supertype_hints = get_type_hints(supertype)
        return all(
            key in subtype_hints
            and is_subtype_of(subtype_hints[key], supertype_hints[key])
            for key in supertype_hints
        )
    elif (is_typed_dict(subtype) and not is_typed_dict(supertype)) or (
        not is_typed_dict(subtype) and is_typed_dict(supertype)
    ):
        return False

    # Break down subtype into primitive types
    elif get_origin(subtype) in (Union, UnionType):
        return all(is_subtype_of(arg, supertype) for arg in get_args(subtype))
    elif get_origin(subtype) is Required:
        return is_subtype_of(get_args(subtype)[0], supertype)

    # Break down supertype into primitive types
    elif get_origin(supertype) in (Union, UnionType):
        return any(is_subtype_of(subtype, arg) for arg in get_args(supertype))
    elif get_origin(supertype) in (
        List,
        Tuple,
        Dict,
        Set,
        list,
        tuple,
        dict,
        set,
    ) or get_origin(subtype) in (List, Tuple, Dict, Set, list, tuple, dict, set):
        return (
            get_origin(subtype) is not None
            and get_origin(supertype) is not None
            and issubclass(get_origin(subtype), get_origin(supertype))
            and all(
                is_subtype_of(sub_arg, super_arg)
                for sub_arg, super_arg in zip(get_args(subtype), get_args(supertype))
            )
        )
    elif get_origin(supertype) is Required:
        return is_subtype_of(subtype, get_args(supertype)[0])

    # Base Case
    else:
        return issubclass(subtype, supertype)


def should_unpack_validator(func: Callable[..., Any], validator: Callable[..., Any]):
    parameters = {k: v for k, v in get_type_hints(func).items() if k != "return"}
    print(f"parameters: {parameters}")
    arguments = get_type_hints(validator)["return"]
    print(f"arguments: {arguments}")
    arg_annotations = get_type_hints(arguments)
    print(f"arg_annotations: {arg_annotations}")

    if len(parameters) == 1:
        if is_subtype_of(arguments, list(parameters.values())[0]):
            return False
        elif len(arg_annotations) > 1:
            raise ValueError(
                f"Incompatible validator return type. Expected: {list(parameters.values())[0]}, got: {arguments}."
            )

    if (
        len(parameters) == 1
        and len(arg_annotations) == 1
        and is_subtype_of(
            list(arg_annotations.values())[0], list(parameters.values())[0]
        )
    ):
        return True

    if len(parameters) != len(arg_annotations):
        raise ValueError(
            f"Number of function parameters: {len(parameters)} does not match number of values in validator return type: {len(arg_annotations)}."
        )

    for key, value in parameters.items():
        if key not in arg_annotations:
            raise ValueError(
                f"Function parameter '{key}' not found in validator return type."
            )
        if not is_subtype_of(arg_annotations[key], value):
            raise ValueError(
                f"Type of function parameter '{key}': {value} does not match validator return type '{key}': {arg_annotations[key]}."
            )

    return True


def should_unpack_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]):
    parameters = {param.name for param in inspect.signature(func).parameters.values()}

    if len(parameters) == 1 and len(kwargs) > 1:
        return False

    matching_keys = parameters.intersection(kwargs.keys())
    if len(matching_keys) == len(parameters):
        return True
    else:
        raise ValueError(
            f"Number of function parameters: {len(parameters)} does not match number of values in kwargs: {len(kwargs)}."
        )


class E_compute_cf_scores(TypedDict, total=False):
    user_ids: "_EComputeCfScoresUserIds"
    """ Aggregation type: anyOf """

    video_ids: "_EComputeCfScoresVideoIds"
    """ Aggregation type: anyOf """


_EComputeCfScoresUserIds = Union[str, List[str]]
""" Aggregation type: anyOf """


_EComputeCfScoresVideoIds = Union[str, List[str]]
""" Aggregation type: anyOf """


def print_bool(b: bool):
    print(f"{Fore.GREEN if b else Fore.RED}{b}{Fore.RESET}")


if __name__ == "__main__":
    # print(should_unpack_validator(cool_function, validator))
    # print(should_unpack_validator(cooler_function, validator))

    # print(should_unpack_kwargs(cool_function, {"name": "John", "age": 42}))
    # print(should_unpack_kwargs(cooler_function, {"name": "John", "age": 42}))

    class A(TypedDict):
        name: "_OnionType"

    _OnionType = Union[str, List[str]]

    def validateA(data) -> A:
        return data

    def func(name: str | list[str]) -> int:
        if isinstance(name, str):
            name = [name]
        return sum(len(n) for n in name)

    print_bool(is_subtype_of(int | float, Optional[int | float | str]))
