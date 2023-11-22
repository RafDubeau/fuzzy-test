from typing import TypedDict, get_args, get_type_hints, Optional, Union, ForwardRef


def is_typed_dict(x):
    return hasattr(x, "__annotations__") and any(
        issubclass(base, dict) for base in x.__bases__
    )


def stringify_typed_dict_type(t):
    return f"class {t.__name__}(TypedDict):\n\t" + "\n\t".join(
        f"{var}: {annotation}" for var, annotation in get_type_hints(t).items()
    )


if __name__ == "__main__":
    Data = Union["GoodData", "BadData"]

    class GoodData(TypedDict, total=False):
        name: str
        age: Optional[int | float]

    class BadData(TypedDict):
        name: str
        age: int

    class tmp(TypedDict):
        data: Data

    print(get_type_hints(tmp)["data"])

    # print(
    #     "\n\n".join(
    #         stringify_typed_dict_type(t) for t in get_args(Data) if is_typed_dict(t)
    #     )
    # )
