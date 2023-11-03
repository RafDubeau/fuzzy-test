from typing import TypedDict

class Hi(TypedDict):
    hello: str
    world: int

class Goodbye(TypedDict):
    goodbye: str
    world: int

class HelloWorld(Hi):
    good: bool

def generate_class_code2(cls):
    lines = []
    lines.append(f"class {cls.__name__}({cls.__bases__[0].__name__}):")
    for key, value in cls.__annotations__.items():
        lines.append(f"    {key}: {value.__name__}")
    return '\n'.join(lines)

def generate_class_code(cls):
    return '\n'.join([f"class {cls.__name__}{'(' if len(cls.__bases__) > 0 else ''}{', '.join([base.__name__ for base in cls.__bases__])}{')' if len(cls.__bases__) > 0 else ''}:"] + [f"    {key}: {value.__name__}" for key, value in cls.__annotations__.items()])


if __name__ == "__main__":
    print(generate_class_code(Goodbye))


