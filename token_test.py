import tiktoken


if __name__ == "__main__":
    encoder = tiktoken.encoding_for_model("gpt-4")

    with open("tile_types.txt", "r") as f:
        tile_types = f.read()

    # print(type(tile_types))
    # print(len(tile_types))

    # print(tile_types[:100])
    enc = encoder.encode(tile_types)
    print(len(enc))
