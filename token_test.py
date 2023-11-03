import tiktoken


if __name__ == "__main__":
    encoder = tiktoken.encoding_for_model("gpt-4")
    enc = encoder.encode("Hello, world!")
    print(len(enc))
    print(enc)

    print([encoder.decode([i]) for i in enc])

    dec = encoder.decode(enc)
    print(dec)