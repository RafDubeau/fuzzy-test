from test_utils import post


user_id = "kPtLzCNeMLNvSLutIzehnQIl7mf1"  # Kyle's Id
wishtrip_id = "7c691c25-659c-4c5f-b417-ab7feec9b4c8"


def generate_wishtrip():
    post("clear_wishtrip_items", wishtrip_id=wishtrip_id)

    post(
        "add_sample_wishtrip",
        user_id=user_id,
        wishtrip_id=wishtrip_id,
        location="London",
        n_tiles=20,
    )


def suggestions():
    post("clear_wishtrip_suggestions", wishtrip_id=wishtrip_id)

    post(
        "add_wishtrip_suggestions",
        wishtrip_id=wishtrip_id,
    )


if __name__ == "__main__":
    # Clears any existing wishtrip and generates a new sample wishtrip
    generate_wishtrip()

    # Clears any existing suggestions and generates new suggestions
    suggestions()
