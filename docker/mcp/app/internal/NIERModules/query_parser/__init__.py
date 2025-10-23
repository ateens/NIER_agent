def validate_query(query: dict) -> None:
    """Basic validation replicated from legacy pipeline."""
    if "type" not in query:
        raise ValueError("Missing required key: type")

    required_keys = ["type"]
    if query["type"] == "time_series":
        required_keys += ["region", "start_time", "end_time", "element"]
    elif query["type"] == "general":
        required_keys += ["question"]

    for key in required_keys:
        if key not in query:
            raise ValueError(f"Missing required key: {key}")

    if query["type"] == "time_series":
        if not isinstance(query["region"], int):
            raise ValueError("`region` must be an integer.")
        if not isinstance(query["start_time"], str) or not isinstance(query["end_time"], str):
            raise ValueError("`start_time` and `end_time` must be strings.")
        if not isinstance(query["element"], str):
            raise ValueError("`element` must be a string.")
    elif query["type"] == "general":
        if not isinstance(query.get("question"), str):
            raise ValueError("`question` must be a string.")
