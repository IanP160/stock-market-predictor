import os

def get_env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val

NEWSAPI_KEY = get_env("NEWSAPI_KEY")
