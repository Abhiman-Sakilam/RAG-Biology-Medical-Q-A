from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / "setup" / ".env")
