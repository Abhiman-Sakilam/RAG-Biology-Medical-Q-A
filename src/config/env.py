from pathlib import Path

from dotenv import load_dotenv

# Candidate .env locations, in the order they are consulted. The project root
# file wins over setup/.env; neither ever overwrites a variable that is already
# in the environment.
_ENV_FILENAMES = (".env", "setup/.env")

_loaded = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_env(force: bool = False) -> None:
    """Load .env files into os.environ without clobbering exported variables.

    Every entry point calls this at import time, so it is idempotent. Variables
    already present in the environment take precedence (`override=False`) --
    exporting GROQ_API_KEY in the shell still beats whatever is on disk.
    """
    global _loaded
    if _loaded and not force:
        return

    root = _project_root()
    for filename in _ENV_FILENAMES:
        path = root / filename
        if path.is_file():
            load_dotenv(path, override=False)

    _loaded = True

