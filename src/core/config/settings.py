from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def load_env(
        env_file: str | Path = ".env",
        *,
        override: bool = False,
) -> Path:
    """
    Load environment variables from a .env-style file.

    Relative paths are resolved from the project root
    (directory containing `src/`).

    You can explicitly override the path by setting:

        ANALYTICS_AIRFLOW_ENV_FILE=/full/path/to/.env
    """
    global _ENV_LOADED

    # -------------------------------------------------
    # 1) Explicit override (highest priority)
    # -------------------------------------------------
    explicit_path = os.getenv("ANALYTICS_AIRFLOW_ENV_FILE")
    if explicit_path:
        env_path = Path(explicit_path).expanduser().resolve()
        if not env_path.exists():
            raise FileNotFoundError(
                f"ANALYTICS_AIRFLOW_ENV_FILE is set but file does not exist:\n"
                f"  {env_path}"
            )
    else:
        # -------------------------------------------------
        # 2) Original project-root discovery logic
        # -------------------------------------------------
        env_path = Path(env_file)

        searched_locations = []

        if not env_path.is_absolute():
            cwd = Path.cwd().resolve()

            for parent in [cwd, *cwd.parents]:
                searched_locations.append(str(parent / env_file))
                if (parent / "src").is_dir():
                    env_path = parent / env_file
                    break

        env_path = env_path.resolve()

        if not env_path.exists():
            searched_str = "\n  - ".join(searched_locations) or str(env_path)
            raise FileNotFoundError(
                f"Env file not found: {env_path}\n\n"
                f"Searched locations:\n  - {searched_str}\n\n"
                f"If running inside Airflow or a non-standard environment,\n"
                f"you can explicitly set the path via:\n\n"
                f"  export ANALYTICS_AIRFLOW_ENV_FILE=/absolute/path/to/.env\n"
            )

    # -------------------------------------------------
    # 3) Load file
    # -------------------------------------------------
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")

        if override or key not in os.environ:
            os.environ[key] = val

    _ENV_LOADED = True
    return env_path


def env_loaded() -> bool:
    return _ENV_LOADED


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)
