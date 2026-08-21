import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    import tomli as tomllib


def project_root() -> Path:
    env = os.environ.get("TG_ROOT")
    if env:
        return Path(env).resolve()
    try:
        here = Path.cwd().resolve()
    except OSError:
        return Path(__file__).resolve().parents[1]
    for p in [here, *here.parents]:
        if (p / "config.toml").exists() and (p / "archive").exists():
            return p
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> dict:
    cfg_path = root / "config.toml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    cfg.setdefault("general", {})
    cfg.setdefault("aliases", {})
    cfg.setdefault("accounts", {"default": "telegram.session"})
    return cfg
