"""MindPalace 配置管理。从环境变量读取，提供合理默认值。"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mindpalace.db"

# --- LLM ---
def _normalize_profile_name(value: str | None) -> str:
    """Normalize a user-facing profile name into an env-safe slug."""
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")


def _profile_env_key(profile_name: str | None, field: str) -> str:
    return f"API_PROFILE_{_normalize_profile_name(profile_name)}_{field}"


def _infer_provider_type(base_url: str | None, explicit: str | None = None) -> str:
    """Return the provider protocol adapter name."""
    if explicit:
        normalized = explicit.strip().lower()
        if normalized in {"anthropic", "claude"}:
            return "anthropic"
        return "openai"

    lowered = (base_url or "").lower()
    if "anthropic.com" in lowered:
        return "anthropic"
    return "openai"


def get_provider_config(prefix: str = "OPENAI") -> dict:
    """获取指定前缀的 Provider 配置，如果缺失则回退到 OPENAI 前缀。"""
    profile_name = os.getenv(f"{prefix}_PROVIDER_PROFILE")

    # 尝试特定的 API KEY
    api_key = os.getenv(f"{prefix}_API_KEY") or (
        os.getenv(_profile_env_key(profile_name, "API_KEY")) if profile_name else None
    )
    if not api_key and prefix != "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")

    # 尝试特定的 BASE URL
    base_url = os.getenv(f"{prefix}_BASE_URL") or (
        os.getenv(_profile_env_key(profile_name, "BASE_URL")) if profile_name else None
    )
    if not base_url and prefix != "OPENAI":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif not base_url:
        base_url = "https://api.openai.com/v1"

    provider_type = os.getenv(f"{prefix}_PROVIDER_TYPE") or (
        os.getenv(_profile_env_key(profile_name, "PROVIDER_TYPE")) if profile_name else None
    )
    if (
        not provider_type
        and prefix != "OPENAI"
        and not os.getenv(f"{prefix}_BASE_URL")
    ):
        provider_type = os.getenv("OPENAI_PROVIDER_TYPE")

    # 尝试特定的模型列表
    models_raw = (
        os.getenv(f"{prefix}_MODEL_NAMES")
        or os.getenv(f"{prefix}_MODEL_NAME")
        or (os.getenv(_profile_env_key(profile_name, "MODEL_NAMES")) if profile_name else None)
        or (os.getenv(_profile_env_key(profile_name, "MODEL_NAME")) if profile_name else None)
    )
    if not models_raw and prefix == "OPENAI":
        models_raw = (
            os.getenv("OPENAI_MODEL_NAMES")
            or os.getenv("MODEL_NAMES")
            or os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("MODEL_NAME")
        )
    if not models_raw and prefix != "OPENAI":
        models_raw = (
            os.getenv("OPENAI_MODEL_NAMES")
            or os.getenv("MODEL_NAMES")
            or os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("MODEL_NAME")
        )
    
    models = [m.strip() for m in (models_raw or "").split(",") if m.strip()]

    return {
        "api_key": api_key,
        "base_url": base_url,
        "models": models,
        "provider_type": _infer_provider_type(base_url, provider_type),
        "provider_profile": profile_name,
    }

# 默认主配置
PRIMARY_CONFIG = get_provider_config("OPENAI")

# 任务特定配置（运行时可调用 get_provider_config）
def get_scout_config(): return get_provider_config("SCOUT")
def get_council_config(): return get_provider_config("COUNCIL")
def get_memory_config(): return get_provider_config("MEMORY")
def get_fast_config(): return get_provider_config("FAST")


def get_embedding_config():
    """Embedding 档：优先 EMBEDDING_*，未配置模型时默认 text-embedding-3-small。"""
    cfg = get_provider_config("EMBEDDING")
    explicit_models = os.getenv("EMBEDDING_MODEL_NAMES") or os.getenv("EMBEDDING_MODEL_NAME")
    if explicit_models:
        return cfg

    cfg["models"] = ["text-embedding-3-small"]
    return cfg


def get_router_config():
    """Router 档：优先使用 ROUTER_*，其次回落到 FAST_*，最后 OPENAI 兜底。"""
    cfg = get_provider_config("ROUTER")
    if not cfg.get("models"):
        cfg = get_fast_config()
    return cfg


def get_judge_config():
    """Judge 档：优先 JUDGE_*，未配置时回落到 COUNCIL_*。"""
    cfg = get_provider_config("JUDGE")
    if not cfg.get("models"):
        cfg = get_council_config()
    return cfg


# --- Scout ---
DEFAULT_SCOUT_FEED_PRESET = "humanities"

FEED_PRESETS = {
    "humanities": [
        "https://aeon.co/feed.rss",
        "https://psyche.co/feed.rss",
        "https://daily.jstor.org/feed/",
        "https://thepointmag.com/feed/",
        "https://www.noemamag.com/feed/",
        "https://crookedtimber.org/feed/",
    ],
    "mixed": [
        "https://aeon.co/feed.rss",
        "https://psyche.co/feed.rss",
        "https://blogs.lse.ac.uk/impactofsocialsciences/feed/",
        "https://www.publicbooks.org/feed/",
        "https://hnrss.org/best",
        "http://arxiv.org/rss/cs.AI",
    ],
    "tech": [
        "https://hnrss.org/best",
        "http://arxiv.org/rss/cs.AI",
    ],
}


def _parse_feed_list(raw: str | None) -> list[str]:
    """解析环境变量中的 feed 列表，支持空格/逗号/换行分隔。"""
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[\s,]+", raw.strip()) if item.strip()]


def get_default_feeds(preset: str | None = None) -> list[str]:
    """返回默认 Scout RSS 源。"""
    if preset:
        selected = preset.strip().lower()
    else:
        env_feeds = _parse_feed_list(os.getenv("SCOUT_FEEDS") or os.getenv("FEEDS"))
        if env_feeds:
            return env_feeds
        selected = (os.getenv("SCOUT_FEED_PRESET") or DEFAULT_SCOUT_FEED_PRESET).strip().lower()

    return FEED_PRESETS.get(selected, FEED_PRESETS[DEFAULT_SCOUT_FEED_PRESET]).copy()


DEFAULT_FEEDS = get_default_feeds()
SCOUT_TOP_K = 5
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
ARTICLE_RETENTION_DAYS = int(os.getenv("ARTICLE_RETENTION_DAYS", "30"))
ARTICLE_AUTO_CLEANUP = os.getenv("ARTICLE_AUTO_CLEANUP", "true").lower() in {"1", "true", "yes", "on"}

# --- Council 辩论参数 ---
MAX_REBUTTAL_ROUNDS = int(os.getenv("MAX_REBUTTAL_ROUNDS", "3"))
CONVERGE_THRESHOLD = float(os.getenv("CONVERGE_THRESHOLD", "0.3"))
COUNCIL_CONVERGENCE_PROTOCOL = (os.getenv("COUNCIL_CONVERGENCE_PROTOCOL", "midcheck")).strip().lower()

# --- Memory 参数 ---
CRYSTAL_WINDOW = int(os.getenv("CRYSTAL_WINDOW", "10"))
