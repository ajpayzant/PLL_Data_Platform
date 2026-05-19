# ============================================================
# PLL DATA PLATFORM — WAREHOUSE BUILDER
# SECTION 2A — IMPORTS, CONFIG, PATHS, API SESSION, HELPERS
# ============================================================
#
# This script is the GitHub/production version of the Colab
# database builder. It is designed to run from:
#
#   python scripts/build_warehouse.py
#
# Required environment variable:
#
#   PLL_BEARER_TOKEN
#
# Optional environment variables:
#
#   PLL_PROJECT_ROOT=data
#   PLL_TARGET_SEASONS=2022,2023,2024,2025,2026
#   PLL_FORCE_RECOLLECT=0
#   PLL_FORCE_REDISCOVER=0
#
# ============================================================

from __future__ import annotations

import os
import re
import json
import gzip
import time
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm


# ============================================================
# GLOBAL PANDAS OPTIONS
# ============================================================

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 250)


# ============================================================
# ENVIRONMENT / CONFIG HELPERS
# ============================================================

def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list_int(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default

    out: list[int] = []

    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))

    return out if out else default


def clean_token_value(x: Any) -> str:
    if x is None:
        return ""

    x = str(x).strip()
    x = x.replace("^", "").strip()
    x = re.sub(r"\s+", " ", x).strip()

    return x


def token_preview(tok: str) -> str:
    if not tok:
        return "MISSING"

    if len(tok) <= 24:
        return tok[:6] + "..."

    return f"{tok[:14]}...{tok[-6:]}"


def now_utc_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")


# ============================================================
# PROJECT PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = Path(os.environ.get("PLL_PROJECT_ROOT", "data"))

if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = REPO_ROOT / PROJECT_ROOT

SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"
API_RESPONSES_DIR = SOURCE_DATA_DIR / "api_responses"

STANDARDIZED_DATA_DIR = PROJECT_ROOT / "standardized_data"
GAME_TABLES_DIR = STANDARDIZED_DATA_DIR / "game_tables"
REFERENCE_TABLES_DIR = STANDARDIZED_DATA_DIR / "reference_tables"

CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
CURATED_ALL_DIR = CURATED_DATA_DIR / "all_requested_seasons"

ANALYTICS_DATABASE_DIR = PROJECT_ROOT / "analytics_database"
QUALITY_CHECKS_DIR = PROJECT_ROOT / "quality_checks"
CONFIG_DIR = PROJECT_ROOT / "config"
EXPORT_DIR = PROJECT_ROOT / "exports"

RUN_ID = utc_run_id()
RUN_CHECK_DIR = QUALITY_CHECKS_DIR / RUN_ID

DB_PATH = ANALYTICS_DATABASE_DIR / "pll_warehouse.duckdb"

MANUAL_SLUG_INVENTORY_FILE = CONFIG_DIR / "manual_slug_inventory.csv"

for p in [
    PROJECT_ROOT,
    SOURCE_DATA_DIR,
    API_RESPONSES_DIR,
    STANDARDIZED_DATA_DIR,
    GAME_TABLES_DIR,
    REFERENCE_TABLES_DIR,
    CURATED_DATA_DIR,
    CURATED_ALL_DIR,
    ANALYTICS_DATABASE_DIR,
    QUALITY_CHECKS_DIR,
    RUN_CHECK_DIR,
    CONFIG_DIR,
    EXPORT_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN CONFIG
# ============================================================

TARGET_SEASONS = env_list_int(
    "PLL_TARGET_SEASONS",
    [2022, 2023, 2024, 2025, 2026],
)

COMPETITION_TYPE = os.environ.get("PLL_COMPETITION_TYPE", "regular").strip() or "regular"

EXPECTED_REGULAR_GAMES: dict[int, Optional[int]] = {
    2022: 40,
    2023: 40,
    2024: 40,
    2025: 40,
    2026: None,  # ongoing / schedule-aware
}

PLL_STATS_SITE = "https://stats.premierlacrosseleague.com"
PLL_API_BASE = "https://api.stats.premierlacrosseleague.com/api/v4"
TIME_ZONE = os.environ.get("PLL_TIME_ZONE", "America/Los_Angeles").strip() or "America/Los_Angeles"

FORCE_RECOLLECT = env_bool("PLL_FORCE_RECOLLECT", False)
FORCE_REDISCOVER = env_bool("PLL_FORCE_REDISCOVER", False)

PLL_BEARER_TOKEN = clean_token_value(os.environ.get("PLL_BEARER_TOKEN", ""))


# ============================================================
# TEAM MAPPINGS
# ============================================================

TEAM_ID_CANONICAL_MAP = {
    "ATL": "ATL",
    "OUT": "OUT",
    "CAN": "CAN",
    "RED": "RED",
    "WAT": "WAT",
    "WHP": "WHP",
    "CHA": "CHA",
    "ARC": "ARC",
    "CHR": "OUT",   # Chrome historical franchise rolls into Outlaws
}

TEAM_NAME_CANONICAL_MAP = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Outlaws",
}

TEAM_NAME_LOOKUP_RAW = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Chrome",
}

TEAM_DISPLAY_NAME_LOOKUP = {
    "ATL": "New York Atlas",
    "OUT": "Denver Outlaws",
    "CAN": "Boston Cannons",
    "RED": "California Redwoods",
    "WAT": "Philadelphia Waterdogs",
    "WHP": "Maryland Whipsnakes",
    "CHA": "Carolina Chaos",
    "ARC": "Utah Archers",
}


def canonical_team_id(team_id: Any) -> Any:
    if pd.isna(team_id):
        return pd.NA

    team_id_str = str(team_id).strip()
    return TEAM_ID_CANONICAL_MAP.get(team_id_str, team_id_str)


def canonical_team_name(team_id_raw: Any, fallback_name: Any = None) -> Any:
    if pd.isna(team_id_raw):
        return fallback_name if fallback_name is not None else pd.NA

    team_id_raw = str(team_id_raw).strip()

    return TEAM_NAME_CANONICAL_MAP.get(
        team_id_raw,
        fallback_name if fallback_name is not None else team_id_raw,
    )


def resolve_team_name_raw(team_id_raw: Any, candidate_name: Any = None) -> Any:
    if pd.isna(team_id_raw) and pd.isna(candidate_name):
        return pd.NA

    raw_id = None if pd.isna(team_id_raw) else str(team_id_raw).strip()
    raw_name = None if pd.isna(candidate_name) else str(candidate_name).strip()

    if raw_name and raw_id and raw_name != raw_id:
        return raw_name

    if raw_name and not raw_id:
        return raw_name

    if raw_id:
        return TEAM_NAME_LOOKUP_RAW.get(raw_id, raw_id)

    return pd.NA


# ============================================================
# HTTP SESSION
# ============================================================

def build_session(bearer_token: str = "") -> requests.Session:
    session = requests.Session()

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": PLL_STATS_SITE,
        "pragma": "no-cache",
        "referer": f"{PLL_STATS_SITE}/",
        "time-zone": TIME_ZONE,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
    }

    if bearer_token:
        tok = clean_token_value(bearer_token)
        headers["authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
        headers["authsource"] = "stats"

    session.headers.update(headers)

    return session


SESSION = build_session(PLL_BEARER_TOKEN)


# ============================================================
# URL BUILDERS
# ============================================================

def event_list_url(year: int, season_segment: str = COMPETITION_TYPE) -> str:
    return f"{PLL_API_BASE}/events?year={year}&seasonSegment={season_segment}"


def event_summary_url(slug: str) -> str:
    return f"{PLL_API_BASE}/events/{slug}"


def player_game_stats_url(slug: str) -> str:
    return f"{PLL_API_BASE}/events/{slug}/players/stats"


def team_game_stats_url(slug: str) -> str:
    return f"{PLL_API_BASE}/events/{slug}/teams/stats"


# ============================================================
# FILE / JSON HELPERS
# ============================================================

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_gzip_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def read_gzip_json(path: Path | str) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch_url(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> requests.Response:
    if session is None:
        session = SESSION

    return session.get(url, timeout=timeout)


def fetch_json_with_cache(
    url: str,
    cache_path: Path | str,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    force: bool = False,
) -> tuple[Any, Optional[int], str]:
    if session is None:
        session = SESSION

    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        try:
            payload = read_gzip_json(cache_path)
            return payload, 200, "cached"
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    response = fetch_url(url, session=session, timeout=timeout)

    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code == 200 and payload is not None:
        write_gzip_json(cache_path, payload)

    return payload, response.status_code, "downloaded"


# ============================================================
# GENERAL DATA HELPERS
# ============================================================

def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d

    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default

    return cur


def snake_case(s: Any) -> str:
    s = str(s)
    s = re.sub(r"[%/\-]+", "_", s)
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def to_num_scalar(x: Any) -> float:
    try:
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    except Exception:
        v = np.nan

    return v


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def safe_nullable_int(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()

    if non_null.empty:
        return s.astype("Int64")

    if np.isclose(non_null % 1, 0).all():
        return s.round().astype("Int64")

    return s


def normalize_person_name(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None

    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x if x else None


def mode_or_first(s: pd.Series) -> Any:
    s2 = s.dropna()

    if len(s2) == 0:
        return pd.NA

    mode = s2.mode()

    if len(mode) > 0:
        return mode.iloc[0]

    return s2.iloc[0]


def latest_non_null_by_game(g: pd.DataFrame, col: str) -> Any:
    if col not in g.columns:
        return pd.NA

    sort_cols = [c for c in ["season", "game_number", "game_id"] if c in g.columns]

    if sort_cols:
        s = g.sort_values(sort_cols)[col].dropna()
    else:
        s = g[col].dropna()

    if len(s) == 0:
        return pd.NA

    return s.iloc[-1]


def extract_game_number_from_slug(slug: Any) -> Any:
    if pd.isna(slug):
        return pd.NA

    slug = str(slug)

    m1 = re.search(r"_game_(\d+)$", slug)
    if m1:
        return int(m1.group(1))

    m2 = re.search(r"^game-(\d+)-\d{4}-\d{2}-\d{2}$", slug)
    if m2:
        return int(m2.group(1))

    m3 = re.search(r"^(\d{4})-ev-(\d+)$", slug)
    if m3:
        return int(m3.group(2))

    return pd.NA


def extract_home_team_obj(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("homeTeam", {}) or {}


def extract_away_team_obj(data: dict[str, Any]) -> dict[str, Any]:
    for key in ["visitorTeam", "awayTeam", "visitor", "away"]:
        obj = data.get(key, {}) or {}
        if obj:
            return obj

    return {}


def extract_team_id_from_obj(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return pd.NA

    return obj.get("officialId") or obj.get("teamId") or obj.get("id")


def extract_team_name_from_obj(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return pd.NA

    return (
        obj.get("name")
        or obj.get("fullName")
        or obj.get("teamName")
        or obj.get("nickname")
        or obj.get("officialId")
        or obj.get("teamId")
        or obj.get("id")
    )


# ============================================================
# EVENT PAYLOAD / FIELD HELPERS
# ============================================================

def validate_event_payload(payload: Any, season: int) -> dict[str, Any]:
    data = safe_get(payload, "data", default={}) if payload else {}

    year_val = to_num_scalar(data.get("year"))
    event_id = data.get("eventId")
    event_numeric_id = data.get("id")
    season_segment = data.get("seasonSegment")
    slugname = data.get("slugname")
    start_time_unix = to_num_scalar(data.get("startTime"))
    event_status = data.get("eventStatus")

    valid = bool(
        not pd.isna(year_val)
        and int(year_val) == int(season)
        and event_id
        and season_segment == COMPETITION_TYPE
    )

    return {
        "valid": valid,
        "year": None if pd.isna(year_val) else int(year_val),
        "event_id": event_id,
        "event_numeric_id": event_numeric_id,
        "competition_type": season_segment,
        "slugname": slugname,
        "start_time_unix": None if pd.isna(start_time_unix) else int(start_time_unix),
        "event_status": event_status,
    }


def recursive_leaf_pairs(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            pairs.extend(recursive_leaf_pairs(v, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            pairs.extend(recursive_leaf_pairs(v, p))

    else:
        pairs.append((prefix, obj))

    return pairs


def find_numeric_leaf_candidates(
    obj: Any,
    normalized_terms: list[str],
) -> list[tuple[str, float]]:
    pairs = recursive_leaf_pairs(obj)
    out: list[tuple[str, float]] = []

    for raw_path, val in pairs:
        path_norm = snake_case(raw_path)

        if all(term in path_norm for term in normalized_terms):
            num = to_num_scalar(val)

            if not pd.isna(num):
                out.append((raw_path, num))

    return out


def coalesce_numeric_with_alt(
    item: dict[str, Any],
    direct_keys: list[str],
    alt_term_groups: list[list[str]],
    allow_zero: bool = True,
) -> float:
    for k in direct_keys:
        if k in item:
            val = to_num_scalar(item.get(k))

            if not pd.isna(val):
                if allow_zero or val != 0:
                    return val

    for term_group in alt_term_groups:
        cands = find_numeric_leaf_candidates(item, term_group)

        if cands:
            cands_sorted = sorted(cands, key=lambda x: (x[1] == 0, len(x[0])))
            best_val = cands_sorted[0][1]

            if allow_zero or best_val != 0:
                return best_val

    return np.nan


def derive_one_point_goals(
    total_goals: Any,
    raw_one_point_goals: Any,
    two_point_goals: Any,
) -> Any:
    tg = to_num_scalar(total_goals)
    rg = to_num_scalar(raw_one_point_goals)
    tw = to_num_scalar(two_point_goals)

    if not pd.isna(tg) and not pd.isna(tw):
        calc = tg - tw

        if pd.isna(rg) or not np.isclose(rg, calc):
            return calc

    return rg


def derive_scoring_points(one_point_goals: Any, two_point_goals: Any) -> float:
    one = to_num_scalar(one_point_goals)
    two = to_num_scalar(two_point_goals)

    if pd.isna(one) and pd.isna(two):
        return np.nan

    return (0 if pd.isna(one) else one) + 2 * (0 if pd.isna(two) else two)


def derive_player_points(
    raw_points: Any,
    scoring_points: Any,
    assists: Any,
) -> Any:
    rp = to_num_scalar(raw_points)
    sp = to_num_scalar(scoring_points)
    ast = to_num_scalar(assists)

    if not pd.isna(sp) and not pd.isna(ast):
        calc = sp + ast

        if pd.isna(rp) or not np.isclose(rp, calc):
            return calc

    return rp


# ============================================================
# RATE / ROLLING HELPERS
# ============================================================

def add_rate_columns(
    df: pd.DataFrame,
    denominator_col: str = "games",
    suffix: str = "_per_game",
    exclude_cols: Optional[set[str]] = None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    if denominator_col not in out.columns:
        return out

    if exclude_cols is None:
        exclude_cols = set()

    denominator = pd.to_numeric(out[denominator_col], errors="coerce").replace(0, np.nan)

    numeric_cols = [
        c for c in out.columns
        if c not in exclude_cols
        and c != denominator_col
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    for c in numeric_cols:
        rate_col = f"{c}{suffix}"

        if rate_col not in out.columns:
            out[rate_col] = pd.to_numeric(out[c], errors="coerce") / denominator

    return out


def safe_divide(numerator: Any, denominator: Any) -> Any:
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")

    return n / d.replace(0, np.nan)


def add_standard_player_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    if "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)

        rate_pairs = {
            "points": "points_per_game",
            "scoring_points": "scoring_points_per_game",
            "one_point_goals": "one_point_goals_per_game",
            "two_point_goals": "two_point_goals_per_game",
            "goals": "goals_per_game",
            "assists": "assists_per_game",
            "shots": "shots_per_game",
            "shots_on_goal": "shots_on_goal_per_game",
            "two_point_shots": "two_point_shots_per_game",
            "ground_balls": "ground_balls_per_game",
            "turnovers": "turnovers_per_game",
            "caused_turnovers": "caused_turnovers_per_game",
            "faceoffs_won": "faceoffs_won_per_game",
            "faceoffs_lost": "faceoffs_lost_per_game",
            "faceoffs": "faceoffs_per_game",
            "saves": "saves_per_game",
            "clean_saves": "clean_saves_per_game",
            "messy_saves": "messy_saves_per_game",
            "scores_against": "scores_against_per_game",
            "goals_against": "goals_against_per_game",
            "touches": "touches_per_game",
            "total_passes": "total_passes_per_game",
            "penalties": "penalties_per_game",
            "penalty_time": "penalty_time_per_game",
        }

        for total_col, rate_col in rate_pairs.items():
            if total_col in out.columns:
                out[rate_col] = pd.to_numeric(out[total_col], errors="coerce") / games

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = safe_divide(out["goals"], out["shots"])

    if "shots_on_goal" in out.columns and "shots" in out.columns:
        out["shots_on_goal_rate_calc"] = safe_divide(out["shots_on_goal"], out["shots"])

    if "faceoffs_won" in out.columns and "faceoffs" in out.columns:
        out["faceoff_pct_calc"] = safe_divide(out["faceoffs_won"], out["faceoffs"])

    if "saves" in out.columns:
        if "goals_against" in out.columns:
            ga = pd.to_numeric(out["goals_against"], errors="coerce")
        elif "scores_against" in out.columns:
            ga = pd.to_numeric(out["scores_against"], errors="coerce")
        else:
            ga = pd.Series(np.nan, index=out.index)

        saves = pd.to_numeric(out["saves"], errors="coerce")
        out["save_pct_calc"] = (saves / (saves + ga).replace(0, np.nan)).clip(0, 1)

    return out


def add_standard_team_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    if "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)

        rate_pairs = {
            "scores": "scores_per_game",
            "goals": "goals_per_game",
            "one_point_goals": "one_point_goals_per_game",
            "two_point_goals": "two_point_goals_per_game",
            "assists": "assists_per_game",
            "shots": "shots_per_game",
            "shots_on_goal": "shots_on_goal_per_game",
            "two_point_shots": "two_point_shots_per_game",
            "ground_balls": "ground_balls_per_game",
            "turnovers": "turnovers_per_game",
            "caused_turnovers": "caused_turnovers_per_game",
            "saves": "saves_per_game",
            "faceoffs_won": "faceoffs_won_per_game",
            "faceoffs_lost": "faceoffs_lost_per_game",
            "faceoffs": "faceoffs_per_game",
            "touches": "touches_per_game",
            "total_passes": "total_passes_per_game",
            "time_in_possession": "time_in_possession_per_game",
            "offensive_sequence_proxy": "offensive_sequence_proxy_per_game",
            "total_possessions": "total_possessions_per_game",
            "official_total_possessions": "official_total_possessions_per_game",
        }

        for total_col, rate_col in rate_pairs.items():
            if total_col in out.columns:
                out[rate_col] = pd.to_numeric(out[total_col], errors="coerce") / games

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = safe_divide(out["goals"], out["shots"])

    if "shots_on_goal" in out.columns and "shots" in out.columns:
        out["shots_on_goal_rate_calc"] = safe_divide(out["shots_on_goal"], out["shots"])

    if "faceoffs_won" in out.columns and "faceoffs" in out.columns:
        out["faceoff_pct_calc"] = safe_divide(out["faceoffs_won"], out["faceoffs"])

    if "total_clears" in out.columns and "clear_attempts" in out.columns:
        out["clear_pct_calc"] = safe_divide(out["total_clears"], out["clear_attempts"])

    if "scores" in out.columns and "offensive_sequence_proxy" in out.columns:
        out["scores_per_offensive_sequence_proxy"] = safe_divide(
            out["scores"],
            out["offensive_sequence_proxy"],
        )

    return out


# ============================================================
# TABLE EXPORT HELPERS
# ============================================================

def ensure_non_empty_schema(df: Any, table_name: str) -> pd.DataFrame:
    """
    DuckDB cannot read Parquet files with zero columns.
    This guarantees every exported table has at least one column.
    """

    if df is None:
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": ["table_was_none"],
        })

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": ["not_a_dataframe"],
        })

    if len(df.columns) == 0:
        return pd.DataFrame(columns=[
            "_empty_table_name",
            "_note",
            "season",
            "game_slug",
            "reason",
            "error",
        ])

    return df


def write_table_artifacts(
    name: str,
    df: pd.DataFrame,
    artifact_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    df_safe = ensure_non_empty_schema(df, name)

    parquet_path = CURATED_ALL_DIR / f"{name}.parquet"
    csv_path = CURATED_ALL_DIR / f"{name}.csv"

    df_safe.to_parquet(parquet_path, index=False)
    df_safe.to_csv(csv_path, index=False)

    artifact_rows.append({
        "table_name": name,
        "rows": len(df_safe),
        "columns": len(df_safe.columns),
        "parquet_path": str(parquet_path.relative_to(REPO_ROOT)) if parquet_path.is_relative_to(REPO_ROOT) else str(parquet_path),
        "csv_path": str(csv_path.relative_to(REPO_ROOT)) if csv_path.is_relative_to(REPO_ROOT) else str(csv_path),
        "updated_at_utc": now_utc_iso(),
    })

    return df_safe


def duckdb_load_parquet(
    con: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> None:
    fp = CURATED_ALL_DIR / f"{table_name}.parquet"

    if not fp.exists():
        print(f"Skipping missing file: {fp}")
        return

    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")

    sql = (
        f"CREATE OR REPLACE TABLE {schema_name}.{table_name} AS "
        f"SELECT * FROM read_parquet('{fp.as_posix()}');"
    )

    con.execute(sql)


# ============================================================
# QC HELPERS
# ============================================================

quality_rows: list[dict[str, Any]] = []


def add_qc_check(
    check_name: str,
    status: str,
    actual: Any = None,
    expected: Any = None,
    notes: str = "",
) -> None:
    quality_rows.append({
        "check_name": check_name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "notes": notes,
        "run_id": RUN_ID,
        "checked_at_utc": now_utc_iso(),
    })


# ============================================================
# STARTUP LOGGING
# ============================================================

def print_startup_summary() -> None:
    print("=" * 90)
    print("PLL DATA PLATFORM — WAREHOUSE BUILDER")
    print("=" * 90)
    print("Repository root:", REPO_ROOT)
    print("Project root:", PROJECT_ROOT)
    print("DuckDB path:", DB_PATH)
    print("Curated dir:", CURATED_ALL_DIR)
    print("Run check dir:", RUN_CHECK_DIR)
    print("Target seasons:", TARGET_SEASONS)
    print("Competition type:", COMPETITION_TYPE)
    print("Force recollect:", FORCE_RECOLLECT)
    print("Force rediscover:", FORCE_REDISCOVER)
    print("Token loaded:", bool(PLL_BEARER_TOKEN))
    print("Token preview:", token_preview(PLL_BEARER_TOKEN))
    print("Authorization header present:", "authorization" in SESSION.headers)
    print("=" * 90)


def require_api_token() -> None:
    if not PLL_BEARER_TOKEN:
        raise RuntimeError(
            "PLL_BEARER_TOKEN is missing. Add it as a GitHub Actions secret "
            "or set it as an environment variable before running the builder."
        )


# ============================================================
# SECTION 2A COMPLETE
# ============================================================

print_startup_summary()

# ============================================================
# SECTION 2B — GAME DISCOVERY AND SCHEDULE COLLECTION
# ============================================================

def unwrap_payload_list(payload: Any) -> list[dict[str, Any]]:
    """
    Flexible unwrapping for PLL API responses.

    Supported shapes:
      {"data": [...]}
      {"data": {"events": [...]}}
      {"events": [...]}
      [...]
    """

    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    data = payload.get("data", payload)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in [
            "events",
            "games",
            "items",
            "results",
            "records",
            "rows",
            "data",
        ]:
            val = data.get(key)

            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]

    for key in [
        "events",
        "games",
        "items",
        "results",
        "records",
        "rows",
    ]:
        val = payload.get(key)

        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]

    return []


def extract_slug_from_event_item(item: dict[str, Any]) -> Any:
    for key in [
        "slugname",
        "slug",
        "eventSlug",
        "gameSlug",
        "permalink",
    ]:
        val = item.get(key)

        if val is not None and str(val).strip():
            return str(val).strip()

    return pd.NA


def extract_event_id_from_event_item(item: dict[str, Any]) -> Any:
    for key in [
        "eventId",
        "event_id",
        "gameId",
        "game_id",
        "id",
    ]:
        val = item.get(key)

        if val is not None and str(val).strip():
            return val

    return pd.NA


def extract_event_status_from_item(item: dict[str, Any]) -> Any:
    for key in [
        "eventStatus",
        "status",
        "statusLabel",
        "gameStatus",
        "state",
    ]:
        val = item.get(key)

        if val is not None and str(val).strip():
            return str(val).strip()

    return pd.NA


def extract_start_time_unix(item: dict[str, Any]) -> Any:
    for key in [
        "startTime",
        "start_time",
        "startTimestamp",
        "gameTime",
        "date",
        "datetime",
    ]:
        val = item.get(key)

        if val is None:
            continue

        num = to_num_scalar(val)

        if not pd.isna(num):
            return int(num)

    return pd.NA


def unix_to_utc_date(start_time_unix: Any) -> Any:
    num = to_num_scalar(start_time_unix)

    if pd.isna(num):
        return pd.NA

    try:
        # PLL startTime is usually seconds, but guard against milliseconds.
        if num > 10_000_000_000:
            num = num / 1000

        return (
            dt.datetime
            .fromtimestamp(float(num), tz=dt.timezone.utc)
            .date()
            .isoformat()
        )
    except Exception:
        return pd.NA


def parse_score_from_team_obj(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return pd.NA

    for key in [
        "score",
        "scores",
        "goals",
        "totalScore",
        "points",
    ]:
        if key in obj:
            val = to_num_scalar(obj.get(key))

            if not pd.isna(val):
                return val

    return pd.NA


def parse_event_item_to_schedule_row(
    item: dict[str, Any],
    season: int,
    source: str,
) -> dict[str, Any]:
    slug = extract_slug_from_event_item(item)
    event_id = extract_event_id_from_event_item(item)

    home_obj = extract_home_team_obj(item)
    away_obj = extract_away_team_obj(item)

    home_team_id_raw = extract_team_id_from_obj(home_obj)
    away_team_id_raw = extract_team_id_from_obj(away_obj)

    home_team_name_raw = extract_team_name_from_obj(home_obj)
    away_team_name_raw = extract_team_name_from_obj(away_obj)

    home_score = parse_score_from_team_obj(home_obj)
    away_score = parse_score_from_team_obj(away_obj)

    start_time_unix = extract_start_time_unix(item)
    game_date_utc = unix_to_utc_date(start_time_unix)

    game_number = extract_game_number_from_slug(slug)

    if pd.isna(game_number):
        gn = to_num_scalar(
            item.get("gameNumber")
            or item.get("game_number")
            or item.get("eventNumber")
            or item.get("event_number")
        )

        game_number = int(gn) if not pd.isna(gn) else pd.NA

    event_status = extract_event_status_from_item(item)

    return {
        "season": int(season),
        "competition_type": item.get("seasonSegment", COMPETITION_TYPE),
        "event_id": event_id,
        "event_numeric_id": item.get("id"),
        "game_id": f"{season}_game_{game_number}" if not pd.isna(game_number) else str(event_id or slug),
        "game_slug": slug,
        "schedule_slug": slug,
        "game_number": game_number,
        "game_date_utc": game_date_utc,
        "start_time_unix": start_time_unix,
        "event_status": event_status,
        "status_display": event_status,
        "home_team_id_raw": home_team_id_raw,
        "away_team_id_raw": away_team_id_raw,
        "home_team_id": canonical_team_id(home_team_id_raw),
        "away_team_id": canonical_team_id(away_team_id_raw),
        "home_team_name_raw": resolve_team_name_raw(home_team_id_raw, home_team_name_raw),
        "away_team_name_raw": resolve_team_name_raw(away_team_id_raw, away_team_name_raw),
        "home_team_name": canonical_team_name(home_team_id_raw, home_team_name_raw),
        "away_team_name": canonical_team_name(away_team_id_raw, away_team_name_raw),
        "home_score": home_score,
        "away_score": away_score,
        "source": source,
        "discovered_at_utc": now_utc_iso(),
    }


def parse_event_summary_to_schedule_row(
    payload: Any,
    season: int,
    slug: str,
    source: str,
) -> dict[str, Any]:
    data = safe_get(payload, "data", default={}) if isinstance(payload, dict) else {}

    if not isinstance(data, dict):
        data = {}

    base_row = parse_event_item_to_schedule_row(data, season=season, source=source)

    if pd.isna(base_row.get("game_slug")):
        base_row["game_slug"] = slug
        base_row["schedule_slug"] = slug

    if pd.isna(base_row.get("game_number")):
        gn = extract_game_number_from_slug(slug)
        base_row["game_number"] = gn
        base_row["game_id"] = f"{season}_game_{gn}" if not pd.isna(gn) else str(base_row.get("event_id") or slug)

    return base_row


def load_manual_slug_inventory() -> pd.DataFrame:
    """
    Optional manual override file. This is useful if the PLL API event list
    misses older historical games.

    Expected optional columns:
      season, game_slug, game_number, game_date_utc
    """

    if not MANUAL_SLUG_INVENTORY_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(MANUAL_SLUG_INVENTORY_FILE)

    if "season" not in df.columns or "game_slug" not in df.columns:
        print(f"Manual slug inventory exists but is missing required columns: {MANUAL_SLUG_INVENTORY_FILE}")
        return pd.DataFrame()

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["game_slug"] = df["game_slug"].astype(str).str.strip()

    if "schedule_slug" not in df.columns:
        df["schedule_slug"] = df["game_slug"]

    if "game_number" not in df.columns:
        df["game_number"] = df["game_slug"].apply(extract_game_number_from_slug)

    if "game_id" not in df.columns:
        df["game_id"] = df.apply(
            lambda r: (
                f"{int(r['season'])}_game_{int(r['game_number'])}"
                if not pd.isna(r["season"]) and not pd.isna(r["game_number"])
                else f"{r['season']}_{r['game_slug']}"
            ),
            axis=1,
        )

    df["source"] = "manual_slug_inventory"
    df["discovered_at_utc"] = now_utc_iso()

    return df


def discover_games_for_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discovers game slugs and schedule metadata for a single season.

    Returns:
      schedule_df
      probe_rows_df
      discovery_log_df
    """

    season_dir = API_RESPONSES_DIR / f"season_{season}"
    season_dir.mkdir(parents=True, exist_ok=True)

    event_list_cache = season_dir / "event_list.json.gz"
    url = event_list_url(season)

    probe_rows: list[dict[str, Any]] = []
    discovery_log_rows: list[dict[str, Any]] = []

    payload, status_code, cache_status = fetch_json_with_cache(
        url,
        event_list_cache,
        force=FORCE_REDISCOVER,
    )

    events = unwrap_payload_list(payload)

    probe_rows.append({
        "season": season,
        "url": url,
        "status_code": status_code,
        "cache_status": cache_status,
        "event_rows": len(events),
        "payload_type": type(payload).__name__,
        "checked_at_utc": now_utc_iso(),
    })

    schedule_rows: list[dict[str, Any]] = []

    for item in events:
        row = parse_event_item_to_schedule_row(
            item,
            season=season,
            source="event_list",
        )

        if not pd.isna(row.get("game_slug")):
            schedule_rows.append(row)

    # Enrich using event summary endpoint when slugs are available.
    enriched_rows: list[dict[str, Any]] = []

    for row in schedule_rows:
        slug = row.get("game_slug")

        if pd.isna(slug) or not str(slug).strip():
            continue

        game_dir = season_dir / f"game_{slug}"
        game_dir.mkdir(parents=True, exist_ok=True)

        summary_cache = game_dir / "event_summary.json.gz"
        summary_url = event_summary_url(str(slug))

        summary_payload, summary_status, summary_cache_status = fetch_json_with_cache(
            summary_url,
            summary_cache,
            force=FORCE_REDISCOVER,
        )

        discovery_log_rows.append({
            "season": season,
            "game_slug": slug,
            "url": summary_url,
            "status_code": summary_status,
            "cache_status": summary_cache_status,
            "valid_event_summary": validate_event_payload(summary_payload, season).get("valid", False),
            "checked_at_utc": now_utc_iso(),
        })

        if summary_status == 200 and summary_payload:
            enriched = parse_event_summary_to_schedule_row(
                summary_payload,
                season=season,
                slug=str(slug),
                source="event_summary",
            )

            # Fill any missing summary fields from list row.
            for k, v in row.items():
                if k not in enriched or pd.isna(enriched.get(k)):
                    enriched[k] = v

            enriched_rows.append(enriched)
        else:
            enriched_rows.append(row)

        time.sleep(0.03)

    if enriched_rows:
        out = pd.DataFrame(enriched_rows)
    else:
        out = pd.DataFrame(schedule_rows)

    manual_df = load_manual_slug_inventory()

    if len(manual_df) > 0:
        manual_season = manual_df[manual_df["season"] == season].copy()

        if len(manual_season) > 0:
            out = pd.concat([out, manual_season], ignore_index=True, sort=False)

    if len(out) > 0:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
        out["game_slug"] = out["game_slug"].astype(str)

        if "schedule_slug" not in out.columns:
            out["schedule_slug"] = out["game_slug"]

        out["schedule_slug"] = out["schedule_slug"].fillna(out["game_slug"]).astype(str)

        if "game_number" not in out.columns:
            out["game_number"] = out["game_slug"].apply(extract_game_number_from_slug)
        else:
            out["game_number"] = out["game_number"].fillna(
                out["game_slug"].apply(extract_game_number_from_slug)
            )

        out["game_number"] = safe_nullable_int(out["game_number"])

        if "game_id" not in out.columns:
            out["game_id"] = out.apply(
                lambda r: (
                    f"{int(r['season'])}_game_{int(r['game_number'])}"
                    if not pd.isna(r["season"]) and not pd.isna(r["game_number"])
                    else f"{r['season']}_{r['game_slug']}"
                ),
                axis=1,
            )
        else:
            out["game_id"] = out["game_id"].fillna(
                out.apply(
                    lambda r: (
                        f"{int(r['season'])}_game_{int(r['game_number'])}"
                        if not pd.isna(r["season"]) and not pd.isna(r["game_number"])
                        else f"{r['season']}_{r['game_slug']}"
                    ),
                    axis=1,
                )
            )

        for col in [
            "home_team_id",
            "away_team_id",
        ]:
            if col in out.columns:
                out[col] = out[col].apply(canonical_team_id)

        if "home_team_name" in out.columns and "home_team_id_raw" in out.columns:
            out["home_team_name"] = out.apply(
                lambda r: canonical_team_name(r.get("home_team_id_raw"), r.get("home_team_name")),
                axis=1,
            )

        if "away_team_name" in out.columns and "away_team_id_raw" in out.columns:
            out["away_team_name"] = out.apply(
                lambda r: canonical_team_name(r.get("away_team_id_raw"), r.get("away_team_name")),
                axis=1,
            )

        # Deduplicate by season/slug, preferring event_summary rows over event_list/manual rows.
        source_priority = {
            "event_summary": 1,
            "event_list": 2,
            "manual_slug_inventory": 3,
        }

        out["_source_priority"] = out["source"].map(source_priority).fillna(9)
        out = (
            out.sort_values(["season", "game_number", "_source_priority"])
            .drop_duplicates(["season", "game_slug"], keep="first")
            .drop(columns=["_source_priority"])
            .reset_index(drop=True)
        )

    probe_df = pd.DataFrame(probe_rows)
    discovery_df = pd.DataFrame(discovery_log_rows)

    return out, probe_df, discovery_df


def discover_all_games(seasons: list[int]) -> dict[str, pd.DataFrame]:
    """
    Discovers games for all requested seasons.
    """

    schedule_frames: list[pd.DataFrame] = []
    probe_frames: list[pd.DataFrame] = []
    discovery_frames: list[pd.DataFrame] = []

    for season in tqdm(seasons, desc="Discovering PLL games"):
        try:
            schedule_df, probe_df, discovery_df = discover_games_for_season(season)

            schedule_frames.append(schedule_df)
            probe_frames.append(probe_df)
            discovery_frames.append(discovery_df)

            add_qc_check(
                check_name=f"event_discovery_{season}",
                status="pass" if len(schedule_df) > 0 else "warn",
                actual=len(schedule_df),
                expected=EXPECTED_REGULAR_GAMES.get(season),
                notes="Discovered schedule rows from PLL event APIs.",
            )

        except Exception as exc:
            add_qc_check(
                check_name=f"event_discovery_{season}",
                status="fail",
                actual=0,
                expected=EXPECTED_REGULAR_GAMES.get(season),
                notes=str(exc),
            )

            discovery_frames.append(pd.DataFrame([{
                "season": season,
                "game_slug": pd.NA,
                "url": event_list_url(season),
                "status_code": pd.NA,
                "cache_status": "error",
                "valid_event_summary": False,
                "error": str(exc),
                "checked_at_utc": now_utc_iso(),
            }]))

    game_schedule_all = (
        pd.concat(schedule_frames, ignore_index=True, sort=False)
        if schedule_frames
        else pd.DataFrame()
    )

    event_list_probe_summary = (
        pd.concat(probe_frames, ignore_index=True, sort=False)
        if probe_frames
        else pd.DataFrame()
    )

    game_discovery_log = (
        pd.concat(discovery_frames, ignore_index=True, sort=False)
        if discovery_frames
        else pd.DataFrame()
    )

    if len(game_schedule_all) > 0:
        # Completed flag is intentionally liberal. Exact stat availability is confirmed later.
        for col in ["home_score", "away_score"]:
            if col in game_schedule_all.columns:
                game_schedule_all[col] = pd.to_numeric(game_schedule_all[col], errors="coerce")

        game_schedule_all["has_final_score"] = (
            game_schedule_all.get("home_score", pd.Series(np.nan, index=game_schedule_all.index)).notna()
            & game_schedule_all.get("away_score", pd.Series(np.nan, index=game_schedule_all.index)).notna()
        )

        game_schedule_all["is_completed_by_schedule"] = game_schedule_all["has_final_score"]

        if "event_status" in game_schedule_all.columns:
            status_text = game_schedule_all["event_status"].astype(str).str.lower()
            game_schedule_all["is_completed_by_schedule"] = (
                game_schedule_all["is_completed_by_schedule"]
                | status_text.str.contains("final|complete|completed|closed", regex=True, na=False)
            )

        game_schedule_all["matchup"] = (
            game_schedule_all.get("away_team_name", pd.Series("", index=game_schedule_all.index)).astype(str)
            + " at "
            + game_schedule_all.get("home_team_name", pd.Series("", index=game_schedule_all.index)).astype(str)
        )

        game_schedule_all["result"] = np.where(
            game_schedule_all["has_final_score"],
            game_schedule_all["away_score"].astype("Int64").astype(str)
            + " - "
            + game_schedule_all["home_score"].astype("Int64").astype(str),
            pd.NA,
        )

        game_schedule_all = game_schedule_all.sort_values(
            ["season", "game_number", "game_date_utc", "game_slug"],
            na_position="last",
        ).reset_index(drop=True)

    if len(game_schedule_all) > 0:
        season_slug_inventory = game_schedule_all[[
            c for c in [
                "season",
                "game_id",
                "game_slug",
                "schedule_slug",
                "game_number",
                "game_date_utc",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "source",
            ]
            if c in game_schedule_all.columns
        ]].copy()
    else:
        season_slug_inventory = pd.DataFrame()

    schedule_2026 = (
        game_schedule_all[game_schedule_all["season"] == 2026].copy()
        if len(game_schedule_all) > 0 and "season" in game_schedule_all.columns
        else pd.DataFrame()
    )

    return {
        "game_schedule_all": game_schedule_all,
        "game_schedule_2026": schedule_2026,
        "season_slug_inventory": season_slug_inventory,
        "event_list_probe_summary": event_list_probe_summary,
        "game_discovery_log": game_discovery_log,
    }


# ============================================================
# SECTION 2B COMPLETE
# ============================================================

# ============================================================
# SECTION 2C — STAT SCRAPING AND RAW PARSING
# ============================================================

def recursive_find_lists(obj: Any, path: str = "") -> list[tuple[str, list[Any]]]:
    found: list[tuple[str, list[Any]]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else str(k)

            if isinstance(v, list):
                found.append((child_path, v))

            found.extend(recursive_find_lists(v, child_path))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_path = f"{path}[{i}]"
            found.extend(recursive_find_lists(v, child_path))

    return found


def score_candidate_list_for_kind(path: str, values: list[Any], kind: str) -> int:
    if not isinstance(values, list) or not values:
        return -999

    dict_count = sum(isinstance(v, dict) for v in values)

    if dict_count == 0:
        return -999

    sample = [v for v in values if isinstance(v, dict)][:5]
    sample_keys = " ".join(" ".join(map(str, s.keys())) for s in sample).lower()
    path_l = path.lower()

    score = dict_count

    if kind == "player":
        positive_terms = [
            "player",
            "athlete",
            "roster",
            "stat",
            "stats",
        ]

        negative_terms = [
            "teamstats",
            "team_stats",
            "teams",
            "officials",
            "broadcast",
        ]

        identity_terms = [
            "first",
            "last",
            "full",
            "name",
            "position",
        ]

    else:
        positive_terms = [
            "team",
            "teamstats",
            "team_stats",
            "stat",
            "stats",
        ]

        negative_terms = [
            "player",
            "athlete",
            "roster",
            "officials",
            "broadcast",
        ]

        identity_terms = [
            "officialid",
            "teamid",
            "team",
            "name",
        ]

    for term in positive_terms:
        if term in path_l or term in sample_keys:
            score += 10

    for term in identity_terms:
        if term in sample_keys:
            score += 5

    for term in negative_terms:
        if term in path_l:
            score -= 20

    return score


def extract_stat_items(payload: Any, kind: str) -> list[dict[str, Any]]:
    """
    Finds the most likely list of player/team stat rows in a flexible payload.
    """

    if payload is None:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        direct = [x for x in payload["data"] if isinstance(x, dict)]
        if direct:
            return direct

    if isinstance(payload, list):
        direct = [x for x in payload if isinstance(x, dict)]
        if direct:
            return direct

    candidates = recursive_find_lists(payload)
    scored = []

    for path, values in candidates:
        score = score_candidate_list_for_kind(path, values, kind)
        if score > -999:
            scored.append((score, path, values))

    if not scored:
        return []

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    best_values = scored[0][2]

    return [x for x in best_values if isinstance(x, dict)]


def extract_nested_obj(item: dict[str, Any], candidate_keys: list[str]) -> dict[str, Any]:
    for key in candidate_keys:
        val = item.get(key)

        if isinstance(val, dict):
            return val

    return {}


def extract_player_obj(item: dict[str, Any]) -> dict[str, Any]:
    return extract_nested_obj(
        item,
        [
            "player",
            "athlete",
            "person",
            "participant",
            "user",
        ],
    )


def extract_team_obj_from_stat_item(item: dict[str, Any]) -> dict[str, Any]:
    return extract_nested_obj(
        item,
        [
            "team",
            "club",
            "franchise",
        ],
    )


def extract_player_id_from_item(item: dict[str, Any]) -> Any:
    player = extract_player_obj(item)

    for source in [item, player]:
        for key in [
            "playerId",
            "player_id",
            "officialId",
            "id",
            "personId",
            "athleteId",
        ]:
            val = source.get(key) if isinstance(source, dict) else None

            if val is not None and str(val).strip():
                return val

    return pd.NA


def extract_player_name_from_item(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    player = extract_player_obj(item)

    first_name = pd.NA
    last_name = pd.NA
    full_name = pd.NA

    for source in [item, player]:
        if not isinstance(source, dict):
            continue

        for key in ["firstName", "first_name", "givenName"]:
            val = source.get(key)
            if val is not None and str(val).strip():
                first_name = str(val).strip()
                break

        for key in ["lastName", "last_name", "familyName"]:
            val = source.get(key)
            if val is not None and str(val).strip():
                last_name = str(val).strip()
                break

        for key in ["fullName", "full_name", "displayName", "name"]:
            val = source.get(key)
            if val is not None and str(val).strip():
                full_name = str(val).strip()
                break

    if pd.isna(full_name):
        if not pd.isna(first_name) or not pd.isna(last_name):
            full_name = f"{'' if pd.isna(first_name) else first_name} {'' if pd.isna(last_name) else last_name}".strip()

    return first_name, last_name, full_name


def extract_position_from_item(item: dict[str, Any]) -> tuple[Any, Any]:
    player = extract_player_obj(item)

    position = pd.NA
    position_name = pd.NA

    for source in [item, player]:
        if not isinstance(source, dict):
            continue

        for key in [
            "position",
            "pos",
            "positionCode",
            "positionAbbreviation",
        ]:
            val = source.get(key)

            if val is not None and str(val).strip():
                if isinstance(val, dict):
                    position = val.get("abbreviation") or val.get("code") or val.get("name")
                else:
                    position = str(val).strip()
                break

        for key in [
            "positionName",
            "position_name",
            "positionLabel",
        ]:
            val = source.get(key)

            if val is not None and str(val).strip():
                position_name = str(val).strip()
                break

    if pd.isna(position_name) and not pd.isna(position):
        pos_map = {
            "A": "Attack",
            "M": "Midfield",
            "D": "Defense",
            "G": "Goalie",
            "FO": "Faceoff",
            "LSM": "Long Stick Midfield",
            "SSDM": "Short Stick Defensive Midfield",
        }

        position_name = pos_map.get(str(position).strip(), pd.NA)

    return position, position_name


def extract_stat_team_id_and_name(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    team_obj = extract_team_obj_from_stat_item(item)

    team_id_raw = pd.NA
    team_name_raw = pd.NA

    for source in [item, team_obj]:
        if not isinstance(source, dict):
            continue

        for key in [
            "teamId",
            "team_id",
            "officialId",
            "teamOfficialId",
            "team",
            "clubId",
        ]:
            val = source.get(key)

            if isinstance(val, dict):
                continue

            if val is not None and str(val).strip():
                team_id_raw = str(val).strip()
                break

        for key in [
            "teamName",
            "team_name",
            "name",
            "fullName",
            "clubName",
        ]:
            val = source.get(key)

            if val is not None and str(val).strip():
                team_name_raw = str(val).strip()
                break

    if pd.isna(team_id_raw):
        team_id_raw = extract_team_id_from_obj(team_obj)

    if pd.isna(team_name_raw):
        team_name_raw = extract_team_name_from_obj(team_obj)

    team_id = canonical_team_id(team_id_raw)
    team_name = canonical_team_name(team_id_raw, team_name_raw)

    return team_id_raw, team_id, team_name


def infer_opponent_for_team(
    team_id: Any,
    home_team_id: Any,
    away_team_id: Any,
    home_team_name: Any,
    away_team_name: Any,
) -> tuple[Any, Any]:
    team_id_c = canonical_team_id(team_id)
    home_id_c = canonical_team_id(home_team_id)
    away_id_c = canonical_team_id(away_team_id)

    if not pd.isna(team_id_c) and not pd.isna(home_id_c) and str(team_id_c) == str(home_id_c):
        return away_id_c, away_team_name

    if not pd.isna(team_id_c) and not pd.isna(away_id_c) and str(team_id_c) == str(away_id_c):
        return home_id_c, home_team_name

    return pd.NA, pd.NA


def parse_player_stat_item(
    item: dict[str, Any],
    schedule_row: pd.Series,
    source_path: str,
) -> dict[str, Any]:
    first_name, last_name, full_name = extract_player_name_from_item(item)
    position, position_name = extract_position_from_item(item)
    team_id_raw, team_id, team_name = extract_stat_team_id_and_name(item)

    opponent_team_id, opponent_team_name = infer_opponent_for_team(
        team_id,
        schedule_row.get("home_team_id"),
        schedule_row.get("away_team_id"),
        schedule_row.get("home_team_name"),
        schedule_row.get("away_team_name"),
    )

    raw_one_point_goals = coalesce_numeric_with_alt(
        item,
        [
            "onePointGoals",
            "one_point_goals",
            "onePtGoals",
            "one_pt_goals",
            "goalsOnePoint",
            "goals1pt",
        ],
        [["one", "point", "goal"], ["1", "pt", "goal"]],
        allow_zero=True,
    )

    two_point_goals = coalesce_numeric_with_alt(
        item,
        [
            "twoPointGoals",
            "two_point_goals",
            "twoPtGoals",
            "two_pt_goals",
            "goalsTwoPoint",
            "goals2pt",
        ],
        [["two", "point", "goal"], ["2", "pt", "goal"]],
        allow_zero=True,
    )

    goals = coalesce_numeric_with_alt(
        item,
        ["goals", "goal", "g"],
        [["goal"]],
        allow_zero=True,
    )

    one_point_goals = derive_one_point_goals(
        goals,
        raw_one_point_goals,
        two_point_goals,
    )

    scoring_points = derive_scoring_points(one_point_goals, two_point_goals)

    assists = coalesce_numeric_with_alt(
        item,
        ["assists", "assist", "a"],
        [["assist"]],
        allow_zero=True,
    )

    raw_points = coalesce_numeric_with_alt(
        item,
        ["points", "pts", "totalPoints"],
        [["point"]],
        allow_zero=True,
    )

    points = derive_player_points(raw_points, scoring_points, assists)

    faceoffs_won = coalesce_numeric_with_alt(
        item,
        [
            "faceoffsWon",
            "faceoffWins",
            "faceoffs_won",
            "foWins",
            "fo_wins",
        ],
        [["faceoff", "won"], ["faceoff", "win"], ["fo", "win"]],
        allow_zero=True,
    )

    faceoffs_lost = coalesce_numeric_with_alt(
        item,
        [
            "faceoffsLost",
            "faceoffLosses",
            "faceoffs_lost",
            "foLosses",
            "fo_losses",
        ],
        [["faceoff", "lost"], ["faceoff", "loss"], ["fo", "loss"]],
        allow_zero=True,
    )

    faceoffs = coalesce_numeric_with_alt(
        item,
        ["faceoffs", "faceoffAttempts", "foAttempts", "fo"],
        [["faceoff"]],
        allow_zero=True,
    )

    if pd.isna(faceoffs) and not pd.isna(faceoffs_won) and not pd.isna(faceoffs_lost):
        faceoffs = faceoffs_won + faceoffs_lost

    row = {
        "season": schedule_row.get("season"),
        "game_id": schedule_row.get("game_id"),
        "game_slug": schedule_row.get("game_slug"),
        "schedule_slug": schedule_row.get("schedule_slug"),
        "game_number": schedule_row.get("game_number"),
        "game_date_utc": schedule_row.get("game_date_utc"),
        "event_status": schedule_row.get("event_status"),
        "player_id": extract_player_id_from_item(item),
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "normalized_name": normalize_person_name(full_name),
        "position": position,
        "position_name": position_name,
        "team_id_raw": team_id_raw,
        "team_id": team_id,
        "team_name": team_name,
        "opponent_team_id": opponent_team_id,
        "opponent_team_name": opponent_team_name,
        "is_home": (
            str(team_id) == str(schedule_row.get("home_team_id"))
            if not pd.isna(team_id) and not pd.isna(schedule_row.get("home_team_id"))
            else pd.NA
        ),
        "points": points,
        "scoring_points": scoring_points,
        "one_point_goals": one_point_goals,
        "two_point_goals": two_point_goals,
        "goals": goals,
        "assists": assists,
        "shots": coalesce_numeric_with_alt(
            item,
            ["shots", "shot", "sh"],
            [["shot"]],
            allow_zero=True,
        ),
        "shots_on_goal": coalesce_numeric_with_alt(
            item,
            ["shotsOnGoal", "shots_on_goal", "sog"],
            [["shot", "goal"], ["sog"]],
            allow_zero=True,
        ),
        "two_point_shots": coalesce_numeric_with_alt(
            item,
            ["twoPointShots", "two_point_shots", "twoPtShots", "two_pt_shots"],
            [["two", "point", "shot"], ["2", "pt", "shot"]],
            allow_zero=True,
        ),
        "ground_balls": coalesce_numeric_with_alt(
            item,
            ["groundBalls", "ground_balls", "gb"],
            [["ground", "ball"], ["gb"]],
            allow_zero=True,
        ),
        "turnovers": coalesce_numeric_with_alt(
            item,
            ["turnovers", "turnover", "to"],
            [["turnover"]],
            allow_zero=True,
        ),
        "caused_turnovers": coalesce_numeric_with_alt(
            item,
            ["causedTurnovers", "caused_turnovers", "ct"],
            [["caused", "turnover"], ["ct"]],
            allow_zero=True,
        ),
        "faceoffs_won": faceoffs_won,
        "faceoffs_lost": faceoffs_lost,
        "faceoffs": faceoffs,
        "saves": coalesce_numeric_with_alt(
            item,
            ["saves", "save", "sv"],
            [["save"]],
            allow_zero=True,
        ),
        "clean_saves": coalesce_numeric_with_alt(
            item,
            ["cleanSaves", "clean_saves"],
            [["clean", "save"]],
            allow_zero=True,
        ),
        "messy_saves": coalesce_numeric_with_alt(
            item,
            ["messySaves", "messy_saves"],
            [["messy", "save"]],
            allow_zero=True,
        ),
        "scores_against": coalesce_numeric_with_alt(
            item,
            ["scoresAgainst", "scores_against", "scoreAgainst"],
            [["score", "against"]],
            allow_zero=True,
        ),
        "goals_against": coalesce_numeric_with_alt(
            item,
            ["goalsAgainst", "goals_against", "goalAgainst", "ga"],
            [["goal", "against"], ["ga"]],
            allow_zero=True,
        ),
        "penalties": coalesce_numeric_with_alt(
            item,
            ["penalties", "penalty"],
            [["penalty"]],
            allow_zero=True,
        ),
        "penalty_time": coalesce_numeric_with_alt(
            item,
            ["penaltyTime", "penalty_time", "penaltyMinutes", "pim"],
            [["penalty", "time"], ["penalty", "minute"], ["pim"]],
            allow_zero=True,
        ),
        "touches": coalesce_numeric_with_alt(
            item,
            ["touches", "touch"],
            [["touch"]],
            allow_zero=True,
        ),
        "total_passes": coalesce_numeric_with_alt(
            item,
            ["totalPasses", "passes", "total_passes"],
            [["pass"]],
            allow_zero=True,
        ),
        "source_path": source_path,
        "raw_stat_keys": "|".join(sorted(map(str, item.keys()))),
    }

    return row


def parse_team_stat_item(
    item: dict[str, Any],
    schedule_row: pd.Series,
    source_path: str,
) -> dict[str, Any]:
    team_id_raw, team_id, team_name = extract_stat_team_id_and_name(item)

    opponent_team_id, opponent_team_name = infer_opponent_for_team(
        team_id,
        schedule_row.get("home_team_id"),
        schedule_row.get("away_team_id"),
        schedule_row.get("home_team_name"),
        schedule_row.get("away_team_name"),
    )

    raw_one_point_goals = coalesce_numeric_with_alt(
        item,
        [
            "onePointGoals",
            "one_point_goals",
            "onePtGoals",
            "one_pt_goals",
            "goalsOnePoint",
            "goals1pt",
        ],
        [["one", "point", "goal"], ["1", "pt", "goal"]],
        allow_zero=True,
    )

    two_point_goals = coalesce_numeric_with_alt(
        item,
        [
            "twoPointGoals",
            "two_point_goals",
            "twoPtGoals",
            "two_pt_goals",
            "goalsTwoPoint",
            "goals2pt",
        ],
        [["two", "point", "goal"], ["2", "pt", "goal"]],
        allow_zero=True,
    )

    goals = coalesce_numeric_with_alt(
        item,
        ["goals", "goal", "g"],
        [["goal"]],
        allow_zero=True,
    )

    one_point_goals = derive_one_point_goals(
        goals,
        raw_one_point_goals,
        two_point_goals,
    )

    scoring_points = derive_scoring_points(one_point_goals, two_point_goals)

    scores = coalesce_numeric_with_alt(
        item,
        ["scores", "score", "totalScore", "points"],
        [["score"]],
        allow_zero=True,
    )

    if pd.isna(scores):
        scores = scoring_points

    faceoffs_won = coalesce_numeric_with_alt(
        item,
        [
            "faceoffsWon",
            "faceoffWins",
            "faceoffs_won",
            "foWins",
            "fo_wins",
        ],
        [["faceoff", "won"], ["faceoff", "win"], ["fo", "win"]],
        allow_zero=True,
    )

    faceoffs_lost = coalesce_numeric_with_alt(
        item,
        [
            "faceoffsLost",
            "faceoffLosses",
            "faceoffs_lost",
            "foLosses",
            "fo_losses",
        ],
        [["faceoff", "lost"], ["faceoff", "loss"], ["fo", "loss"]],
        allow_zero=True,
    )

    faceoffs = coalesce_numeric_with_alt(
        item,
        ["faceoffs", "faceoffAttempts", "foAttempts", "fo"],
        [["faceoff"]],
        allow_zero=True,
    )

    if pd.isna(faceoffs) and not pd.isna(faceoffs_won) and not pd.isna(faceoffs_lost):
        faceoffs = faceoffs_won + faceoffs_lost

    total_clears = coalesce_numeric_with_alt(
        item,
        ["totalClears", "clears", "total_clears", "clearSuccesses"],
        [["clear"]],
        allow_zero=True,
    )

    failed_clears = coalesce_numeric_with_alt(
        item,
        ["failedClears", "failed_clears", "clearFailures"],
        [["failed", "clear"]],
        allow_zero=True,
    )

    clear_attempts = coalesce_numeric_with_alt(
        item,
        ["clearAttempts", "clear_attempts"],
        [["clear", "attempt"]],
        allow_zero=True,
    )

    if pd.isna(clear_attempts) and not pd.isna(total_clears) and not pd.isna(failed_clears):
        clear_attempts = total_clears + failed_clears

    time_in_possession = coalesce_numeric_with_alt(
        item,
        [
            "timeInPossession",
            "time_in_possession",
            "possessionTime",
            "possession_time",
        ],
        [["time", "possession"], ["possession", "time"]],
        allow_zero=True,
    )

    time_in_possession_pct = coalesce_numeric_with_alt(
        item,
        [
            "timeInPossessionPct",
            "time_in_possession_pct",
            "possessionPct",
            "possession_pct",
        ],
        [["possession", "pct"], ["possession", "percent"]],
        allow_zero=True,
    )

    total_possessions = coalesce_numeric_with_alt(
        item,
        [
            "totalPossessions",
            "total_possessions",
            "possessions",
        ],
        [["possession"]],
        allow_zero=True,
    )

    official_total_possessions = coalesce_numeric_with_alt(
        item,
        [
            "officialTotalPossessions",
            "official_total_possessions",
        ],
        [["official", "possession"]],
        allow_zero=True,
    )

    offensive_sequence_proxy = coalesce_numeric_with_alt(
        item,
        [
            "offensiveSequences",
            "offensiveSequenceProxy",
            "offensive_sequence_proxy",
            "sequences",
        ],
        [["offensive", "sequence"], ["sequence"]],
        allow_zero=True,
    )

    if pd.isna(offensive_sequence_proxy):
        # Fallback proxy from touches/pass volume when provider sequence field is unavailable.
        touches = coalesce_numeric_with_alt(
            item,
            ["touches", "touch"],
            [["touch"]],
            allow_zero=True,
        )

        total_passes = coalesce_numeric_with_alt(
            item,
            ["totalPasses", "passes", "total_passes"],
            [["pass"]],
            allow_zero=True,
        )

        turnovers = coalesce_numeric_with_alt(
            item,
            ["turnovers", "turnover"],
            [["turnover"]],
            allow_zero=True,
        )

        shots = coalesce_numeric_with_alt(
            item,
            ["shots", "shot"],
            [["shot"]],
            allow_zero=True,
        )

        offensive_sequence_proxy = np.nan

        if not pd.isna(shots) or not pd.isna(turnovers):
            offensive_sequence_proxy = (
                (0 if pd.isna(shots) else shots)
                + (0 if pd.isna(turnovers) else turnovers)
            )

        if pd.isna(offensive_sequence_proxy) and not pd.isna(touches):
            offensive_sequence_proxy = touches / 5

        if pd.isna(offensive_sequence_proxy) and not pd.isna(total_passes):
            offensive_sequence_proxy = total_passes / 4

    row = {
        "season": schedule_row.get("season"),
        "game_id": schedule_row.get("game_id"),
        "game_slug": schedule_row.get("game_slug"),
        "schedule_slug": schedule_row.get("schedule_slug"),
        "game_number": schedule_row.get("game_number"),
        "game_date_utc": schedule_row.get("game_date_utc"),
        "event_status": schedule_row.get("event_status"),
        "team_id_raw": team_id_raw,
        "team_id": team_id,
        "team_name": team_name,
        "opponent_team_id": opponent_team_id,
        "opponent_team_name": opponent_team_name,
        "is_home": (
            str(team_id) == str(schedule_row.get("home_team_id"))
            if not pd.isna(team_id) and not pd.isna(schedule_row.get("home_team_id"))
            else pd.NA
        ),
        "scores": scores,
        "scoring_points": scoring_points,
        "one_point_goals": one_point_goals,
        "two_point_goals": two_point_goals,
        "goals": goals,
        "assists": coalesce_numeric_with_alt(
            item,
            ["assists", "assist", "a"],
            [["assist"]],
            allow_zero=True,
        ),
        "shots": coalesce_numeric_with_alt(
            item,
            ["shots", "shot", "sh"],
            [["shot"]],
            allow_zero=True,
        ),
        "shots_on_goal": coalesce_numeric_with_alt(
            item,
            ["shotsOnGoal", "shots_on_goal", "sog"],
            [["shot", "goal"], ["sog"]],
            allow_zero=True,
        ),
        "two_point_shots": coalesce_numeric_with_alt(
            item,
            ["twoPointShots", "two_point_shots", "twoPtShots", "two_pt_shots"],
            [["two", "point", "shot"], ["2", "pt", "shot"]],
            allow_zero=True,
        ),
        "ground_balls": coalesce_numeric_with_alt(
            item,
            ["groundBalls", "ground_balls", "gb"],
            [["ground", "ball"], ["gb"]],
            allow_zero=True,
        ),
        "turnovers": coalesce_numeric_with_alt(
            item,
            ["turnovers", "turnover", "to"],
            [["turnover"]],
            allow_zero=True,
        ),
        "caused_turnovers": coalesce_numeric_with_alt(
            item,
            ["causedTurnovers", "caused_turnovers", "ct"],
            [["caused", "turnover"], ["ct"]],
            allow_zero=True,
        ),
        "saves": coalesce_numeric_with_alt(
            item,
            ["saves", "save", "sv"],
            [["save"]],
            allow_zero=True,
        ),
        "faceoffs_won": faceoffs_won,
        "faceoffs_lost": faceoffs_lost,
        "faceoffs": faceoffs,
        "total_clears": total_clears,
        "failed_clears": failed_clears,
        "clear_attempts": clear_attempts,
        "touches": coalesce_numeric_with_alt(
            item,
            ["touches", "touch"],
            [["touch"]],
            allow_zero=True,
        ),
        "total_passes": coalesce_numeric_with_alt(
            item,
            ["totalPasses", "passes", "total_passes"],
            [["pass"]],
            allow_zero=True,
        ),
        "time_in_possession": time_in_possession,
        "time_in_possession_pct": time_in_possession_pct,
        "total_possessions": total_possessions,
        "official_total_possessions": official_total_possessions,
        "offensive_sequence_proxy": offensive_sequence_proxy,
        "source_path": source_path,
        "raw_stat_keys": "|".join(sorted(map(str, item.keys()))),
    }

    return row


def fill_team_opponent_stats_from_pair(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills *_against fields by matching the opposite team row in the same game.

    This avoids relying on stale winner/provider fields and keeps defensive
    context score-based.
    """

    if team_df is None or len(team_df) == 0:
        return team_df

    out = team_df.copy()

    base_cols = [
        "scores",
        "scoring_points",
        "one_point_goals",
        "two_point_goals",
        "goals",
        "assists",
        "shots",
        "shots_on_goal",
        "ground_balls",
        "turnovers",
        "caused_turnovers",
        "saves",
        "faceoffs_won",
        "faceoffs_lost",
        "faceoffs",
        "total_clears",
        "failed_clears",
        "clear_attempts",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
        "official_total_possessions",
        "offensive_sequence_proxy",
    ]

    available_cols = [c for c in base_cols if c in out.columns]

    opp = out[[
        "season",
        "game_id",
        "team_id",
        "team_name",
    ] + available_cols].copy()

    rename = {
        "team_id": "opponent_team_id_join",
        "team_name": "opponent_team_name_join",
    }

    for c in available_cols:
        rename[c] = f"{c}_against"

    opp = opp.rename(columns=rename)

    out = out.merge(
        opp,
        left_on=["season", "game_id", "opponent_team_id"],
        right_on=["season", "game_id", "opponent_team_id_join"],
        how="left",
    )

    if "opponent_team_name_join" in out.columns:
        out["opponent_team_name"] = out["opponent_team_name"].fillna(out["opponent_team_name_join"])

    out = out.drop(columns=[
        c for c in [
            "opponent_team_id_join",
            "opponent_team_name_join",
        ]
        if c in out.columns
    ])

    if "scores" in out.columns and "scores_against" in out.columns:
        out["score_based_win"] = (
            pd.to_numeric(out["scores"], errors="coerce")
            > pd.to_numeric(out["scores_against"], errors="coerce")
        ).astype("Int64")

        out["score_based_loss"] = (
            pd.to_numeric(out["scores"], errors="coerce")
            < pd.to_numeric(out["scores_against"], errors="coerce")
        ).astype("Int64")

    return out


def collect_game_stats(game_schedule_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Downloads/caches player and team stat payloads, then parses them into
    row-level game tables.
    """

    require_api_token()

    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    api_log_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    if game_schedule_all is None or len(game_schedule_all) == 0:
        return {
            "player_game_stats": pd.DataFrame(),
            "team_game_stats": pd.DataFrame(),
            "api_collection_log": pd.DataFrame(),
            "skipped_games": pd.DataFrame([{
                "season": pd.NA,
                "game_slug": pd.NA,
                "reason": "empty_game_schedule",
                "created_at_utc": now_utc_iso(),
            }]),
        }

    schedule_iter = game_schedule_all.copy()

    if "game_slug" not in schedule_iter.columns:
        raise ValueError("game_schedule_all is missing game_slug.")

    schedule_iter = schedule_iter[
        schedule_iter["game_slug"].notna()
        & schedule_iter["game_slug"].astype(str).str.strip().ne("")
    ].copy()

    for _, sched in tqdm(
        schedule_iter.iterrows(),
        total=len(schedule_iter),
        desc="Collecting PLL game stats",
    ):
        season = int(sched["season"])
        slug = str(sched["game_slug"]).strip()

        game_dir = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}"
        game_dir.mkdir(parents=True, exist_ok=True)

        player_cache = game_dir / "player_game_stats.json.gz"
        team_cache = game_dir / "team_game_stats.json.gz"

        player_url = player_game_stats_url(slug)
        team_url = team_game_stats_url(slug)

        player_payload, player_status, player_cache_status = fetch_json_with_cache(
            player_url,
            player_cache,
            force=FORCE_RECOLLECT,
        )

        team_payload, team_status, team_cache_status = fetch_json_with_cache(
            team_url,
            team_cache,
            force=FORCE_RECOLLECT,
        )

        player_items = extract_stat_items(player_payload, "player")
        team_items = extract_stat_items(team_payload, "team")

        api_log_rows.append({
            "season": season,
            "game_id": sched.get("game_id"),
            "game_slug": slug,
            "player_stats_url": player_url,
            "player_status_code": player_status,
            "player_cache_status": player_cache_status,
            "player_rows_detected": len(player_items),
            "team_stats_url": team_url,
            "team_status_code": team_status,
            "team_cache_status": team_cache_status,
            "team_rows_detected": len(team_items),
            "checked_at_utc": now_utc_iso(),
        })

        if player_status not in [200, 304] or team_status not in [200, 304]:
            skipped_rows.append({
                "season": season,
                "game_id": sched.get("game_id"),
                "game_slug": slug,
                "reason": "non_200_stats_response",
                "player_status_code": player_status,
                "team_status_code": team_status,
                "created_at_utc": now_utc_iso(),
            })

        if len(player_items) == 0 and len(team_items) == 0:
            skipped_rows.append({
                "season": season,
                "game_id": sched.get("game_id"),
                "game_slug": slug,
                "reason": "no_stat_rows_detected",
                "player_status_code": player_status,
                "team_status_code": team_status,
                "created_at_utc": now_utc_iso(),
            })

        source_path_player = str(player_cache)
        source_path_team = str(team_cache)

        for item in player_items:
            try:
                player_rows.append(
                    parse_player_stat_item(
                        item,
                        schedule_row=sched,
                        source_path=source_path_player,
                    )
                )
            except Exception as exc:
                skipped_rows.append({
                    "season": season,
                    "game_id": sched.get("game_id"),
                    "game_slug": slug,
                    "reason": "player_parse_error",
                    "error": str(exc),
                    "created_at_utc": now_utc_iso(),
                })

        for item in team_items:
            try:
                team_rows.append(
                    parse_team_stat_item(
                        item,
                        schedule_row=sched,
                        source_path=source_path_team,
                    )
                )
            except Exception as exc:
                skipped_rows.append({
                    "season": season,
                    "game_id": sched.get("game_id"),
                    "game_slug": slug,
                    "reason": "team_parse_error",
                    "error": str(exc),
                    "created_at_utc": now_utc_iso(),
                })

        time.sleep(0.03)

    player_game_stats = pd.DataFrame(player_rows)
    team_game_stats = pd.DataFrame(team_rows)
    api_collection_log = pd.DataFrame(api_log_rows)
    skipped_games = pd.DataFrame(skipped_rows)

    # Clean numeric columns.
    player_numeric_cols = [
        "season",
        "game_number",
        "points",
        "scoring_points",
        "one_point_goals",
        "two_point_goals",
        "goals",
        "assists",
        "shots",
        "shots_on_goal",
        "two_point_shots",
        "ground_balls",
        "turnovers",
        "caused_turnovers",
        "faceoffs_won",
        "faceoffs_lost",
        "faceoffs",
        "saves",
        "clean_saves",
        "messy_saves",
        "scores_against",
        "goals_against",
        "penalties",
        "penalty_time",
        "touches",
        "total_passes",
    ]

    team_numeric_cols = [
        "season",
        "game_number",
        "scores",
        "scoring_points",
        "one_point_goals",
        "two_point_goals",
        "goals",
        "assists",
        "shots",
        "shots_on_goal",
        "two_point_shots",
        "ground_balls",
        "turnovers",
        "caused_turnovers",
        "saves",
        "faceoffs_won",
        "faceoffs_lost",
        "faceoffs",
        "total_clears",
        "failed_clears",
        "clear_attempts",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
        "official_total_possessions",
        "offensive_sequence_proxy",
    ]

    player_game_stats = coerce_numeric(player_game_stats, player_numeric_cols)
    team_game_stats = coerce_numeric(team_game_stats, team_numeric_cols)

    if len(team_game_stats) > 0:
        team_game_stats = fill_team_opponent_stats_from_pair(team_game_stats)

    if len(player_game_stats) > 0:
        player_game_stats = add_standard_player_rates(player_game_stats)

    if len(team_game_stats) > 0:
        team_game_stats = add_standard_team_rates(team_game_stats)

    # Dedupe defensively.
    if len(player_game_stats) > 0:
        dedupe_cols = [
            c for c in [
                "season",
                "game_id",
                "player_id",
                "team_id",
            ]
            if c in player_game_stats.columns
        ]

        if dedupe_cols:
            before = len(player_game_stats)
            player_game_stats = player_game_stats.drop_duplicates(dedupe_cols, keep="first")
            after = len(player_game_stats)

            add_qc_check(
                "duplicate_player_game_rows_removed",
                "pass",
                before - after,
                0,
                "Duplicate player-game rows removed by key.",
            )

    if len(team_game_stats) > 0:
        dedupe_cols = [
            c for c in [
                "season",
                "game_id",
                "team_id",
            ]
            if c in team_game_stats.columns
        ]

        if dedupe_cols:
            before = len(team_game_stats)
            team_game_stats = team_game_stats.drop_duplicates(dedupe_cols, keep="first")
            after = len(team_game_stats)

            add_qc_check(
                "duplicate_team_game_rows_removed",
                "pass",
                before - after,
                0,
                "Duplicate team-game rows removed by key.",
            )

    add_qc_check(
        "player_game_stats_rows",
        "pass" if len(player_game_stats) > 0 else "warn",
        len(player_game_stats),
        None,
        "Parsed player-game stat rows.",
    )

    add_qc_check(
        "team_game_stats_rows",
        "pass" if len(team_game_stats) > 0 else "warn",
        len(team_game_stats),
        None,
        "Parsed team-game stat rows.",
    )

    return {
        "player_game_stats": player_game_stats.reset_index(drop=True),
        "team_game_stats": team_game_stats.reset_index(drop=True),
        "api_collection_log": api_collection_log.reset_index(drop=True),
        "skipped_games": skipped_games.reset_index(drop=True),
    }


# ============================================================
# SECTION 2C COMPLETE
# ============================================================
