# ============================================================
# PLL DATA PLATFORM — STREAMLIT APP
# GitHub / Streamlit Cloud production version
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from html import escape

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(os.environ.get("PLL_PROJECT_ROOT", "data"))
DB_PATH = PROJECT_ROOT / "analytics_database" / "pll_warehouse.duckdb"
CURATED_DIR = PROJECT_ROOT / "curated_data" / "all_requested_seasons"

st.set_page_config(
    page_title="PLL Data Platform",
    page_icon="🥍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1720px;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .section-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 0.75rem;
        line-height: 1.45;
    }

    .stat-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        box-shadow: 0 8px 20px rgba(0,0,0,0.11);
        min-height: 92px;
        margin-bottom: 10px;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        margin-bottom: 6px;
    }

    .stat-value {
        font-size: 1.55rem;
        font-weight: 850;
        line-height: 1.15;
        color: #f8fafc;
    }

    .stat-sub {
        color: #cbd5e1;
        font-size: 0.80rem;
        margin-top: 5px;
    }

    .profile-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(30,41,59,0.88), rgba(15,23,42,0.75));
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        margin-bottom: 14px;
    }

    .profile-title {
        font-size: 1.55rem;
        font-weight: 850;
        margin-bottom: 4px;
    }

    .profile-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .mini-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(15, 23, 42, 0.54);
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
        margin-bottom: 10px;
        min-height: 126px;
    }

    .mini-title {
        font-size: 1.04rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .mini-line {
        color: #cbd5e1;
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .mini-label {
        color: #94a3b8;
        font-weight: 700;
    }

    .note-box {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(15, 23, 42, 0.52);
        margin-bottom: 14px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY LABELS
# ============================================================

COL_LABELS = {
    "season": "Season",
    "game_id": "Game ID",
    "game_number": "Game",
    "game_date_utc": "Date",
    "game_slug": "Game Slug",
    "event_status": "Status",
    "status_display": "Status",
    "matchup": "Matchup",
    "result": "Result",
    "team_id": "Team ID",
    "team_name": "Team",
    "team_display_name": "Team",
    "opponent_team_id": "Opponent ID",
    "opponent_team_name": "Opponent",
    "away_team_name": "Away",
    "home_team_name": "Home",
    "away_score": "Away Score",
    "home_score": "Home Score",
    "is_home": "Home/Away",
    "player_id": "Player ID",
    "full_name": "Player",
    "first_name": "First Name",
    "last_name": "Last Name",
    "position": "Pos",
    "position_name": "Position",
    "latest_team_name": "Latest Team",
    "games": "Games",
    "wins": "Wins",
    "losses": "Losses",
    "win_pct": "Win %",
    "score_margin": "Score Margin",
    "score_margin_per_game": "Score Margin/G",
    "points": "Points",
    "points_per_game": "Points/G",
    "scoring_points": "Scoring Pts",
    "scoring_points_per_game": "Scoring Pts/G",
    "scores": "Scores",
    "scores_per_game": "Scores/G",
    "scores_against": "Scores Against",
    "scores_against_per_game": "Scores Against/G",
    "goals": "Goals",
    "goals_per_game": "Goals/G",
    "goals_against": "Goals Against",
    "goals_against_per_game": "Goals Against/G",
    "one_point_goals": "1PT Goals",
    "one_point_goals_per_game": "1PT Goals/G",
    "two_point_goals": "2PT Goals",
    "two_point_goals_per_game": "2PT Goals/G",
    "assists": "Assists",
    "assists_per_game": "Assists/G",
    "shots": "Shots",
    "shots_per_game": "Shots/G",
    "shots_against": "Shots Against",
    "opponent_shots_per_game": "Opp. Shots/G",
    "def_opponent_shots_per_game": "Opp. Shots/G",
    "shots_on_goal": "SOG",
    "shots_on_goal_per_game": "SOG/G",
    "shots_on_goal_against": "SOG Against",
    "two_point_shots": "2PT Shots",
    "shot_pct_calc": "Shot %",
    "shot_pct": "Shot %",
    "shots_on_goal_rate_calc": "SOG Rate",
    "ground_balls": "GB",
    "ground_balls_per_game": "GB/G",
    "turnovers": "TO",
    "turnovers_per_game": "TO/G",
    "caused_turnovers": "CT",
    "caused_turnovers_per_game": "CT/G",
    "saves": "Saves",
    "saves_per_game": "Saves/G",
    "save_pct_calc": "Save %",
    "save_pct": "Save %",
    "faceoffs": "FO",
    "faceoffs_per_game": "FO/G",
    "faceoffs_won": "FO Won",
    "faceoffs_won_per_game": "FO Won/G",
    "faceoffs_lost": "FO Lost",
    "faceoffs_lost_per_game": "FO Lost/G",
    "faceoff_pct_calc": "FO %",
    "faceoff_pct": "FO %",
    "touches": "Touches",
    "touches_per_game": "Touches/G",
    "total_passes": "Passes",
    "total_passes_per_game": "Passes/G",
    "time_in_possession": "Poss. Time",
    "time_in_possession_per_game": "Poss. Time/G",
    "time_in_possession_per_game_mmss": "Poss. Time/G",
    "possession_pg": "Possession/G",
    "total_possessions": "Possessions",
    "official_total_possessions": "Official Poss.",
    "offensive_sequence_proxy": "Off. Seq.",
    "offensive_sequence_proxy_per_game": "Off. Seq./G",
    "clear_pct_calc": "Clear %",
    "total_clears": "Clears",
    "clear_attempts": "Clear Att.",
    "ranking_context": "Context",
    "ranking_context_type": "Context Type",
    "v22_overall_rank": "Rank",
    "overall_rank": "Rank",
    "v22_overall_score": "Overall Score",
    "overall_score": "Overall Score",
    "overall_impact_score": "Overall Score",
    "v22_overall_percentile": "Overall Percentile",
    "overall_percentile": "Overall Percentile",
    "v22_position_rank": "Position Rank",
    "position_rank": "Position Rank",
    "v22_position_percentile": "Position Percentile",
    "position_percentile": "Position Percentile",
    "role_group": "Role",
    "role_primary_score": "Role Score",
    "role_context_value_score": "Role Context Value",
    "role_separation_score": "Peer Separation Score",
    "role_adjusted_z": "Peer Separation Z",
    "role_value_tier": "Role Tier",
    "offensive_score": "Offensive Score",
    "usage_score": "Usage Score",
    "usage_possession_score": "Usage Score",
    "defensive_score": "Defensive Score",
    "faceoff_score": "Faceoff Score",
    "goalie_score": "Goalie Score",
    "goal_value_score": "Scoring Value",
    "profile_context": "Context",
    "profile_context_type": "Context Type",
    "profile_rank": "Rank",
    "team_style_overall_score": "Overall Style Score",
    "offensive_volume_score": "Offensive Volume",
    "offensive_efficiency_score": "Offensive Efficiency",
    "ball_movement_score": "Ball Movement",
    "possession_control_score": "Possession Control",
    "defensive_suppression_score": "Defensive Suppression",
    "pace_tempo_score": "Pace / Tempo",
    "style_summary": "Style Summary",
    "pace_label": "Pace",
    "offensive_profile_label": "Offense",
    "defensive_profile_label": "Defense",
    "possession_profile_label": "Possession",
    "check_name": "Check",
    "status": "Status",
    "actual": "Actual",
    "expected": "Expected",
    "notes": "Notes",
    "rows": "Rows",
    "columns": "Columns",
    "table_name": "Table",
    "table_schema": "Schema",
}

DEFAULT_HIDE_COLS = {
    "player_id",
    "team_id",
    "opponent_team_id",
    "game_id",
    "event_id",
    "event_numeric_id",
    "schedule_slug",
    "game_slug",
    "source_path",
    "raw_stat_keys",
    "normalized_name",
    "first_name",
    "last_name",
    "team_id_raw",
    "home_team_id_raw",
    "away_team_id_raw",
    "opponent_team_id_join",
    "opponent_team_name_join",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def pretty_col(col: str) -> str:
    return COL_LABELS.get(str(col), str(col).replace("_", " ").title())


def make_unique_columns(cols):
    seen = {}
    out = []

    for col in cols:
        base = str(col)

        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base} {seen[base] + 1}")

    return out


def fmt_value(x, digits: int = 2, pct: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"

    try:
        v = float(x)
    except Exception:
        return str(x)

    if pct:
        return f"{v:.1%}"

    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"

    return f"{v:,.{digits}f}"


def seconds_to_mmss(value) -> str:
    try:
        seconds = int(round(float(value)))
    except Exception:
        return "—"

    if seconds < 0:
        return "—"

    return f"{seconds // 60}:{seconds % 60:02d}"


def stat_card(label, value, sub=None):
    sub_html = f'<div class="stat-sub">{escape(str(sub))}</div>' if sub else ""

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{escape(str(label))}</div>
            <div class="stat-value">{escape(str(value))}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def profile_header(title, subtitle):
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">{escape(str(title))}</div>
            <div class="profile-subtitle">{escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_box(title, body):
    st.markdown(
        f"""
        <div class="note-box">
            <div class="mini-title">{escape(str(title))}</div>
            <div class="mini-line">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_table(df, height=420, hide_cols=None, max_cols=None):
    if df is None or len(df) == 0:
        st.info("No rows available for the selected filters.")
        return

    out = df.copy().reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    hide = set(DEFAULT_HIDE_COLS)
    if hide_cols:
        hide.update(hide_cols)

    keep_cols = [c for c in out.columns if c not in hide]
    out = out[keep_cols]

    if max_cols is not None and len(out.columns) > max_cols:
        out = out.iloc[:, :max_cols]

    for col in out.columns:
        if "date" in str(col).lower():
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")

    for col in out.columns:
        if str(col).lower() == "season":
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.astype("Int64").astype(str).replace("<NA>", "")

    if "is_home" in out.columns:
        out["is_home"] = out["is_home"].map({1: "Home", 0: "Away", True: "Home", False: "Away"}).fillna(out["is_home"])

    out.columns = make_unique_columns([pretty_col(c) for c in out.columns])

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    try:
        fmt_map = {
            c: "{:,.2f}"
            for c in numeric_cols
        }

        st.dataframe(
            out.style.format(fmt_map, na_rep=""),
            use_container_width=True,
            hide_index=True,
            height=height,
        )
    except Exception:
        st.dataframe(out, use_container_width=True, hide_index=True, height=height)


def download_csv(df, filename, label="Download CSV"):
    if df is None or len(df) == 0:
        return

    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def first_existing(df, candidates, default=None):
    for c in candidates:
        if df is not None and c in df.columns:
            return c
    return default


def safe_sort(df, by, ascending=False):
    if df is None or len(df) == 0 or by not in df.columns:
        return df

    out = df.copy()
    out[by] = pd.to_numeric(out[by], errors="coerce")

    return out.sort_values(by, ascending=ascending, na_position="last")


def apply_min_games(df, min_games_value):
    if df is None or len(df) == 0 or "games" not in df.columns:
        return df

    out = df.copy()
    return out[pd.to_numeric(out["games"], errors="coerce").fillna(0) >= min_games_value].copy()


def filter_positions(df, positions):
    if not positions or df is None or len(df) == 0 or "position" not in df.columns:
        return df

    return df[df["position"].astype(str).isin(positions)].copy()


def filter_seasons(df, seasons):
    if not seasons or df is None or len(df) == 0 or "season" not in df.columns:
        return df

    return df[pd.to_numeric(df["season"], errors="coerce").isin(seasons)].copy()


def safe_bar_chart(df, x_col, y_col, title, color_col=None, orientation="v"):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        st.info("Required chart columns are not available.")
        return

    chart_df = df.copy()

    if str(x_col).lower() == "season":
        chart_df[x_col] = pd.to_numeric(chart_df[x_col], errors="coerce").astype("Int64").astype(str).replace("<NA>", "")

    labels = {c: pretty_col(c) for c in chart_df.columns}

    if orientation == "h":
        fig = px.bar(
            chart_df,
            x=y_col,
            y=x_col,
            color=color_col if color_col in chart_df.columns else None,
            orientation="h",
            text=y_col,
            title=title,
            labels=labels,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in chart_df.columns else None,
            text=y_col,
            title=title,
            labels=labels,
        )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=35))
    st.plotly_chart(fig, use_container_width=True)


def safe_line_chart(df, x_col, y_cols, title, color_col=None):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return

    y_cols = [c for c in y_cols if c in df.columns]

    if x_col not in df.columns or not y_cols:
        st.info("Required chart columns are not available.")
        return

    chart_df = df.copy()

    if str(x_col).lower() == "season":
        chart_df[x_col] = pd.to_numeric(chart_df[x_col], errors="coerce").astype("Int64").astype(str).replace("<NA>", "")

    fig = px.line(
        chart_df,
        x=x_col,
        y=y_cols,
        color=color_col if color_col in chart_df.columns else None,
        markers=True,
        title=title,
        labels={c: pretty_col(c) for c in chart_df.columns},
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=35), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def safe_scatter(df, x_col, y_col, title, size_col=None, color_col=None):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        st.info("Required chart columns are not available.")
        return

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col if size_col in df.columns else None,
        color=color_col if color_col in df.columns else None,
        hover_name="full_name" if "full_name" in df.columns else None,
        title=title,
        labels={c: pretty_col(c) for c in df.columns},
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=35))
    st.plotly_chart(fig, use_container_width=True)


def comparison_matrix(df, entity_col, metrics):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        row = {"Metric": pretty_col(metric)}

        for _, r in df.iterrows():
            entity = str(r.get(entity_col, "Unknown"))
            row[entity] = fmt_value(r.get(metric, np.nan))

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data(ttl=600, show_spinner=False)
def query_df(sql, params=None):
    con = get_connection()

    try:
        if params is None:
            return con.execute(sql).df()

        return con.execute(sql, params).df()
    finally:
        con.close()


@st.cache_data(ttl=600, show_spinner=False)
def table_exists(schema_name, table_name):
    try:
        df = query_df(
            """
            SELECT COUNT(*) AS n
            FROM information_schema.tables
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [schema_name, table_name],
        )
        return int(df["n"].iloc[0]) > 0
    except Exception:
        return False


@st.cache_data(ttl=600, show_spinner=False)
def read_table(table_path):
    schema, table = table_path.split(".", 1)

    if not table_exists(schema, table):
        return pd.DataFrame()

    return query_df(f"SELECT * FROM {schema}.{table}")


@st.cache_data(ttl=600, show_spinner=False)
def table_index():
    return query_df(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('clean', 'marts', 'qc')
        ORDER BY table_schema, table_name
        """
    )


@st.cache_data(ttl=600, show_spinner=False)
def startup_counts():
    def safe_count(table_path):
        schema, table = table_path.split(".", 1)
        if not table_exists(schema, table):
            return 0
        return int(query_df(f"SELECT COUNT(*) AS n FROM {table_path}")["n"].iloc[0])

    return {
        "completed_games": safe_count("clean.game_manifest"),
        "player_game_rows": safe_count("clean.player_game_stats"),
        "team_game_rows": safe_count("clean.team_game_stats"),
        "players": safe_count("clean.player_directory"),
        "teams": safe_count("clean.team_directory"),
    }


@st.cache_data(ttl=600, show_spinner=False)
def filter_values():
    schedule = read_table("clean.game_schedule_all")
    teams = read_table("clean.team_directory")
    players = read_table("clean.player_directory")
    player_games = read_table("clean.player_game_stats")

    seasons = []

    if len(schedule) > 0 and "season" in schedule.columns:
        seasons = (
            pd.to_numeric(schedule["season"], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

    if len(teams) > 0 and "team_name" in teams.columns:
        teams = teams.sort_values("team_name").reset_index(drop=True)

    if len(players) > 0 and "full_name" in players.columns:
        players = players.sort_values("full_name").reset_index(drop=True)

    if len(player_games) > 0 and "position" in player_games.columns:
        positions = sorted(player_games["position"].dropna().astype(str).unique().tolist())
    else:
        positions = []

    return seasons, teams, players, positions


def schedule_display_table():
    schedule = read_table("clean.game_schedule_all")
    manifest = read_table("clean.game_manifest")

    if len(schedule) == 0:
        return schedule

    out = schedule.copy()

    if "status_display" not in out.columns:
        out["status_display"] = out.get("event_status", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)

    if len(manifest) > 0 and "game_id" in manifest.columns and "game_id" in out.columns:
        completed_ids = set(manifest["game_id"].dropna().astype(str))
        out.loc[out["game_id"].astype(str).isin(completed_ids), "status_display"] = "final"

    if "matchup" not in out.columns and "away_team_name" in out.columns and "home_team_name" in out.columns:
        out["matchup"] = out["away_team_name"].astype(str) + " at " + out["home_team_name"].astype(str)

    if "result" not in out.columns and "away_score" in out.columns and "home_score" in out.columns:
        out["result"] = np.where(
            out["away_score"].notna() & out["home_score"].notna(),
            out["away_score"].astype("Int64").astype(str) + " - " + out["home_score"].astype("Int64").astype(str),
            "",
        )

    return out


# ============================================================
# STARTUP
# ============================================================

if not DB_PATH.exists():
    st.error("DuckDB warehouse not found.")
    st.markdown(
        f"""
        Expected path:

        ```text
        {DB_PATH}
        ```

        Run the GitHub Action **Update PLL Data Warehouse** once before deploying or opening the app.
        """
    )
    st.stop()

try:
    counts = startup_counts()
    seasons, teams_df, players_df, positions = filter_values()
except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PLL Data Platform")
st.sidebar.caption("Production dashboard")

st.sidebar.divider()

selected_seasons = st.sidebar.multiselect(
    "Global Season Filter",
    options=seasons,
    default=seasons,
)

team_options = teams_df["team_name"].dropna().tolist() if len(teams_df) and "team_name" in teams_df.columns else []

selected_teams = st.sidebar.multiselect(
    "Global Team Filter",
    options=team_options,
    default=[],
)

selected_positions = st.sidebar.multiselect(
    "Global Position Filter",
    options=positions,
    default=[],
)

min_games = st.sidebar.number_input(
    "Minimum Games",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

st.sidebar.caption("Global filters primarily affect Overview and Leaderboards. Explorer pages have their own filters.")


# ============================================================
# MAIN TABS
# ============================================================

st.title("PLL Data Platform")
st.caption("Player, team, season, matchup, specialty, ranking, schedule, and data-quality dashboard.")

tabs = st.tabs(
    [
        "Overview",
        "Season Page",
        "Matchup Preview / Review",
        "Player Explorer",
        "Team Explorer",
        "Player Rankings",
        "Team Profiles",
        "Goalie / Faceoff",
        "Player Comparison",
        "Team Comparison",
        "Leaderboards",
        "Schedule",
        "Data Guide",
        "Data Quality",
    ]
)

(
    tab_overview,
    tab_season,
    tab_matchup,
    tab_players,
    tab_teams,
    tab_player_rankings,
    tab_team_profiles,
    tab_specialists,
    tab_player_compare,
    tab_team_compare,
    tab_leaders,
    tab_schedule,
    tab_guide,
    tab_quality,
) = tabs


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("Overview")
    st.markdown('<div class="section-note">High-level warehouse status and leaguewide summary views.</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    overview_cards = [
        ("Completed Games", counts["completed_games"]),
        ("Player-Game Rows", counts["player_game_rows"]),
        ("Team-Game Rows", counts["team_game_rows"]),
        ("Players", counts["players"]),
        ("Teams", counts["teams"]),
    ]

    for i, (label, value) in enumerate(overview_cards):
        with cols[i]:
            stat_card(label, fmt_value(value, digits=0))

    schedule = schedule_display_table()
    player_seasons = read_table("marts.player_season_stats")
    team_seasons = read_table("marts.team_season_stats")

    if len(schedule) > 0:
        season_counts = (
            schedule[schedule["status_display"].astype(str).str.lower().eq("final")]
            .groupby("season", dropna=False)
            .size()
            .reset_index(name="completed_stat_games")
        )

        status_counts = (
            schedule.groupby(["season", "status_display"], dropna=False)
            .size()
            .reset_index(name="games")
            .sort_values(["season", "status_display"])
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Completed Games by Season")
            safe_bar_chart(season_counts, "season", "completed_stat_games", "Completed / Stat-Available Games")

        with c2:
            st.markdown("### Schedule Status")
            safe_bar_chart(status_counts, "season", "games", "Schedule Status by Season", color_col="status_display")

    st.markdown("### Top Player Seasons")

    if len(player_seasons) > 0:
        player_seasons_filtered = filter_seasons(player_seasons, selected_seasons)
        player_seasons_filtered = filter_positions(player_seasons_filtered, selected_positions)
        player_seasons_filtered = apply_min_games(player_seasons_filtered, min_games)

        metric_options = [
            c for c in [
                "points",
                "points_per_game",
                "scoring_points",
                "scoring_points_per_game",
                "goals",
                "goals_per_game",
                "one_point_goals",
                "two_point_goals",
                "assists",
                "assists_per_game",
                "shots",
                "shots_per_game",
                "ground_balls",
                "caused_turnovers",
                "touches_per_game",
            ]
            if c in player_seasons_filtered.columns
        ]

        player_metric = st.selectbox(
            "Player season chart metric",
            options=metric_options,
            index=0 if metric_options else None,
            format_func=pretty_col,
            key="overview_player_metric",
        )

        if player_metric:
            top_players = safe_sort(player_seasons_filtered, player_metric, ascending=False).head(25)
            safe_bar_chart(
                top_players.head(15).sort_values(player_metric),
                "full_name",
                player_metric,
                f"Top Player Seasons by {pretty_col(player_metric)}",
                color_col="position",
                orientation="h",
            )
            display_table(top_players, height=360)
    else:
        st.info("Player season stats are not available.")

    st.markdown("### Top Team Seasons")

    if len(team_seasons) > 0:
        team_seasons_filtered = filter_seasons(team_seasons, selected_seasons)

        if selected_teams and "team_name" in team_seasons_filtered.columns:
            team_seasons_filtered = team_seasons_filtered[team_seasons_filtered["team_name"].isin(selected_teams)].copy()

        team_seasons_filtered = apply_min_games(team_seasons_filtered, min_games)

        team_metric_options = [
            c for c in [
                "scores",
                "scores_per_game",
                "score_margin_per_game",
                "goals",
                "shots",
                "shots_per_game",
                "turnovers",
                "turnovers_per_game",
                "saves",
                "touches",
                "touches_per_game",
                "time_in_possession",
                "time_in_possession_per_game",
            ]
            if c in team_seasons_filtered.columns
        ]

        team_metric = st.selectbox(
            "Team season chart metric",
            options=team_metric_options,
            index=1 if len(team_metric_options) > 1 else 0,
            format_func=pretty_col,
            key="overview_team_metric",
        )

        if team_metric:
            top_teams = safe_sort(team_seasons_filtered, team_metric, ascending=False).head(25)
            safe_bar_chart(
                top_teams.head(15).sort_values(team_metric),
                "team_name",
                team_metric,
                f"Top Team Seasons by {pretty_col(team_metric)}",
                color_col="season",
                orientation="h",
            )
            display_table(top_teams, height=360)
    else:
        st.info("Team season stats are not available.")


# ============================================================
# SEASON PAGE
# ============================================================

with tab_season:
    st.subheader("Season Page")
    st.markdown('<div class="section-note">Choose a season and review standings, team rankings, player leaders, goalie leaders, faceoff leaders, and schedule status.</div>', unsafe_allow_html=True)

    if not seasons:
        st.info("No seasons available.")
    else:
        selected_season_page = st.selectbox(
            "Select season",
            options=seasons,
            index=len(seasons) - 1,
            key="season_page_season",
        )

        team_seasons = read_table("marts.team_season_stats")
        player_seasons = read_table("marts.player_season_stats")
        schedule = schedule_display_table()

        season_team = filter_seasons(team_seasons, [selected_season_page])
        season_player = filter_seasons(player_seasons, [selected_season_page])
        season_schedule = filter_seasons(schedule, [selected_season_page])

        st.markdown("### Team Standings")

        if len(season_team) > 0:
            sort_col = "win_pct" if "win_pct" in season_team.columns else "wins"
            standings = safe_sort(season_team, sort_col, ascending=False)
            display_table(
                standings[
                    [c for c in [
                        "team_name",
                        "games",
                        "wins",
                        "losses",
                        "win_pct",
                        "scores_per_game",
                        "scores_against_per_game",
                        "score_margin_per_game",
                        "shots_per_game",
                        "touches_per_game",
                    ] if c in standings.columns]
                ],
                height=330,
            )
        else:
            st.info("No team standings available for this season.")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Player Leaders")
            if len(season_player) > 0:
                metric = st.selectbox(
                    "Player leader metric",
                    options=[c for c in ["points", "points_per_game", "scoring_points", "goals", "assists", "shots", "ground_balls", "caused_turnovers", "touches_per_game"] if c in season_player.columns],
                    format_func=pretty_col,
                    key="season_player_metric",
                )
                leaders = safe_sort(apply_min_games(season_player, min_games), metric, ascending=False).head(25)
                safe_bar_chart(leaders.head(15).sort_values(metric), "full_name", metric, f"Player Leaders — {pretty_col(metric)}", color_col="position", orientation="h")
                display_table(leaders, height=360)
            else:
                st.info("No player leaders available.")

        with c2:
            st.markdown("### Team Leaders")
            if len(season_team) > 0:
                metric = st.selectbox(
                    "Team leader metric",
                    options=[c for c in ["scores_per_game", "score_margin_per_game", "shots_per_game", "saves_per_game", "touches_per_game", "time_in_possession_per_game"] if c in season_team.columns],
                    format_func=pretty_col,
                    key="season_team_metric",
                )
                leaders = safe_sort(season_team, metric, ascending=False).head(20)
                safe_bar_chart(leaders.sort_values(metric), "team_name", metric, f"Team Leaders — {pretty_col(metric)}", orientation="h")
                display_table(leaders, height=360)
            else:
                st.info("No team leaders available.")

        st.markdown("### Goalie Leaders")
        if len(season_player) > 0 and "position" in season_player.columns:
            goalies = season_player[season_player["position"].astype(str).str.upper().eq("G")].copy()
            metric = first_existing(goalies, ["save_pct_calc", "save_pct", "saves_per_game"], "saves_per_game")
            goalies = safe_sort(apply_min_games(goalies, 1), metric, ascending=False).head(20)
            display_table(goalies, height=300)
        else:
            st.info("No goalie leaders available.")

        st.markdown("### Faceoff Leaders")
        if len(season_player) > 0 and "position" in season_player.columns:
            faceoffs = season_player[season_player["position"].astype(str).str.upper().eq("FO")].copy()
            metric = first_existing(faceoffs, ["faceoff_pct_calc", "faceoff_pct", "faceoffs_won_per_game"], "faceoffs_won_per_game")
            faceoffs = safe_sort(apply_min_games(faceoffs, 1), metric, ascending=False).head(20)
            display_table(faceoffs, height=300)
        else:
            st.info("No faceoff leaders available.")

        st.markdown("### Season Schedule")
        display_table(season_schedule, height=360)


# ============================================================
# MATCHUP PREVIEW / REVIEW
# ============================================================

with tab_matchup:
    st.subheader("Matchup Preview / Review")
    st.markdown('<div class="section-note">Review actual completed-game box scores or compare season/career profiles for upcoming matchups.</div>', unsafe_allow_html=True)

    schedule = schedule_display_table()
    team_game = read_table("clean.team_game_stats")
    player_game = read_table("clean.player_game_stats")
    team_seasons = read_table("marts.team_season_stats")
    team_career = read_table("marts.team_career_stats")

    if len(schedule) == 0:
        st.info("No schedule data available.")
    else:
        schedule = schedule.copy()

        if "game_label" not in schedule.columns:
            schedule["game_label"] = (
                schedule["season"].astype(str)
                + " G"
                + schedule.get("game_number", pd.Series("", index=schedule.index)).astype(str)
                + " — "
                + schedule.get("matchup", pd.Series(schedule.get("game_slug", ""), index=schedule.index)).astype(str)
            )

        selected_game_label = st.selectbox(
            "Select game",
            options=schedule["game_label"].tolist(),
            index=len(schedule) - 1,
            key="matchup_game_select",
        )

        game = schedule[schedule["game_label"] == selected_game_label].iloc[0]
        game_id = game.get("game_id")
        season = game.get("season")

        away = game.get("away_team_name", "Away")
        home = game.get("home_team_name", "Home")
        away_score = game.get("away_score", np.nan)
        home_score = game.get("home_score", np.nan)
        status = str(game.get("status_display", "")).lower()

        c1, c2 = st.columns(2)

        with c1:
            stat_card(str(away), fmt_value(away_score, 0), "Away Team")

        with c2:
            stat_card(str(home), fmt_value(home_score, 0), "Home Team")

        is_completed = status == "final" or (pd.notna(away_score) and pd.notna(home_score))

        if is_completed:
            st.markdown("### Completed Game Team Box Score")

            tg = team_game[team_game["game_id"].astype(str).eq(str(game_id))].copy() if len(team_game) > 0 and "game_id" in team_game.columns else pd.DataFrame()

            if len(tg) >= 2:
                left_team = str(tg.iloc[0].get("team_name", "Team 1"))
                right_team = str(tg.iloc[1].get("team_name", "Team 2"))

                stat_cols = [
                    c for c in [
                        "scores",
                        "goals",
                        "one_point_goals",
                        "two_point_goals",
                        "assists",
                        "shots",
                        "shots_on_goal",
                        "saves",
                        "ground_balls",
                        "turnovers",
                        "caused_turnovers",
                        "faceoffs_won",
                        "faceoffs",
                        "touches",
                        "total_passes",
                        "time_in_possession",
                        "official_total_possessions",
                        "offensive_sequence_proxy",
                    ]
                    if c in tg.columns
                ]

                rows = []

                for stat in stat_cols:
                    rows.append(
                        {
                            left_team: seconds_to_mmss(tg.iloc[0][stat]) if stat == "time_in_possession" else tg.iloc[0][stat],
                            "Stat": pretty_col(stat),
                            right_team: seconds_to_mmss(tg.iloc[1][stat]) if stat == "time_in_possession" else tg.iloc[1][stat],
                        }
                    )

                display_table(pd.DataFrame(rows), height=430, hide_cols=[])

            else:
                display_table(tg, height=360)

            st.markdown("### Completed Game Player Box Score")

            pg = player_game[player_game["game_id"].astype(str).eq(str(game_id))].copy() if len(player_game) > 0 and "game_id" in player_game.columns else pd.DataFrame()

            if len(pg) > 0:
                teams_in_game = sorted(pg["team_name"].dropna().astype(str).unique().tolist()) if "team_name" in pg.columns else []
                selected_box_team = st.selectbox("Team", options=teams_in_game, key="box_score_team_filter") if teams_in_game else None

                if selected_box_team:
                    pg = pg[pg["team_name"].astype(str).eq(selected_box_team)].copy()

                box_view = st.radio(
                    "Player box score view",
                    options=["Offense", "Defense", "Faceoff", "Goalie", "Full"],
                    horizontal=True,
                    key="box_score_view",
                )

                cols_by_view = {
                    "Offense": ["full_name", "position", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "shot_pct_calc", "ground_balls", "turnovers", "touches"],
                    "Defense": ["full_name", "position", "caused_turnovers", "ground_balls", "turnovers", "penalties", "penalty_time", "shots", "touches"],
                    "Faceoff": ["full_name", "position", "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct_calc", "ground_balls", "points", "assists", "shots", "touches"],
                    "Goalie": ["full_name", "position", "saves", "scores_against", "goals_against", "save_pct_calc", "clean_saves", "messy_saves", "touches"],
                }

                if box_view == "Full":
                    display_table(pg, height=500)
                else:
                    display_cols = [c for c in cols_by_view[box_view] if c in pg.columns]
                    display_table(pg[display_cols], height=500, hide_cols=[])
            else:
                st.info("No player box score rows are available for this game.")

        else:
            st.markdown("### Matchup Preview")

            teams_for_preview = [game.get("away_team_id"), game.get("home_team_id")]
            preview_context = st.radio("Preview context", options=["Season", "Career"], horizontal=True, key="matchup_preview_context")

            source = team_seasons if preview_context == "Season" else team_career

            if len(source) > 0:
                rows = source[source["team_id"].astype(str).isin([str(x) for x in teams_for_preview])].copy()

                if preview_context == "Season" and "season" in rows.columns:
                    rows = rows[pd.to_numeric(rows["season"], errors="coerce").eq(pd.to_numeric(pd.Series([season]), errors="coerce").iloc[0])].copy()

                display_table(rows, height=360)

                metrics = [c for c in ["scores_per_game", "scores_against_per_game", "score_margin_per_game", "shots_per_game", "touches_per_game", "time_in_possession_per_game"] if c in rows.columns]
                if metrics:
                    matrix = comparison_matrix(rows, "team_name", metrics)
                    st.markdown("### Matchup Comparison Matrix")
                    display_table(matrix, height=260, hide_cols=[])
            else:
                st.info("Team profile data is not available for preview.")


# ============================================================
# PLAYER EXPLORER
# ============================================================

with tab_players:
    st.subheader("Player Explorer")
    st.markdown('<div class="section-note">Review an individual player profile, season totals, per-game averages, recent form, and game log.</div>', unsafe_allow_html=True)

    players = read_table("clean.player_directory")
    player_season = read_table("marts.player_season_stats")
    player_career = read_table("marts.player_career_stats")
    player_last5 = read_table("marts.player_last5_stats")
    player_last10 = read_table("marts.player_last10_stats")
    player_games = read_table("clean.player_game_stats")

    if len(players) == 0 or "full_name" not in players.columns:
        st.info("No player directory available.")
    else:
        player_names = players["full_name"].dropna().astype(str).sort_values().tolist()

        selected_player = st.selectbox(
            "Select player",
            options=player_names,
            key="player_explorer_select",
        )

        player_row = players[players["full_name"].astype(str).eq(selected_player)].head(1)
        player_id = player_row["player_id"].iloc[0] if len(player_row) and "player_id" in player_row.columns else None

        subtitle_parts = []

        if len(player_row) > 0:
            for c in ["position", "latest_team_name", "games_in_database"]:
                if c in player_row.columns and pd.notna(player_row[c].iloc[0]):
                    subtitle_parts.append(f"{pretty_col(c)}: {player_row[c].iloc[0]}")

        profile_header(selected_player, " | ".join(map(str, subtitle_parts)) if subtitle_parts else "Player profile")

        context = st.radio(
            "Profile context",
            options=["Career", "Last 10", "Last 5"],
            horizontal=True,
            key="player_profile_context",
        )

        context_df = {
            "Career": player_career,
            "Last 10": player_last10,
            "Last 5": player_last5,
        }.get(context, player_career)

        if len(context_df) > 0 and "player_id" in context_df.columns:
            pr = context_df[context_df["player_id"].astype(str).eq(str(player_id))].head(1)
        else:
            pr = pd.DataFrame()

        if len(pr) > 0:
            r = pr.iloc[0]
            cards = st.columns(6)

            card_specs = [
                ("Games", "games", 0, False),
                ("Points/G", "points_per_game", 2, False),
                ("Goals/G", "goals_per_game", 2, False),
                ("Assists/G", "assists_per_game", 2, False),
                ("Shots/G", "shots_per_game", 2, False),
                ("Touches/G", "touches_per_game", 2, False),
            ]

            for i, spec in enumerate(card_specs):
                with cards[i]:
                    stat_card(spec[0], fmt_value(r.get(spec[1]), spec[2], spec[3]))

        st.markdown("### Season Totals and Averages")
        st.caption("Season-by-season totals and per-game averages for the selected player.")

        if len(player_season) > 0 and "player_id" in player_season.columns:
            ps = player_season[player_season["player_id"].astype(str).eq(str(player_id))].copy()
            ps = ps.sort_values("season") if "season" in ps.columns else ps

            view = st.radio(
                "Season table view",
                options=["Summary", "Per Game", "Full Detail"],
                horizontal=True,
                key=f"player_profile_season_view_{player_id}",
            )

            if view == "Summary":
                cols = [c for c in ["season", "teams", "position", "games", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers", "touches", "total_passes"] if c in ps.columns]
                display_table(ps[cols], height=330, hide_cols=[])
            elif view == "Per Game":
                cols = [c for c in ["season", "teams", "position", "games", "points_per_game", "scoring_points_per_game", "one_point_goals_per_game", "two_point_goals_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "shots_on_goal_per_game", "ground_balls_per_game", "turnovers_per_game", "caused_turnovers_per_game", "touches_per_game", "total_passes_per_game"] if c in ps.columns]
                display_table(ps[cols], height=330, hide_cols=[])
            else:
                display_table(ps, height=380)

            metric_cols = [c for c in ["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "touches_per_game"] if c in ps.columns]
            if metric_cols and "season" in ps.columns:
                safe_line_chart(ps, "season", metric_cols, f"{selected_player} Season Trend")
        else:
            st.info("No season-level player totals are available for this player.")

        st.markdown("### Game Log")

        if len(player_games) > 0 and "player_id" in player_games.columns:
            pg = player_games[player_games["player_id"].astype(str).eq(str(player_id))].copy()
            pg = pg.sort_values([c for c in ["season", "game_number"] if c in pg.columns], na_position="last")
            display_table(pg, height=420)
            download_csv(pg, f"{selected_player.replace(' ', '_').lower()}_game_log.csv", "Download player game log CSV")
        else:
            st.info("No player game log is available.")


# ============================================================
# TEAM EXPLORER
# ============================================================

with tab_teams:
    st.subheader("Team Explorer")
    st.markdown('<div class="section-note">Review team profile, style, player totals, season trends, and game logs.</div>', unsafe_allow_html=True)

    teams = read_table("clean.team_directory")
    team_season = read_table("marts.team_season_stats")
    team_career = read_table("marts.team_career_stats")
    team_games = read_table("clean.team_game_stats")
    player_by_team = read_table("marts.player_season_stats_by_team")

    if len(teams) == 0 or "team_name" not in teams.columns:
        st.info("No team directory available.")
    else:
        selected_team = st.selectbox(
            "Select team",
            options=teams["team_name"].dropna().astype(str).sort_values().tolist(),
            key="team_explorer_select",
        )

        team_lookup = teams[teams["team_name"].astype(str).eq(selected_team)].head(1)
        team_id = team_lookup["team_id"].iloc[0] if len(team_lookup) and "team_id" in team_lookup.columns else None

        team_context = st.radio(
            "Team profile context",
            options=["Career"] + [str(s) for s in seasons],
            horizontal=True,
            key="team_explorer_context",
        )

        profile_header(selected_team, f"Context: {team_context}")

        if team_context == "Career":
            source = team_career
            tr = source[source["team_id"].astype(str).eq(str(team_id))].head(1) if len(source) and "team_id" in source.columns else pd.DataFrame()
        else:
            source = team_season
            tr = source[
                source["team_id"].astype(str).eq(str(team_id))
                & pd.to_numeric(source["season"], errors="coerce").eq(int(team_context))
            ].head(1) if len(source) and "team_id" in source.columns and "season" in source.columns else pd.DataFrame()

        if len(tr) > 0:
            r = tr.iloc[0]
            cards = st.columns(6)
            specs = [
                ("Games", "games", 0, False),
                ("Wins", "wins", 0, False),
                ("Losses", "losses", 0, False),
                ("Scores/G", "scores_per_game", 2, False),
                ("Against/G", "scores_against_per_game", 2, False),
                ("Touches/G", "touches_per_game", 2, False),
            ]
            for i, spec in enumerate(specs):
                with cards[i]:
                    stat_card(spec[0], fmt_value(r.get(spec[1]), spec[2], spec[3]))

        st.markdown("### Team Player Totals")
        st.caption("Player production for the selected team. Use the filters below to review all-time team totals, a specific season, or the active team profile context.")

        if len(player_by_team) == 0:
            st.info("Player season totals by team are not available.")
        else:
            pb = player_by_team[player_by_team["team_id"].astype(str).eq(str(team_id))].copy() if "team_id" in player_by_team.columns else pd.DataFrame()

            if len(pb) == 0 and "team_name" in player_by_team.columns:
                pb = player_by_team[player_by_team["team_name"].astype(str).eq(str(selected_team))].copy()

            if len(pb) == 0:
                st.info("No player totals are available for the selected team.")
            else:
                seasons_available = sorted(pd.to_numeric(pb["season"], errors="coerce").dropna().astype(int).unique().tolist()) if "season" in pb.columns else []

                c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.4, 0.8, 1.2])

                with c1:
                    time_frame = st.selectbox(
                        "Time Frame",
                        options=["Team Profile Context", "All Time", "Specific Season"],
                        key=f"team_player_timeframe_{team_id}_{team_context}",
                    )

                default_season = seasons_available[-1] if seasons_available else None
                if team_context != "Career":
                    try:
                        if int(team_context) in seasons_available:
                            default_season = int(team_context)
                    except Exception:
                        pass

                with c2:
                    specific_season = st.selectbox(
                        "Season",
                        options=seasons_available,
                        index=seasons_available.index(default_season) if default_season in seasons_available else 0,
                        disabled=time_frame != "Specific Season",
                        key=f"team_player_specific_season_{team_id}_{team_context}",
                    )

                if time_frame == "Team Profile Context":
                    if team_context == "Career":
                        pb_base = pb.copy()
                        context_label = "All Time"
                    else:
                        pb_base = pb[pd.to_numeric(pb["season"], errors="coerce").eq(int(team_context))].copy()
                        context_label = f"{team_context} Season"
                elif time_frame == "All Time":
                    pb_base = pb.copy()
                    context_label = "All Time"
                else:
                    pb_base = pb[pd.to_numeric(pb["season"], errors="coerce").eq(int(specific_season))].copy()
                    context_label = f"{specific_season} Season"

                if context_label == "All Time" and "player_id" in pb_base.columns:
                    total_cols = [
                        c for c in [
                            "games", "points", "scoring_points", "one_point_goals", "two_point_goals",
                            "goals", "assists", "shots", "shots_on_goal", "ground_balls",
                            "turnovers", "caused_turnovers", "faceoffs_won", "faceoffs_lost",
                            "faceoffs", "saves", "scores_against", "goals_against",
                            "touches", "total_passes",
                        ]
                        if c in pb_base.columns
                    ]
                    agg = {c: "sum" for c in total_cols}

                    for c in ["full_name", "position", "position_name", "team_name"]:
                        if c in pb_base.columns:
                            agg[c] = "last"

                    pb_base = pb_base.groupby("player_id", dropna=False).agg(agg).reset_index()
                    pb_base["season"] = "All Time"

                    if "games" in pb_base.columns:
                        games = pd.to_numeric(pb_base["games"], errors="coerce").replace(0, np.nan)
                        for total, rate in {
                            "points": "points_per_game",
                            "scoring_points": "scoring_points_per_game",
                            "one_point_goals": "one_point_goals_per_game",
                            "two_point_goals": "two_point_goals_per_game",
                            "goals": "goals_per_game",
                            "assists": "assists_per_game",
                            "shots": "shots_per_game",
                            "shots_on_goal": "shots_on_goal_per_game",
                            "ground_balls": "ground_balls_per_game",
                            "turnovers": "turnovers_per_game",
                            "caused_turnovers": "caused_turnovers_per_game",
                            "touches": "touches_per_game",
                            "total_passes": "total_passes_per_game",
                        }.items():
                            if total in pb_base.columns:
                                pb_base[rate] = pd.to_numeric(pb_base[total], errors="coerce") / games

                position_options = sorted(pb_base["position"].dropna().astype(str).unique().tolist()) if "position" in pb_base.columns else []

                with c3:
                    selected_team_positions = st.multiselect(
                        "Positions",
                        options=position_options,
                        default=[],
                        key=f"team_player_positions_{team_id}_{team_context}_{context_label}",
                    )

                with c4:
                    team_min_games = st.number_input(
                        "Min Games",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        key=f"team_player_min_games_{team_id}_{team_context}_{context_label}",
                    )

                sort_options = [
                    c for c in [
                        "points", "points_per_game", "scoring_points", "scoring_points_per_game",
                        "one_point_goals", "two_point_goals", "goals", "goals_per_game",
                        "assists", "assists_per_game", "shots", "shots_per_game",
                        "touches", "touches_per_game", "ground_balls", "caused_turnovers",
                        "faceoff_pct_calc", "save_pct_calc",
                    ]
                    if c in pb_base.columns
                ]

                with c5:
                    sort_metric = st.selectbox(
                        "Sort By",
                        options=sort_options,
                        index=0 if sort_options else None,
                        format_func=pretty_col,
                        key=f"team_player_sort_{team_id}_{team_context}_{context_label}",
                    )

                table_view = st.radio(
                    "Player Table View",
                    options=["Summary", "Per Game", "Specialists"],
                    horizontal=True,
                    key=f"team_player_view_{team_id}_{team_context}_{context_label}",
                )

                pb_filtered = pb_base.copy()
                pb_filtered = filter_positions(pb_filtered, selected_team_positions)
                pb_filtered = apply_min_games(pb_filtered, team_min_games)

                if sort_metric:
                    pb_filtered = safe_sort(pb_filtered, sort_metric, ascending=False)

                cards = st.columns(4)
                with cards[0]:
                    stat_card("Players", fmt_value(len(pb_filtered), 0))
                with cards[1]:
                    stat_card("Team", selected_team)
                with cards[2]:
                    stat_card("Time Frame", context_label)
                with cards[3]:
                    top_name = pb_filtered["full_name"].iloc[0] if len(pb_filtered) and "full_name" in pb_filtered.columns else "—"
                    stat_card("Top Player", top_name)

                if sort_metric and len(pb_filtered) > 0:
                    safe_bar_chart(
                        pb_filtered.head(15).sort_values(sort_metric),
                        "full_name",
                        sort_metric,
                        f"{selected_team} — {context_label} Player Leaders by {pretty_col(sort_metric)}",
                        color_col="position",
                        orientation="h",
                    )

                if table_view == "Summary":
                    cols = [c for c in ["season", "full_name", "position", "games", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers", "touches"] if c in pb_filtered.columns]
                elif table_view == "Per Game":
                    cols = [c for c in ["season", "full_name", "position", "games", "points_per_game", "scoring_points_per_game", "one_point_goals_per_game", "two_point_goals_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "shots_on_goal_per_game", "ground_balls_per_game", "turnovers_per_game", "caused_turnovers_per_game", "touches_per_game", "total_passes_per_game"] if c in pb_filtered.columns]
                else:
                    cols = [c for c in ["season", "full_name", "position", "position_name", "games", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "shot_pct_calc", "ground_balls", "turnovers", "caused_turnovers", "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct_calc", "saves", "scores_against", "goals_against", "save_pct_calc", "touches", "total_passes"] if c in pb_filtered.columns]

                display_table(pb_filtered[cols], height=430, hide_cols=[])

                with st.expander("Full player table", expanded=False):
                    display_table(pb_filtered, height=430)

                download_csv(pb_filtered, f"{selected_team.replace(' ', '_').lower()}_{context_label.replace(' ', '_').lower()}_player_totals.csv")

        st.markdown("### Team Game Log")

        if len(team_games) > 0:
            tg = team_games[team_games["team_id"].astype(str).eq(str(team_id))].copy() if "team_id" in team_games.columns else pd.DataFrame()
            display_table(tg.sort_values([c for c in ["season", "game_number"] if c in tg.columns]), height=430)
        else:
            st.info("No team game log available.")


# ============================================================
# PLAYER RANKINGS
# ============================================================

with tab_player_rankings:
    st.subheader("Player Rankings")
    st.markdown('<div class="section-note">Rank players using the official overall score, which combines production, role value, usage, scoring value, and peer separation.</div>', unsafe_allow_html=True)

    rankings = read_table("marts.player_ranking_profiles")

    if len(rankings) == 0:
        st.info("Player ranking profiles are not available yet. Rebuild the warehouse to refresh player ranking data.")
    else:
        contexts = rankings["ranking_context"].dropna().astype(str).unique().tolist() if "ranking_context" in rankings.columns else []
        default_context = contexts[-1] if contexts else None

        c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.4, 1.0])

        with c1:
            ranking_context = st.selectbox("Ranking Context", options=contexts, index=contexts.index(default_context) if default_context in contexts else 0)

        filtered = rankings[rankings["ranking_context"].astype(str).eq(str(ranking_context))].copy() if ranking_context else rankings.copy()

        role_options = sorted(filtered["role_group"].dropna().astype(str).unique().tolist()) if "role_group" in filtered.columns else []
        pos_options = sorted(filtered["position"].dropna().astype(str).unique().tolist()) if "position" in filtered.columns else []

        with c2:
            selected_roles = st.multiselect("Roles", options=role_options, default=[])

        with c3:
            selected_ranking_positions = st.multiselect("Positions", options=pos_options, default=[])

        with c4:
            ranking_min_games = st.number_input("Min Games", min_value=0, max_value=100, value=1, step=1, key="ranking_min_games")

        if selected_roles and "role_group" in filtered.columns:
            filtered = filtered[filtered["role_group"].isin(selected_roles)].copy()

        filtered = filter_positions(filtered, selected_ranking_positions)
        filtered = apply_min_games(filtered, ranking_min_games)

        eligible_col = "is_ranking_eligible"
        if eligible_col in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered[eligible_col], errors="coerce").fillna(1).eq(1)].copy()

        sort_col = first_existing(filtered, ["v22_overall_rank", "overall_rank"], None)

        if sort_col:
            filtered = filtered.sort_values(sort_col, ascending=True, na_position="last")
        else:
            filtered = safe_sort(filtered, "v22_overall_score", ascending=False)

        cards = st.columns(5)
        with cards[0]:
            stat_card("Players", fmt_value(len(filtered), 0))
        with cards[1]:
            stat_card("Context", ranking_context)
        with cards[2]:
            top_player = filtered["full_name"].iloc[0] if len(filtered) and "full_name" in filtered.columns else "—"
            stat_card("Top Player", top_player)
        with cards[3]:
            top_score = filtered["v22_overall_score"].iloc[0] if len(filtered) and "v22_overall_score" in filtered.columns else np.nan
            stat_card("Top Score", fmt_value(top_score))
        with cards[4]:
            top_role = filtered["role_group"].iloc[0] if len(filtered) and "role_group" in filtered.columns else "—"
            stat_card("Top Role", top_role)

        chart_metric = st.selectbox(
            "Chart metric",
            options=[c for c in ["v22_overall_score", "role_context_value_score", "offensive_score", "defensive_score", "faceoff_score", "goalie_score", "usage_score"] if c in filtered.columns],
            format_func=pretty_col,
            key="ranking_chart_metric",
        )

        if chart_metric:
            chart_df = filtered.head(25).sort_values(chart_metric)
            safe_bar_chart(chart_df, "full_name", chart_metric, f"Top Players by {pretty_col(chart_metric)}", color_col="role_group", orientation="h")

        show_advanced = st.checkbox("Show advanced columns", value=False, key="rankings_show_advanced")

        base_cols = [
            "v22_overall_rank", "full_name", "position", "role_group", "teams", "games",
            "v22_overall_score", "role_context_value_score", "role_value_tier",
            "points_per_game", "scoring_points_per_game", "goals_per_game", "assists_per_game",
            "shots_per_game", "touches_per_game",
        ]

        advanced_cols = [
            "offensive_score", "usage_score", "defensive_score", "faceoff_score", "goalie_score",
            "goal_value_score", "role_primary_score", "role_separation_score", "role_adjusted_z",
            "v22_position_rank", "v22_position_percentile", "sample_size_note",
        ]

        display_cols = base_cols + advanced_cols if show_advanced else base_cols
        display_cols = [c for c in display_cols if c in filtered.columns]

        display_table(filtered[display_cols].head(150), height=540, hide_cols=[])
        download_csv(filtered, f"player_rankings_{ranking_context.replace(' ', '_').lower()}.csv")


# ============================================================
# TEAM PROFILES
# ============================================================

with tab_team_profiles:
    st.subheader("Team Profiles")
    st.markdown('<div class="section-note">Compare team identity using offense, defense, possession, ball movement, pace, and scoring margin.</div>', unsafe_allow_html=True)

    profiles = read_table("marts.team_style_profiles")

    if len(profiles) == 0:
        st.info("Team style profiles are not available yet. Rebuild the warehouse to refresh team profile data.")
    else:
        contexts = profiles["profile_context"].dropna().astype(str).unique().tolist() if "profile_context" in profiles.columns else []
        default_context = contexts[-1] if contexts else None

        c1, c2 = st.columns([1.4, 1.4])

        with c1:
            profile_context = st.selectbox("Profile Context", options=contexts, index=contexts.index(default_context) if default_context in contexts else 0)

        filtered = profiles[profiles["profile_context"].astype(str).eq(str(profile_context))].copy()

        with c2:
            profile_view = st.radio("View", options=["Summary", "Metrics", "Full Detail"], horizontal=True)

        if "profile_rank" in filtered.columns:
            filtered = filtered.sort_values("profile_rank", ascending=True, na_position="last")

        cards = st.columns(4)
        with cards[0]:
            stat_card("Teams", fmt_value(len(filtered), 0))
        with cards[1]:
            top_team = filtered["team_name"].iloc[0] if len(filtered) and "team_name" in filtered.columns else "—"
            stat_card("Top Team", top_team)
        with cards[2]:
            top_score = filtered["team_style_overall_score"].iloc[0] if len(filtered) and "team_style_overall_score" in filtered.columns else np.nan
            stat_card("Top Style Score", fmt_value(top_score))
        with cards[3]:
            context_label = profile_context if profile_context else "—"
            stat_card("Context", context_label)

        if "team_style_overall_score" in filtered.columns:
            safe_bar_chart(
                filtered.sort_values("team_style_overall_score"),
                "team_name",
                "team_style_overall_score",
                f"Team Style Scores — {profile_context}",
                color_col="defensive_profile_label" if "defensive_profile_label" in filtered.columns else None,
                orientation="h",
            )

        if profile_view == "Summary":
            cols = [c for c in ["profile_rank", "team_name", "games", "wins", "losses", "win_pct", "team_style_overall_score", "scores_per_game", "scores_allowed_per_game", "shots_per_game", "touches_per_game", "possession_pg", "style_summary", "sample_size_note"] if c in filtered.columns]
        elif profile_view == "Metrics":
            cols = [c for c in ["profile_rank", "team_name", "team_style_overall_score", "offensive_volume_score", "offensive_efficiency_score", "ball_movement_score", "possession_control_score", "defensive_suppression_score", "pace_tempo_score", "pace_label", "offensive_profile_label", "defensive_profile_label", "possession_profile_label"] if c in filtered.columns]
        else:
            cols = list(filtered.columns)

        display_table(filtered[cols], height=520, hide_cols=[])
        download_csv(filtered, f"team_profiles_{profile_context.replace(' ', '_').lower()}.csv")


# ============================================================
# GOALIE / FACEOFF
# ============================================================

with tab_specialists:
    st.subheader("Goalie / Faceoff")
    st.markdown('<div class="section-note">Specialist leaderboards for goalies and faceoff players.</div>', unsafe_allow_html=True)

    player_seasons = read_table("marts.player_season_stats")

    if len(player_seasons) == 0:
        st.info("Player season stats are not available.")
    else:
        season_choice = st.selectbox("Season", options=seasons, index=len(seasons) - 1 if seasons else 0, key="specialists_season")
        season_df = filter_seasons(player_seasons, [season_choice])

        gcol, focol = st.columns(2)

        with gcol:
            st.markdown("### Goalies")
            goalies = season_df[season_df["position"].astype(str).str.upper().eq("G")].copy() if "position" in season_df.columns else pd.DataFrame()
            metric = first_existing(goalies, ["save_pct_calc", "save_pct", "saves_per_game"], "saves_per_game")
            display_table(safe_sort(goalies, metric, ascending=False), height=420)

        with focol:
            st.markdown("### Faceoff")
            faceoffs = season_df[season_df["position"].astype(str).str.upper().eq("FO")].copy() if "position" in season_df.columns else pd.DataFrame()
            metric = first_existing(faceoffs, ["faceoff_pct_calc", "faceoff_pct", "faceoffs_won_per_game"], "faceoffs_won_per_game")
            display_table(safe_sort(faceoffs, metric, ascending=False), height=420)


# ============================================================
# PLAYER COMPARISON
# ============================================================

with tab_player_compare:
    st.subheader("Player Comparison")
    st.markdown('<div class="section-note">Compare selected players across career, recent, or season-level production.</div>', unsafe_allow_html=True)

    players = read_table("clean.player_directory")
    player_career = read_table("marts.player_career_stats")
    player_season = read_table("marts.player_season_stats")
    player_last5 = read_table("marts.player_last5_stats")
    player_last10 = read_table("marts.player_last10_stats")

    if len(players) == 0:
        st.info("No players available.")
    else:
        names = players["full_name"].dropna().astype(str).sort_values().tolist()
        selected_compare_players = st.multiselect("Select players", options=names, default=names[:2] if len(names) >= 2 else names)

        context = st.radio("Comparison context", options=["Career", "Season", "Last 10", "Last 5"], horizontal=True, key="player_compare_context")

        if context == "Career":
            source = player_career
        elif context == "Season":
            source = player_season
            comp_season = st.selectbox("Season", options=seasons, index=len(seasons) - 1 if seasons else 0, key="player_compare_season")
            source = filter_seasons(source, [comp_season])
        elif context == "Last 10":
            source = player_last10
        else:
            source = player_last5

        ids = players[players["full_name"].isin(selected_compare_players)]["player_id"].astype(str).tolist() if "player_id" in players.columns else []

        df = source[source["player_id"].astype(str).isin(ids)].copy() if len(source) > 0 and "player_id" in source.columns else pd.DataFrame()

        metrics = [c for c in ["games", "points_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game", "touches_per_game", "save_pct_calc", "faceoff_pct_calc"] if c in df.columns]

        if len(df) > 0:
            display_table(comparison_matrix(df, "full_name", metrics), height=360, hide_cols=[])
            display_table(df, height=420)
        else:
            st.info("No comparison data available for selected players.")


# ============================================================
# TEAM COMPARISON
# ============================================================

with tab_team_compare:
    st.subheader("Team Comparison")
    st.markdown('<div class="section-note">Compare selected teams across career or season-level team performance.</div>', unsafe_allow_html=True)

    teams = read_table("clean.team_directory")
    team_career = read_table("marts.team_career_stats")
    team_season = read_table("marts.team_season_stats")

    if len(teams) == 0:
        st.info("No teams available.")
    else:
        names = teams["team_name"].dropna().astype(str).sort_values().tolist()
        selected_compare_teams = st.multiselect("Select teams", options=names, default=names[:2] if len(names) >= 2 else names)

        context = st.radio("Comparison context", options=["Career", "Season"], horizontal=True, key="team_compare_context")

        if context == "Career":
            source = team_career
        else:
            source = team_season
            comp_season = st.selectbox("Season", options=seasons, index=len(seasons) - 1 if seasons else 0, key="team_compare_season")
            source = filter_seasons(source, [comp_season])

        ids = teams[teams["team_name"].isin(selected_compare_teams)]["team_id"].astype(str).tolist() if "team_id" in teams.columns else []

        df = source[source["team_id"].astype(str).isin(ids)].copy() if len(source) > 0 and "team_id" in source.columns else pd.DataFrame()

        metrics = [c for c in ["games", "wins", "losses", "win_pct", "scores_per_game", "scores_against_per_game", "score_margin_per_game", "shots_per_game", "touches_per_game", "time_in_possession_per_game"] if c in df.columns]

        if len(df) > 0:
            display_table(comparison_matrix(df, "team_name", metrics), height=360, hide_cols=[])
            display_table(df, height=420)
        else:
            st.info("No comparison data available for selected teams.")


# ============================================================
# LEADERBOARDS
# ============================================================

with tab_leaders:
    st.subheader("Leaderboards")
    st.markdown('<div class="section-note">Flexible player and team leaderboards using season, position, team, and minimum-games filters.</div>', unsafe_allow_html=True)

    leaderboard_type = st.radio("Leaderboard Type", options=["Players", "Teams"], horizontal=True)

    if leaderboard_type == "Players":
        df = read_table("marts.player_season_stats")
        df = filter_seasons(df, selected_seasons)
        df = filter_positions(df, selected_positions)
        df = apply_min_games(df, min_games)

        metric_options = [c for c in ["points", "points_per_game", "scoring_points", "goals", "one_point_goals", "two_point_goals", "assists", "shots", "ground_balls", "caused_turnovers", "touches_per_game", "save_pct_calc", "faceoff_pct_calc"] if c in df.columns]
        metric = st.selectbox("Metric", options=metric_options, format_func=pretty_col, key="player_leader_metric")
        leaders = safe_sort(df, metric, ascending=False).head(100)

        safe_bar_chart(leaders.head(20).sort_values(metric), "full_name", metric, f"Player Leaderboard — {pretty_col(metric)}", color_col="position", orientation="h")
        display_table(leaders, height=560)
        download_csv(leaders, f"player_leaderboard_{metric}.csv")

    else:
        df = read_table("marts.team_season_stats")
        df = filter_seasons(df, selected_seasons)

        if selected_teams and "team_name" in df.columns:
            df = df[df["team_name"].isin(selected_teams)].copy()

        df = apply_min_games(df, min_games)

        metric_options = [c for c in ["wins", "win_pct", "scores_per_game", "scores_against_per_game", "score_margin_per_game", "shots_per_game", "touches_per_game", "time_in_possession_per_game", "turnovers_per_game"] if c in df.columns]
        metric = st.selectbox("Metric", options=metric_options, format_func=pretty_col, key="team_leader_metric")
        leaders = safe_sort(df, metric, ascending=False).head(100)

        safe_bar_chart(leaders.head(20).sort_values(metric), "team_name", metric, f"Team Leaderboard — {pretty_col(metric)}", color_col="season", orientation="h")
        display_table(leaders, height=560)
        download_csv(leaders, f"team_leaderboard_{metric}.csv")


# ============================================================
# SCHEDULE
# ============================================================

with tab_schedule:
    st.subheader("Schedule")
    st.markdown('<div class="section-note">PLL schedule inventory with game status, final scores, and stat-availability signal.</div>', unsafe_allow_html=True)

    schedule = schedule_display_table()

    if len(schedule) == 0:
        st.info("No schedule data available.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            sched_seasons = st.multiselect("Season", options=seasons, default=seasons, key="schedule_seasons")

        with c2:
            status_options = sorted(schedule["status_display"].dropna().astype(str).unique().tolist()) if "status_display" in schedule.columns else []
            sched_status = st.multiselect("Status", options=status_options, default=[])

        with c3:
            team_filter = st.multiselect("Team", options=team_options, default=[], key="schedule_team_filter")

        filtered = filter_seasons(schedule, sched_seasons)

        if sched_status and "status_display" in filtered.columns:
            filtered = filtered[filtered["status_display"].isin(sched_status)].copy()

        if team_filter:
            team_mask = pd.Series(False, index=filtered.index)
            for c in ["home_team_name", "away_team_name"]:
                if c in filtered.columns:
                    team_mask = team_mask | filtered[c].isin(team_filter)
            filtered = filtered[team_mask].copy()

        display_table(filtered, height=620)
        download_csv(filtered, "pll_schedule.csv")


# ============================================================
# DATA GUIDE
# ============================================================

with tab_guide:
    st.subheader("Data Guide")
    st.markdown('<div class="section-note">Definitions, ranking method notes, and warehouse interpretation guide.</div>', unsafe_allow_html=True)

    note_box(
        "Official Player Ranking",
        """
        The official player ranking combines overall production, role-specific value,
        peer ranking, and peer separation. A player is rewarded not only for ranking highly
        within his role, but also for being meaningfully separated from comparable players.
        """,
    )

    guide_rows = [
        {
            "Metric": "Overall Score",
            "Definition": "Primary player ranking output. Blends production, role value, usage, scoring value, and peer separation.",
        },
        {
            "Metric": "Role Context Value",
            "Definition": "Role-aware score built from role score, peer percentile, and peer separation.",
        },
        {
            "Metric": "Peer Separation",
            "Definition": "Signal for whether a player is meaningfully ahead of comparable players, not merely ranked first by percentile.",
        },
        {
            "Metric": "Team Style Score",
            "Definition": "Team identity score combining offensive volume, efficiency, ball movement, possession control, defensive suppression, and pace.",
        },
        {
            "Metric": "Possession Time",
            "Definition": "Provider possession seconds formatted as MM:SS where useful. Some older games are labeled in QC if possession data is missing or non-standard.",
        },
    ]

    display_table(pd.DataFrame(guide_rows), height=300, hide_cols=[])

    st.markdown("### Available Warehouse Tables")
    display_table(table_index(), height=420, hide_cols=[])


# ============================================================
# DATA QUALITY
# ============================================================

with tab_quality:
    st.subheader("Data Quality")
    st.markdown('<div class="section-note">Warehouse quality checks, scrape logs, possession-data flags, and table inventory.</div>', unsafe_allow_html=True)

    quality = read_table("qc.quality_summary")

    if len(quality) > 0:
        st.markdown("### Quality Summary")
        display_table(quality, height=420, hide_cols=[])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Table Inventory")
        display_table(table_index(), height=400, hide_cols=[])

    with c2:
        possession = read_table("qc.game_possession_quality")
        st.markdown("### Possession Quality")
        display_table(possession, height=400, hide_cols=[])

    with st.expander("API Collection Log", expanded=False):
        api_log = read_table("qc.api_collection_log")
        display_table(api_log, height=500, hide_cols=[])

    with st.expander("Skipped Games", expanded=False):
        skipped = read_table("qc.skipped_games")
        display_table(skipped, height=400, hide_cols=[])

    with st.expander("DuckDB Table Counts", expanded=False):
        try:
            counts_df = query_df("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema IN ('clean', 'marts', 'qc')
                ORDER BY table_schema, table_name
            """)
            display_table(counts_df, height=500, hide_cols=[])
        except Exception as exc:
            st.exception(exc)
