# PLL Data Platform

Streamlit app and automated warehouse builder for Premier Lacrosse League data.

## Main files
- `app.py` — Streamlit dashboard
- `scripts/build_warehouse.py` — scraper, cleaner, mart builder, and DuckDB warehouse writer
- `.github/workflows/update-pll-data.yml` — scheduled/manual GitHub Action

## Required secret
Add `PLL_BEARER_TOKEN` in GitHub repository secrets.
