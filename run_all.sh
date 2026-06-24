#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -r requirements.txt
python3 reproduce_tables.py
python3 make_cdf_plots.py
echo "DONE -> tables in reproduced_tables.md, figures in figs/"
