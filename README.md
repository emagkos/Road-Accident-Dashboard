PROJECT: Road Accident Dashboard
============================================================

AUTHORS  /  Credits
------------------------------------------------------------
- Author: Evripidis Magkos emagkos@hotmail.com

Special thanks to the open-source projects and communities behind:
Streamlit, Folium/Leaflet, streamlit-folium, Plotly, GeoPandas, Shapely, scikit-learn, python-docx, Kaleido.

This project was developed within the FACTUAL development program Tools and Techniques for Geospatial ML

THANKS  /  Acknowledgments
------------------------------------------------------------
- Lisbon city open data providers for making geospatial accident data available.

CHANGELOG  /  A detailed changelog (for developers)
------------------------------------------------------------

v1.3.0 — 2025-10-14
- Added “Accident type (%) by month” stacked bar chart (expects month as Jan–Dec).
- Introduced FastMarkerCluster for large datasets; automatic fallback based on row count.
- Stabilized HeatMap gradient by using string keys (fixes a float split error).
- Word report now optionally includes charts when Kaleido is installed.
- Consistent red severity palette; distinct Serious vs Fatal shades.

v1.2.0 — 2025-10-07
- Street column auto-hidden when empty (header present but cells blank).
- Improved CSV upload handling and geometry sanitization.

v1.1.0 — 2025-09-28
- Added “Top 10 Streets” chart.
- KPI cards and basic insights section.

v1.0.0 — 2025-09-10
- Initial release with interactive map, MarkerCluster, HeatMap, and base charts.


NEWS  /  A basic changelog (for users)
------------------------------------------------------------
- New: Monthly view shows the share of accident severity each month (Jan–Dec).
- Faster: Large files render faster with automatic FastMarkerCluster.
- Export: Generate a Word report with KPIs, insights, tables, and charts (if Kaleido installed).
- Cleaner: Street chart hides itself if the street column is blank.


INSTALL  /  Installation instructions
------------------------------------------------------------
Option A: Using pip (recommended for most users)

1) (Optional) Create a virtual environment
   Windows PowerShell:
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   macOS/Linux:
     python -m venv .venv
     source .venv/bin/activate

2) Install requirements
   pip install -r requirements.txt
   # If not using a requirements file:
   pip install streamlit pandas numpy geopandas shapely folium streamlit-folium plotly scikit-learn python-docx kaleido

   NOTE (Windows Geo stack):
   If GeoPandas/Shapely fail to install, try:
   conda install -c conda-forge geopandas

3) Run the app
   streamlit run lisbon_accidents_dashboard.py
   Then open http://localhost:8501/ if it doesn’t auto-open.

Data:
- Put your CSV in ./data/ or upload via the sidebar.
- Required: latitude, longitude
- Optional: severity (or fatal_injuries/serious_injuries/minor_injuries), street, hour, weekday, month (Jan–Dec).

Template
------------------------------------------------------------


BUGS  /  Known issues & reporting
------------------------------------------------------------
Known issues:
- Very large CSVs may still cause slower interactions on some machines.
- Word export requires Kaleido for chart images (report still generates without images).

Report a bug:
1) Include OS, Python version, and package versions (pip freeze).
2) Share a minimal CSV sample if possible (with a few rows reproducing the bug).
3) Provide the Streamlit logs (terminal output) and the full traceback.

Open an issue at: https://github.com/emagkos/issues


CONTRIBUTING  /  HACKING  /  Guide for contributors
------------------------------------------------------------
Thank you for considering a contribution!

Workflow:
1) Fork the repository and create a feature branch:
   git checkout -b feat/short-description
2) Install dev dependencies & run locally:
   pip install -r requirements.txt
   streamlit run lisbon_accidents_dashboard.py
3) Write clear commits and include tests or screenshots for UI changes.
4) Open a Pull Request describing:
   - Problem statement & solution
   - Any performance or UX impact
   - How to test your change

Coding style:
- Keep functions small and focused.
- Prefer vectorized pandas operations where possible.
- Use consistent naming (snake_case) and docstrings for public helpers.

Performance notes:
- FastMarkerCluster is used automatically for large datasets (>~8k rows).
- Avoid unnecessary recalculation; respect Streamlit’s caching patterns.
- Keep “month” as 3-letter abbreviations (Jan–Dec) for the monthly chart.

Security/Privacy:
- Do not commit real personal data.
- Redact or anonymize sample datasets before sharing.

License & DCO:
- By contributing, you agree your contributions are licensed under the project’s MIT license.
- If your organization requires a DCO or CLA, please note it in your PR.

Contact:
- Maintainer: Evripidis Magkos emagkos@hotmail.com
- Discussions: https://github.com/emagkos/discussions
