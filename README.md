# 🏙️ Smart City Digital Twin Ecosystem

A data analysis project that brings together key parts of a city—districts, traffic, weather, air quality, public transport, power, emergencies, and city events—to explore how they relate and support day-to-day planning.

The project combines data preparation and cross-domain analysis with prototype machine learning models and an offline desktop app. It is intended as an exploratory decision-support demonstration: the analyses show patterns and associations in the supplied dataset, and the predictions should be treated as prototypes rather than operational forecasts.

## 🔎 What the project does

- **Prepares city datasets:** Jupyter notebooks inspect and clean each domain, including timestamp handling, duplicate checks, missing values, outliers, and useful derived fields.
- **Connects city systems:** Cross-table analysis aligns data by district and time to examine relationships such as weather and traffic, energy demand and outages, pollution, and event-period activity.
- **Explores prediction use cases:** Separate notebooks build and evaluate models for traffic accident risk, pollution alerts, transit service disruption, and emergency response time.
- **Provides an interactive demo:** A local Tkinter application lets users enter a scenario and view predictions from those four model pipelines. It derives calendar and district features where applicable and caches trained models locally.

## 🗂️ Repository layout

- `*.csv` — source datasets, district metadata, and the data dictionary.
- `data/cleaned_*.csv` — cleaned datasets used by the application and analysis.
- `cleaning_*.ipynb` — domain-specific data preparation notebooks.
- `cross_table.ipynb` — joined analysis across city systems.
- `prediction_*.ipynb` — model development and evaluation notebooks.
- `Smart_City_Report.ipynb` — consolidated project findings.
- `smart_city_gui/` — offline desktop demo, its dependencies, and model cache.

## 🚀 Run the desktop application

Run these commands from the repository root:

```powershell
python -m pip install -r smart_city_gui/requirements.txt
python smart_city_gui/app.py
```

The app needs a Python installation with Tkinter available, `districts.csv` in the repository root, and the cleaned CSV files in `data/`. On first launch it trains the pipelines and saves a local cache in `smart_city_gui/`; later launches reuse the cache. If the cleaned source data changes, the app refreshes the cache.

In the app, enter values using the units shown beside each field. Date and time provide hour and weekday features; district information and selected percentages are derived automatically.

## 📓 Explore the notebooks

Open the notebooks in Jupyter from the repository root. Start with the relevant `cleaning_*.ipynb` notebook, then use `cross_table.ipynb` for joined analysis or a `prediction_*.ipynb` notebook for an individual model. The notebooks document the analysis and are separate from the desktop app's runtime.

