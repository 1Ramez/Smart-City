"""Offline Tkinter demo for the Smart City prediction models.

The app trains standalone pipelines from the cleaned CSVs. It does not read or
modify any notebook. The first launch trains and caches the models locally.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_PATH = APP_DIR / "smart_city_models.joblib"
CACHE_VERSION = 3

MODEL_SPECS = {
    "accident": {
        "title": "Traffic accident risk",
        "target": "accident_flag",
        "features": [
            "traffic_volume", "average_speed_kmh", "road_incident_count",
            "rush_hour_flag", "weekend_flag", "rainfall_mm",
            "weather_condition", "district_name", "road_capacity_index",
            "population_density", "area_km2", "hour", "day_of_week",
        ],
        "categorical": ["weather_condition", "district_name"],
        "kind": "classification",
        "smote": True,
        "note": (
            "Prototype only: the data dictionary marks road_incident_count as a "
            "leakage risk. Validate again without it before using this model "
            "for advance warnings."
        ),
    },
    "pollution": {
        "title": "Pollution alert",
        "target": "pollution_alert_flag",
        "features": [
            "pm25", "o3", "co", "rainfall_mm", "weather_condition",
            "district_name", "industrial_activity_index",
            "commercial_activity_index", "population_density", "hour",
            "day_of_week",
        ],
        "categorical": ["weather_condition", "district_name"],
        "kind": "classification",
        "smote": False,
        "note": (
            "This classifies the alert for the supplied observation. The label "
            "is derived from synthetic AQI, so this is not yet a demonstrated "
            "future-alert forecast."
        ),
    },
    "service": {
        "title": "Transit service disruption",
        "target": "service_disruption_flag",
        "features": [
            "service_frequency", "traffic_volume", "average_speed_kmh",
            "road_incident_count", "rush_hour_flag", "weekend_flag",
            "rainfall_mm", "weather_condition", "district_name",
            "road_capacity_index", "population_density",
            "public_transport_access_index", "hour", "day_of_week",
        ],
        "categorical": ["weather_condition", "district_name"],
        "kind": "classification",
        "smote": False,
        "note": (
            "The notebook model was precise but missed some disruptions "
            "(recall about 60%). Use its result as a screening signal."
        ),
    },
    "response": {
        "title": "Emergency response time",
        "target": "response_time_minutes",
        "features": [
            "emergency_type", "severity_level", "ambulance_required",
            "traffic_volume", "average_speed_kmh", "congestion_index",
            "road_incident_count", "rush_hour_flag", "weekend_flag",
            "rainfall_mm", "weather_condition", "district_name",
            "road_capacity_index", "population_density", "area_km2",
            "hour", "day_of_week",
        ],
        "categorical": ["emergency_type", "weather_condition", "district_name"],
        "kind": "regression",
        "smote": False,
        "note": (
            "The notebook's random test split performed well (MAE about 1.7 "
            "minutes). Future-period performance has not yet been tested."
        ),
    },
}

DISTRICT_FEATURES = {
    "area_km2", "population_density", "road_capacity_index",
    "public_transport_access_index", "industrial_activity_index",
    "commercial_activity_index", "grid_capacity_mwh",
}
FLAG_FEATURES = {"rush_hour_flag", "weekend_flag", "ambulance_required"}
DERIVED_INPUT_FEATURES = {
    "hour", "day_of_week", "weekend_flag", "grid_load_percent",
    "renewable_generation_percent",
}
SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

FIELD_LABELS = {
    "traffic_volume": "Traffic volume (vehicles/hour)",
    "average_speed_kmh": "Average traffic speed (km/h)",
    "road_incident_count": "Road incidents recorded this hour",
    "rush_hour_flag": "Is this during rush hour?",
    "weekend_flag": "Weekend? (derived from date)",
    "rainfall_mm": "Hourly rainfall (mm)",
    "electricity_demand_mwh": "Electricity demand (MWh)",
    "solar_generation_mwh": "Solar generation (MWh)",
    "wind_generation_mwh": "Wind generation (MWh)",
    "grid_load_percent": "Grid load (calculated from demand/capacity)",
    "renewable_generation_percent": "Renewable share (calculated)",
    "pm25": "PM2.5 concentration (µg/m³)",
    "o3": "Ozone concentration (ppb)",
    "co": "Carbon monoxide concentration (ppm)",
    "service_frequency": "Minutes between transit vehicles",
    "ambulance_required": "Ambulance required?",
    "hour": "Hour (derived from date/time)",
    "day_of_week": "Day of week (derived from date/time)",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required cleaned data file was not found: {path}")
    return pd.read_csv(path)


def _add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    return frame


def _load_training_frames() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, list[str]]]:
    districts = pd.read_csv(PROJECT_DIR / "districts.csv")
    weather = _add_calendar_columns(_read_csv("cleaned_weather.csv"))
    traffic = _add_calendar_columns(_read_csv("cleaned_traffic.csv"))
    air = _add_calendar_columns(_read_csv("cleaned_air_quality.csv"))
    transport = _add_calendar_columns(_read_csv("cleaned_public_transport.csv"))
    emergencies = _add_calendar_columns(_read_csv("cleaned_emergency_events.csv"))

    weather_fields = ["district_id", "timestamp", "rainfall_mm", "weather_condition"]
    traffic_weather = traffic.merge(
        weather[weather_fields], on=["district_id", "timestamp"], how="left",
        validate="one_to_one",
    )

    district_fields = [
        "district_id", "district_name", "area_km2", "population_density",
        "road_capacity_index", "public_transport_access_index",
        "industrial_activity_index", "commercial_activity_index",
    ]
    district_fields = [c for c in district_fields if c in districts.columns]

    accident = traffic_weather.merge(
        districts[district_fields], on="district_id", how="left", validate="many_to_one"
    )

    pollution = air.merge(
        weather[weather_fields], on=["district_id", "timestamp"], how="left",
        validate="one_to_one",
    ).merge(
        districts[[c for c in ["district_id", "district_name", "industrial_activity_index",
                              "commercial_activity_index", "population_density"]
                   if c in districts.columns]],
        on="district_id", how="left", validate="many_to_one",
    )

    transit_traffic_fields = [
        "district_id", "timestamp", "traffic_volume", "average_speed_kmh",
        "road_incident_count", "rush_hour_flag", "weekend_flag",
    ]
    service = transport.merge(
        traffic[transit_traffic_fields], on=["district_id", "timestamp"], how="left",
        validate="one_to_one",
    ).merge(
        weather[weather_fields], on=["district_id", "timestamp"], how="left",
        validate="one_to_one",
    ).merge(
        districts[[c for c in ["district_id", "district_name", "road_capacity_index",
                              "population_density", "public_transport_access_index"]
                   if c in districts.columns]],
        on="district_id", how="left", validate="many_to_one",
    )

    # Emergency dispatches are minute-level; operational observations are hourly.
    emergencies["timestamp"] = emergencies["timestamp"].dt.floor("h")
    response = emergencies.merge(
        traffic, on=["district_id", "timestamp"], how="left", validate="many_to_one",
        suffixes=("", "_traffic"),
    ).merge(
        weather, on=["district_id", "timestamp"], how="left", validate="many_to_one",
        suffixes=("", "_weather"),
    ).merge(
        districts, on="district_id", how="left", validate="many_to_one",
        suffixes=("", "_district"),
    )
    response["severity_level"] = response["severity_level"].map(
        {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    )

    frames = {
        "accident": _add_calendar_columns(accident),
        "pollution": _add_calendar_columns(pollution),
        "service": _add_calendar_columns(service),
        "response": _add_calendar_columns(response),
    }
    categories = {
        "districts": sorted(districts["district_name"].dropna().astype(str).unique().tolist()),
        "weather": sorted(weather["weather_condition"].dropna().astype(str).unique().tolist()),
        "emergency_types": sorted(emergencies["emergency_type"].dropna().astype(str).unique().tolist()),
        "latest_observation": weather["timestamp"].max().strftime("%Y-%m-%d %H:00"),
    }
    return frames, districts, categories


def _make_estimator(key: str):
    if key == "accident":
        return LogisticRegression(max_iter=1000), True
    if key == "pollution":
        return RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=1, random_state=42,
        ), False
    if key == "service":
        return LogisticRegression(max_iter=1000, class_weight="balanced"), False
    return LinearRegression(), False


def _make_pipeline(key: str):
    spec = MODEL_SPECS[key]
    numeric = [c for c in spec["features"] if c not in spec["categorical"]]
    prep = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             spec["categorical"]),
        ],
        remainder="drop",
    )
    estimator, use_smote = _make_estimator(key)
    steps = [("preprocessing", prep)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", estimator))
    pipe_type = ImbalancedPipeline if use_smote else Pipeline
    return pipe_type(steps)


def _data_fingerprint() -> str:
    files = [
        "districts.csv", "data/cleaned_traffic.csv", "data/cleaned_weather.csv",
        "data/cleaned_air_quality.csv",
        "data/cleaned_public_transport.csv", "data/cleaned_emergency_events.csv",
    ]
    details = []
    for name in files:
        path = PROJECT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Required project file was not found: {path}")
        stat = path.stat()
        details.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(details).encode("utf-8")).hexdigest()


def train_or_load_models(progress_callback=None, force=False):
    fingerprint = _data_fingerprint()
    if CACHE_PATH.exists() and not force:
        try:
            saved = joblib.load(CACHE_PATH)
            if saved.get("version") == CACHE_VERSION and saved.get("fingerprint") == fingerprint:
                return saved["models"], saved["metadata"], True
        except Exception:
            pass

    frames, districts, categories = _load_training_frames()
    models = {}
    metadata = {"categories": categories, "district_attributes": {}}
    district_columns = [
        c for c in DISTRICT_FEATURES if c in districts.columns
    ]
    for _, row in districts.iterrows():
        metadata["district_attributes"][str(row["district_name"])] = {
            col: float(row[col]) for col in district_columns if pd.notna(row[col])
        }

    for key, spec in MODEL_SPECS.items():
        if progress_callback:
            progress_callback(f"Training {spec['title']}…")
        frame = frames[key]
        missing = [c for c in spec["features"] + [spec["target"]] if c not in frame.columns]
        if missing:
            raise ValueError(f"{spec['title']} is missing required columns: {missing}")
        training = frame[spec["features"] + [spec["target"]]].dropna()
        X = training[spec["features"]].copy()
        y = training[spec["target"]].astype(int if spec["kind"] == "classification" else float)
        model = _make_pipeline(key)
        model.fit(X, y)
        models[key] = model
        numeric = [c for c in spec["features"] if c not in spec["categorical"]]
        metadata.setdefault("defaults", {})[key] = {
            c: float(pd.to_numeric(X[c], errors="coerce").median())
            for c in numeric if c not in DISTRICT_FEATURES and c not in FLAG_FEATURES
        }

    joblib.dump(
        {"version": CACHE_VERSION, "fingerprint": fingerprint,
         "models": models, "metadata": metadata},
        CACHE_PATH,
        compress=3,
    )
    return models, metadata, False


def _pretty_name(name: str) -> str:
    labels = {
        "hour": "Hour (0–23)", "day_of_week": "Day of week (Mon=0, Sun=6)",
        "rainfall_mm": "Rainfall (mm)", "pm25": "PM2.5", "o3": "Ozone (O₃)",
        "co": "Carbon monoxide (CO)", "area_km2": "District area (km²)",
        "average_speed_kmh": "Average speed (km/h)",
        "electricity_demand_mwh": "Electricity demand (MWh)",
        "service_frequency": "Service frequency (minutes between vehicles)",
        "population_density": "Population density",
        "road_incident_count": "Road incidents in this hour",
    }
    return labels.get(name, name.replace("_", " ").title())


class SmartCityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart City | ML Decision-Support Demo")
        self.root.geometry("1000x760")
        self.root.minsize(840, 660)
        self.root.configure(bg="#101820")
        self.models = {}
        self.metadata = {}
        self.widgets = {}
        self.messages = queue.Queue()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101820")
        style.configure("Card.TFrame", background="#18232d")
        style.configure("Card.TLabel", background="#18232d", foreground="#e6edf3",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0d1b2a", foreground="#f8fafc",
                        font=("Segoe UI", 20, "bold"), padding=18)
        style.configure("Hint.TLabel", background="#18232d", foreground="#a9bac8",
                        font=("Segoe UI", 10))
        style.configure("Intro.TLabel", background="#101820", foreground="#c1cfda",
                        font=("Segoe UI", 11))
        style.configure("Disclaimer.TLabel", background="#101820", foreground="#8495a3",
                        font=("Segoe UI", 9))
        style.configure("Result.TLabel", background="#18232d", foreground="#7dd3fc",
                        font=("Segoe UI", 15, "bold"))
        style.configure("TLabel", background="#18232d", foreground="#e6edf3",
                        font=("Segoe UI", 10))
        style.configure("TButton", background="#155e75", foreground="#f8fafc",
                        font=("Segoe UI", 10, "bold"), padding=8, bordercolor="#26758a")
        style.map("TButton", background=[("active", "#0e7490"), ("disabled", "#293743")],
                  foreground=[("disabled", "#82919c")])
        style.configure("TEntry", fieldbackground="#22313d", foreground="#e6edf3",
                        insertcolor="#f8fafc", bordercolor="#344656", padding=5)
        style.configure("TCombobox", fieldbackground="#22313d", background="#22313d",
                        foreground="#e6edf3", arrowcolor="#c1cfda", bordercolor="#344656",
                        padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", "#22313d")],
                  foreground=[("readonly", "#e6edf3")],
                  selectbackground=[("readonly", "#155e75")],
                  selectforeground=[("readonly", "#f8fafc")])
        style.configure("TSpinbox", fieldbackground="#22313d", foreground="#e6edf3",
                        arrowcolor="#c1cfda", bordercolor="#344656", padding=5)
        root.option_add("*TCombobox*Listbox.background", "#22313d")
        root.option_add("*TCombobox*Listbox.foreground", "#e6edf3")
        root.option_add("*TCombobox*Listbox.selectBackground", "#155e75")
        root.option_add("*TCombobox*Listbox.selectForeground", "#f8fafc")

        header = ttk.Label(
            root, text="SMART CITY  |  Prediction Explorer", style="Title.TLabel"
        )
        header.pack(fill="x")
        ttk.Label(
            root,
            text=("Explore four trained models using district, weather, traffic, "
                  "air-quality, transit, and emergency inputs."),
            style="Intro.TLabel",
            padding=(18, 12),
        ).pack(fill="x")

        card = ttk.Frame(root, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Prediction task:", style="Card.TLabel").pack(side="left")
        self.model_var = tk.StringVar(value=MODEL_SPECS["accident"]["title"])
        self.model_keys = {spec["title"]: key for key, spec in MODEL_SPECS.items()}
        self.model_select = ttk.Combobox(
            row, textvariable=self.model_var, state="readonly",
            values=list(self.model_keys), width=34,
        )
        self.model_select.pack(side="left", padx=10)
        self.model_select.bind("<<ComboboxSelected>>", self._render_form)
        self.train_button = ttk.Button(
            row, text="Retrain models", command=lambda: self._start_training(force=True)
        )
        self.train_button.pack(side="right")

        self.status_var = tk.StringVar(value="Preparing models…")
        ttk.Label(card, textvariable=self.status_var, style="Hint.TLabel", padding=(0, 10)).pack(anchor="w")
        self.note_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.note_var, style="Hint.TLabel", wraplength=900,
                  padding=(0, 4)).pack(anchor="w", fill="x")

        self.form_frame = ttk.Frame(card, style="Card.TFrame")
        self.form_frame.pack(fill="x", pady=(12, 10))
        self.result_frame = ttk.Frame(card, style="Card.TFrame", padding=(12, 10))
        self.result_frame.pack(fill="x", pady=6)
        self.result_var = tk.StringVar(value="")
        ttk.Label(self.result_frame, textvariable=self.result_var,
                  style="Result.TLabel", wraplength=880).pack(anchor="w")
        self.predict_button = ttk.Button(card, text="Run prediction", command=self._predict,
                                         state="disabled")
        self.predict_button.pack(anchor="e", pady=(8, 0))

        self._start_training(force=False)
        self.root.after(150, self._poll_messages)

    def _selected_key(self):
        return self.model_keys.get(self.model_var.get(), "accident")

    def _start_training(self, force=False):
        self.train_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.result_var.set("")
        self.status_var.set("Loading data and training models in the background…")

        def progress(text):
            self.messages.put(("progress", text))

        def worker():
            try:
                models, metadata, from_cache = train_or_load_models(progress, force=force)
                self.messages.put(("ready", (models, metadata, from_cache)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "progress":
                    self.status_var.set(payload)
                elif kind == "ready":
                    self.models, self.metadata, from_cache = payload
                    state = "Loaded saved models." if from_cache else "Models trained and cached."
                    self.status_var.set(state + " Select a task and enter a scenario.")
                    self.train_button.configure(state="normal")
                    self.predict_button.configure(state="normal")
                    self._render_form()
                elif kind == "error":
                    self.status_var.set("Could not prepare models.")
                    self.train_button.configure(state="normal")
                    messagebox.showerror("Model setup failed", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_messages)

    def _render_form(self, _event=None):
        if not self.models:
            return
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.widgets = {}
        key = self._selected_key()
        spec = MODEL_SPECS[key]
        self.note_var.set(
            "Enter normal units. Date/time supplies hour, weekday, and weekend; district "
            "details and grid percentages are calculated automatically. " + spec["note"]
        )
        defaults = self.metadata.get("defaults", {}).get(key, {})
        cats = self.metadata.get("categories", {})
        district_attributes = self.metadata.get("district_attributes", {})

        visible_fields = []
        for feature in spec["features"]:
            if feature in DERIVED_INPUT_FEATURES or feature in DISTRICT_FEATURES:
                continue
            visible_fields.append(feature)
            if feature == "district_name":
                visible_fields.append("__observation_time")
        for index, feature in enumerate(visible_fields):
            r, c = index // 2, (index % 2) * 2
            label = ("Observation date/time (YYYY-MM-DD HH:MM)" if feature == "__observation_time"
                     else FIELD_LABELS.get(feature, _pretty_name(feature)))
            ttk.Label(self.form_frame, text=label,
                      style="Card.TLabel").grid(row=r, column=c, sticky="w", padx=(0, 8), pady=5)
            variable = tk.StringVar()
            if feature == "__observation_time":
                variable.set(cats.get("latest_observation", "2025-12-31 23:00"))
                widget = ttk.Entry(self.form_frame, textvariable=variable, width=26)
            elif feature == "district_name":
                values = cats.get("districts", [])
                variable.set(values[0] if values else "")
                widget = ttk.Combobox(self.form_frame, textvariable=variable, values=values,
                                      state="readonly", width=23)
                widget.bind("<<ComboboxSelected>>", self._district_changed)
            elif feature == "weather_condition":
                values = cats.get("weather", [])
                variable.set(values[0] if values else "")
                widget = ttk.Combobox(self.form_frame, textvariable=variable, values=values,
                                      state="readonly", width=23)
            elif feature == "emergency_type":
                values = cats.get("emergency_types", [])
                variable.set(values[0] if values else "")
                widget = ttk.Combobox(self.form_frame, textvariable=variable, values=values,
                                      state="readonly", width=23)
            elif feature == "severity_level":
                variable.set("Medium")
                widget = ttk.Combobox(self.form_frame, textvariable=variable,
                                      values=SEVERITY_LEVELS, state="readonly", width=23)
            elif feature in FLAG_FEATURES:
                variable.set("No")
                widget = ttk.Combobox(self.form_frame, textvariable=variable,
                                      values=["No", "Yes"], state="readonly", width=23)
            elif feature == "hour":
                variable.set(str(int(round(defaults.get(feature, 12))) % 24))
                widget = ttk.Spinbox(self.form_frame, from_=0, to=23, textvariable=variable, width=22)
            elif feature == "day_of_week":
                variable.set(str(int(round(defaults.get(feature, 0))) % 7))
                widget = ttk.Spinbox(self.form_frame, from_=0, to=6, textvariable=variable, width=22)
            else:
                value = defaults.get(feature, 0.0)
                variable.set(f"{value:.3f}".rstrip("0").rstrip("."))
                widget = ttk.Entry(self.form_frame, textvariable=variable, width=26)
            widget.grid(row=r, column=c + 1, sticky="ew", padx=(0, 18), pady=5)
            self.widgets[feature] = variable

        for col in range(4):
            self.form_frame.columnconfigure(col, weight=1 if col % 2 else 0)

    def _district_changed(self, _event=None):
        key = self._selected_key()
        district = self.widgets.get("district_name")
        if district is None:
            return
        attributes = self.metadata.get("district_attributes", {}).get(district.get(), {})
        for field in MODEL_SPECS[key]["features"]:
            if field in DISTRICT_FEATURES and field in attributes:
                self.metadata.setdefault("defaults", {}).setdefault(key, {})[field] = attributes[field]

    def _build_input(self, key: str) -> pd.DataFrame:
        values = {}
        for feature, variable in self.widgets.items():
            raw = variable.get().strip()
            if feature == "__observation_time":
                try:
                    observation_time = pd.to_datetime(raw, format="%Y-%m-%d %H:%M")
                except ValueError as exc:
                    raise ValueError("Use date/time format YYYY-MM-DD HH:MM, for example 2025-06-14 08:00.") from exc
            elif feature in FLAG_FEATURES:
                values[feature] = 1 if raw == "Yes" else 0
            elif feature == "severity_level":
                values[feature] = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}[raw]
            elif feature in MODEL_SPECS[key]["categorical"]:
                values[feature] = raw
            else:
                number = float(raw)
                if not np.isfinite(number) or number < 0:
                    raise ValueError(f"{_pretty_name(feature)} must be a non-negative number.")
                values[feature] = number
        district = values.get("district_name")
        attrs = self.metadata.get("district_attributes", {}).get(district, {})
        features = MODEL_SPECS[key]["features"]
        if "hour" in features:
            values["hour"] = int(observation_time.hour)
        if "day_of_week" in features:
            values["day_of_week"] = int(observation_time.dayofweek)
        if "weekend_flag" in features:
            values["weekend_flag"] = int(observation_time.dayofweek >= 5)
        if "grid_load_percent" in features:
            capacity = attrs.get("grid_capacity_mwh")
            if capacity is None or capacity <= 0:
                raise ValueError("District grid capacity is unavailable for calculating grid load.")
            values["grid_load_percent"] = 100 * values["electricity_demand_mwh"] / capacity
        if "renewable_generation_percent" in features:
            demand = values["electricity_demand_mwh"]
            if demand <= 0:
                raise ValueError("Electricity demand must be greater than zero.")
            values["renewable_generation_percent"] = 100 * (
                values["solar_generation_mwh"] + values["wind_generation_mwh"]
            ) / demand
        for feature in MODEL_SPECS[key]["features"]:
            if feature in DISTRICT_FEATURES:
                if feature not in attrs:
                    raise ValueError(f"No district value is available for {_pretty_name(feature)}.")
                values[feature] = attrs[feature]
        return pd.DataFrame([values], columns=MODEL_SPECS[key]["features"])

    def _predict(self):
        key = self._selected_key()
        try:
            sample = self._build_input(key)
            model = self.models[key]
            if MODEL_SPECS[key]["kind"] == "regression":
                minutes = float(model.predict(sample)[0])
                self.result_var.set(f"Estimated emergency response time: {minutes:.1f} minutes")
            else:
                prediction = int(model.predict(sample)[0])
                probability = float(model.predict_proba(sample)[0, 1])
                outcome = "Alert / event predicted" if prediction == 1 else "No alert / event predicted"
                self.result_var.set(
                    f"{outcome}\nModel score for a positive case: {probability:.1%} "
                    "(not a calibrated probability)"
                )
        except (ValueError, TypeError) as exc:
            messagebox.showwarning("Check the inputs", f"Please enter valid values.\n\n{exc}")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))


def main():
    root = tk.Tk()
    SmartCityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
