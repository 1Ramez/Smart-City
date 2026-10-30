# Smart City ML Decision-Support Demo

An offline desktop GUI for trying four prediction models from the project:
accidents, pollution alerts, transit disruptions, and emergency response time.

## Run it

From the project folder, install the Python packages if needed:

```powershell
python -m pip install -r smart_city_gui/requirements.txt
python smart_city_gui/app.py
```

Tkinter is included with most Windows Python installations. The first launch trains
the models from the existing `data/cleaned_*.csv` files and `districts.csv`; this can
take a few minutes. Trained pipelines are cached in `smart_city_gui/` and reused on
later launches. The cache is refreshed automatically when source data files change.

The application reads the cleaned CSVs and never opens or modifies notebook files.
It does not send data over the network.

In the input form, enter regular values with the displayed units. The hourly
date/time automatically supplies hour, weekday, and weekend status. District
attributes are filled from `districts.csv`, and grid-load/renewable percentages
are calculated from the entered generation and demand values.

## Important model limitations

- Accident prediction currently uses `road_incident_count`, which the data dictionary
  marks as a potential leakage feature. Treat it as a demonstration until evaluated
  without that feature.
- Pollution-alert prediction classifies an alert for the supplied observation. Since
  the target is based on synthetic AQI, this is not yet evidence of a future forecast.
- The notebooks used random train/test splits. Future-period validation is still needed.
- The predictions are demonstrations, not operational advice or calibrated probabilities.
