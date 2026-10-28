# San Diego Wildfire Predictor

An XGBoost-based machine learning program that predicts where a San Diego County wildfire will spread **the next day**. It treats the area around a fire as a grid of pixels, pulls in interpolated weather from nearby stations plus fire-history features, and estimates the probability that each pixel burns within the next 24 hours, producing a day-by-day pixel map.

## How It Works

1. **Load data.** Daily fire perimeters for San Diego (GeoPackage), weather station locations, and daily weather observations.
2. **Build pixel training data.** For each fire and each consecutive pair of days (day *n* → day *n+1*), a grid of candidate points is laid over the day-*n* perimeter plus a 2 km buffer, at 250 m spacing. Each point is labeled `burned = 1` if it falls inside the day-*n+1* perimeter, otherwise `0`.
3. **Attach weather.** Each pixel is converted to lat/lon and assigned temperature, humidity, wind speed, and wind direction (as u/v components) from the nearest weather stations using inverse distance weighting (IDW).
4. **Add fire-history features.** Previous-day growth, previous-day wind speed, and cumulative burned area before the predicted day.
5. **Train and evaluate.** An XGBoost classifier is cross-validated with `GroupKFold` (grouped by fire, so no fire appears in both the train and test splits), then retrained on all data for the final model.
6. **Inspect feature importance** to see which factors drive predicted spread.

## Features Used

| Feature | Description |
|---|---|
| `dist_to_current_perim` | Distance (m) from the pixel to the current fire perimeter |
| `temp` | IDW-interpolated temperature from nearby stations |
| `humidity` | IDW-interpolated relative humidity |
| `wind_speed` | IDW-interpolated wind speed |
| `wind_u`, `wind_v` | East-west and north-south wind components |
| `prev_day_growth` | Area (km²) the fire grew on the previous day |
| `prev_day_wind_speed` | Wind speed on the previous day |
| `total_area_currently` | Cumulative area burned before the predicted day |

## Model

- **Algorithm:** `xgboost.XGBClassifier`
- **Hyperparameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `random_state=1`
- **Class imbalance:** most pixels do not burn, so `scale_pos_weight` is set to the negative/positive ratio of the training labels
- **Validation:** 5-fold `GroupKFold` by `fire_id`, reporting ROC AUC and a classification report per fold

## Project Structure

```
San_Diego_Wildfire_Predictor/
├── main.py          # Entry point: runs the full pipeline end to end
├── data_prep.py     # Candidate pixel generation, labeling, and fire-history features
├── model.py         # XGBoost training, cross-validation, and feature importance
├── weather.py       # Weather station KD-tree and IDW interpolation helpers imported by main.py
└── California_Historic_Fire_Perimeters_*.csv   # Historic California fire perimeter data
```

## Requirements

- Python 3.9+
- `pandas`
- `numpy`
- `geopandas`
- `shapely`
- `scikit-learn`
- `xgboost`
- `scipy` (for the station KD-tree)

```bash
pip install pandas numpy geopandas shapely scikit-learn xgboost scipy
```

## Data

The pipeline expects three input files:

| File | Contents |
|---|---|
| `filtered_san_diego_daily.gpkg` | Daily fire perimeter polygons for San Diego fires. Needs columns `id_left` (fire ID), `event_day`, `date`, `dy_ar_km2` (daily area growth in km²), and geometry, in **EPSG:32611** (UTM zone 11N). |
| `weather_stations.csv` | Station IDs and coordinates. |
| `weather_data.csv` | Daily observations per station: `temp`, `humidity`, `wind_speed`, and wind direction. |

The repository includes the **California Historic Fire Perimeters** dataset (CAL FIRE), which is the source for building the daily San Diego perimeter file.

## Usage

1. Update the file paths near the top of `main.py` to point at your data:

   ```python
   sd_daily = gpd.read_file(r'C:\path\to\your\filtered_san_diego_daily.gpkg')
   weather_stations = pd.read_csv(r'C:\path\to\weather_stations.csv')
   weather_data = pd.read_csv(r'C:\path\to\weather_data.csv')
   ```

2. Run the pipeline:

   ```bash
   python main.py
   ```

This builds the training set, prints per-fold ROC AUC and classification reports, trains the final model, and prints a feature importance table.

## Results

Cross-validation prints per-fold and mean ROC AUC when you run `main.py`. Add your numbers here once you have them:

| Metric | Value |
|---|---|
| Mean ROC AUC (5-fold, grouped by fire) | _TBD_ |
| Top features | _TBD_ |

## Roadmap

- Add terrain features (altitude/elevation, slope, aspect)
- Add nearby fuel and vegetation data
- Render predicted probabilities as a day-by-day pixel map
- Add a `requirements.txt` and make data paths configurable via command-line arguments

## Notes and Limitations

- Predictions are one day ahead and rely on the quality and density of nearby weather stations.
- Candidate pixels are limited to a 2 km buffer around the current perimeter, so spread beyond that distance in a single day cannot be predicted.
- This is a research/educational project and should not be used for emergency decision-making.

## License

No license specified yet. Consider adding one (for example MIT) if you want others to reuse the code.
