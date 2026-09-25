from data_prep import build_pixel_training_data, merge_fire_attributes, last_day_models
from weather import build_station_tree, idw_lookup, idw_wind_direction
from model import train_and_evaluate, train_final, get_feature_col_importance
import pandas as pd

if __name__ == "__main__":
    import geopandas as gpd

    sd_daily = gpd.read_file(r'C:\path\to\your\filtered_san_diego_daily.gpkg')
    weather_stations = pd.read_csv(r'C:\path\to\weather_stations.csv')
    weather_data = pd.read_csv(r'C:\path\to\weather_data.csv')

    training_df = build_pixel_training_data(sd_daily)
    training_df = merge_fire_attributes(training_df, sd_daily, fire_id_col='id_left')

    pts_gdf = gpd.GeoDataFrame(
        training_df,
        geometry=gpd.points_from_xy(training_df['x'], training_df['y']),
        crs='EPSG:32611'
    ).to_crs(epsg=4326)
    training_df['lat'] = pts_gdf.geometry.y
    training_df['lon'] = pts_gdf.geometry.x

    tree = build_station_tree(weather_stations)
    training_df['temp'] = idw_lookup(training_df, weather_data, tree, weather_stations, value_col='temp')
    training_df['humidity'] = idw_lookup(training_df, weather_data, tree, weather_stations, value_col='humidity')
    training_df['wind_speed'] = idw_lookup(training_df, weather_data, tree, weather_stations, value_col='wind_speed')
    training_df['wind_u'], training_df['wind_v'], training_df['wind_dir'] = idw_wind_direction(
        training_df, weather_data, tree, weather_stations
    )

    training_df = last_day_models(training_df)

    feature_cols = [
        'dist_to_current_perim', 'temp', 'wind_speed', 'wind_u', 'wind_v', 'humidity',
        'prev_day_growth', 'prev_day_wind_speed', 'total_area_currently'
    ]

    fold_scores = train_and_evaluate(training_df, feature_cols)
    final_model = train_final(training_df, feature_cols)
    importance_df = get_feature_col_importance(final_model, feature_cols)
    print(importance_df)
