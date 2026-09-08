from sklearn.neighbors import BallTree
import numpy as np




# Functions that determine the nearest possible weather station to the fire perimeter
def build_station_tree(weather_stations, lat_col='lat', lon_col='lon'):
    """ 
    weather_stations: dataframes from all San Diego weather stations
    Builds a BallTree from the station dataset. Calls once, then is reused daily
    """
    coords_radians = np.radians(weather_stations[[lat_col, lon_col]].values)
    tree = BallTree(coords_radians, metric='haversine')
    return tree



def get_nearest_stations(points, tree, weather_stations, k=5, lat_col='lat', lon_col='lon'):
    """
    points: the candidate points
    tree: output of build function
    weather_stations: dataframes from all san diego weather stations
    returns the distances and station ids for the k nearest stations
    """
    # point_coords_radians = np.radians(points_df[lat_col, lon_col].values)
    # distances, indices = tree.query(point_coords_rad, k=k)
    # distances_km = distances * 6371  # convert to kilometers
    # station_ids = weather_stations['station_id'].values[indices]
    # return distances, station_ids
    
def 


