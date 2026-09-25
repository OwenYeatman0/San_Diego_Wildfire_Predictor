from shapely.geometry import Point
import numpy as np
import pandas as pd

def merge_fire_attributes(training_df, sd_daily, fire_id_col='id_left'):
    attrs = sd_daily[[fire_id_col, 'event_day', 'dy_ar_km2']].drop_duplicates()
    attrs = attrs.rename(columns={fire_id_col: 'fire_id'})

    merged = training_df.merge(attrs, on=['fire_id', 'event_day'], how='left')

    print('Rows before merge:', len(training_df))
    print('Rows after merge:', len(merged))
    print('Missing dy_ar_km2 after merge:', merged['dy_ar_km2'].isna().sum())

    return merged


def generate_candidate_data(perimeter_geometry, buffer_distance_m = 2000, spacing_m = 250 ):
    buffered = perimeter_geometry.buffer(buffer_distance_m)
    minx, miny, maxx, maxy = buffered.bounds
    
    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)
    
    points = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if buffered.contains(pt):
                points.append(pt)
    return points



def build_pixel_training_data(sd_daily):
    records = []

    for fire_id, group in sd_daily.groupby('id_left'):  # adjust to your actual id column
        group = group.sort_values('event_day').reset_index(drop=True)

        for i in range(len(group) - 1):
            day_n = group.iloc[i]
            day_n1 = group.iloc[i + 1]

            perim_n = day_n.geometry
            perim_n1 = day_n1.geometry

            candidates = generate_candidate_data(perim_n)

            for pt in candidates:
                dist_to_perim = pt.distance(perim_n.boundary)
                burned_next_day = perim_n1.contains(pt)

                records.append({
                    'fire_id': fire_id,
                    'date': day_n1['date'],       # the day being predicted
                    'event_day': day_n1['event_day'],
                    'x': pt.x,
                    'y': pt.y,
                    'dist_to_current_perim': dist_to_perim,
                    'inside_current_perim': perim_n.contains(pt),
                    'burned': int(burned_next_day)   # this is your target label
                })
    return pd.DataFrame(records)

def last_day_models(training_df, group_col ='fire_id', day_col = 'event_day'):
    
    training_df = training_df.sort_values([group_col, day_col]).copy()
    training_df['prev_day_growth'] = training_df.groupby(group_col)['dy_ar_km2'].shift(1)
    training_df['prev_day_wind_speed'] = training_df.groupby(group_col)['wind_speed'].shift(1)
    training_df['total_area_currently'] = training_df.groupby(group_col)['dy_ar_km2'].cumsum() - training_df['dy_ar_km2']
    
    # first day of the fire
    training_df[['prev_day_growth', 'prev_day_wind_speed']] = training_df[['prev_day_growth', 'prev_day_wind_speed']].fillna(0)
    return training_df