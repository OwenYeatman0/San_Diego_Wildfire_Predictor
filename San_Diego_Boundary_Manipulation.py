import geopandas as gpd
from pyproj import Transformer

counties = gpd.read_file('us_counties_fips.json')
san_diego = counties[(counties['STATE'] == '06') &
                     (counties['NAME'] == 'San Diego')]


# get the boundary box from the us county file
bbox = san_diego.total_bounds  # returns (minx, miny, maxx, maxy)
print(bbox)




    
def change_boundary_coords():
    """figures out the coordinate system being used for each dataset (sinosodial
    and Modis) and converts in order to fit to the San Diego boundary box"""
    # code to determine what geo-coordinate system is being used
    info = gpd.read_file(
    r'C:\Users\OWENY\Downloads\fired_uscan_to2021121_daily_gpkg\fired_uscan_to2021121_daily_fixed.gpkg', rows=1)
    print(info)

    # transform bbox coordinates from Sinosoidal to MODIS
    minx, miny, maxx, maxy = -117.59588, 32.34156, -116.08109, 33.505019
    

    # MODIS Sinosodial Projection
    modis_sinu = (
        'proj = sinu +lon_0 = 0 +x_0 =0 +y_0 = 0 '
        '+R = 6371007.181 +units=m +no_defs '
        )
    transformer = Transformer.from_crs('EPSG:4326', modis_sinu, always_xy=True)


    # Make sure every corner is transformed
    corners_x = []
    corners_y = []
    
    for lon, lat in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
        x, y = transformer.transform(lon, lat)
        corners_x.append(x)
        corners_y.append(y)

    bbox_projection = (min(corners_x), min(corners_y),
                   max(corners_x), max(corners_y))
    print(bbox_projection)
    return bbox_projection



def remove_other_counties(bbox_projection, san_diego):
    """changes the rough boundary box and fits it to the exact county lines"""
    # uses the bounding box generated
    San_Diego_Fires = gpd.read_file(
        r'C:\Users\OWENY\Downloads\fired_uscan_to2021121_daily_gpkg\fired_uscan_to2021121_daily_fixed.gpkg', bbox=bbox_projection)
    print(len(San_Diego_Fires))


    # convert to lat/lon for easier manipulation
    San_Diego_Fires = San_Diego_Fires.to_crs(epsg=4326)


    # remove edges clipping into other counties
    san_diego = san_diego.to_crs(San_Diego_Fires.crs)
    sd_daily = gpd.sjoin(San_Diego_Fires, san_diego,
                     how='inner', predicate='intersects')
    San_Diego_Fires = sd_daily
    print(len(sd_daily))
    print('Unique fires:', sd_daily['id_left'].nunique())




remove_other_counties(change_boundary_coords(), san_diego)