# Data pre-processing
import pandas as pd
import numpy as np

# 1. Data integration 
def merge_select(accidents, accident_locations):
    merged_accidents_with_location = pd.merge(accidents, accident_locations, on="ACCIDENT_NO", how="left")
    columns = ['ACCIDENT_NO', 'ROAD_GEOMETRY', 'ROAD_GEOMETRY_DESC', 'SEVERITY', 'SPEED_ZONE', 'ROAD_NAME', 'ROAD_TYPE', 'ROAD_NAME_INT', 'ROAD_TYPE_INT']
    accidents_with_location = merged_accidents_with_location[columns]
    return accidents_with_location

# 2. Validate data quality
def validate_speed_zone(accidents_with_location):
    min_speed = 30
    max_speed = 110
    
    # Handle any non-validated data
    accidents_with_location['SPEED_ZONE'] = accidents_with_location['SPEED_ZONE'].apply(lambda x:x if min_speed <= x <= max_speed else np.nan)    
    accidents_with_location['SPEED_ZONE'] = accidents_with_location.groupby('ROAD_TYPE')['SPEED_ZONE'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

    mode = accidents_with_location['SPEED_ZONE'].mode()
    accidents_with_location['SPEED_ZONE'] = accidents_with_location['SPEED_ZONE'].fillna(mode[0])
    return accidents_with_location

# 3. Transformation
def discretise_speed_zone(accidents_with_location):    
    
    # Manual binning
    """
    Speed Zones:
        1: 30-40
        2: 50-60
        3: 70-80
        4: 90-110
    """

    bins = [0, 41, 61, 81, 111]
    accidents_with_location['SPEED_ZONE_CAT'] = pd.cut(accidents_with_location['SPEED_ZONE'], bins=bins, labels=['1', '2', '3', '4'])
    return accidents_with_location

# 4. Encoding
def encoding_road_type(accidents_with_location):
    
    # Seperate road type by location
    rural_road_type = ['CHASE', 'DRIVE', 'PARK', 'DRIVEWAY', 'ALLEY', 'CIRCUIT', 'WAY' ]
    intercity_road_type = ['BYPASS', 'PARKWAY', 'FREEWAY', 'HIGHWAY', 'TUNNEL', 'THROUGHWAY']
    residential_road_type = ['CIRCLE', 'SQUARE', 'CROSS', 'ROW', 'CRESENT', 'BAY', 'PLACE', 'PLAZA', 'COURT', 'BOULEVARD', 'AVENUE', 'STREET', 'ROAD', 'BEND']
    def map_road_type(road_type):
        if road_type in rural_road_type:
            return 0
        elif road_type in intercity_road_type:
            return 1
        elif road_type in residential_road_type:
            return 2
        # Unknown or not normal road type
        else:
            return 3
    
    accidents_with_location['ROAD_TYPE_CAT'] = accidents_with_location['ROAD_TYPE'].apply(map_road_type)
    accidents_with_location['ROAD_TYPE_INT_CAT'] = accidents_with_location['ROAD_TYPE_INT'].apply(map_road_type)
    
    return accidents_with_location

accidents = pd.read_csv('./datasets/accident.csv')
accident_locations = pd.read_csv('./datasets/accident_location.csv')

accidents_with_location = merge_select(accidents, accident_locations)
validate_speed_zone(accidents_with_location)
discretise_speed_zone(accidents_with_location)
encoding_road_type(accidents_with_location)

accidents_with_location.to_csv("accidents_with_location.csv", index=False)

