import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Create a pivot table to fill in distances between locations
    distance_matrix = pd.pivot_table(df, values='distance', index='from', columns='to', fill_value=0)
    
    # Fill in the diagonal with 0 (distance from a location to itself)
    for location in distance_matrix.index:
        distance_matrix.loc[location, location] = 0

    # Make the distance matrix symmetric
    df = distance_matrix.add(distance_matrix.T, fill_value=0).combine_first(distance_matrix)
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
     unrolled_data = []

    # Iterate through each row and column in the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = df.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a new DataFrame from the list of dictionaries
    df = pd.DataFrame(unrolled_data)
    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    
    import numpy as np
    # Calculate average distance for the reference_id
    ref_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    # Calculate the threshold values
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    # Calculate average distances for all id_start values
    avg_distances = df.groupby('id_start')['distance'].mean()

    # Filter IDs within the specified threshold
    ids_within_threshold = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)]

    # Sort the result and convert to DataFrame
    df = ids_within_threshold.reset_index().sort_values(by='distance')
    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
     rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates by multiplying distance with respective rate coefficients
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define days of the week and time ranges
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Assign start_day and end_day based on unique ids
    df['start_day'] = df['id_start'].apply(lambda x: days_of_week[x % 7])  # Example mapping
    df['end_day'] = df['id_end'].apply(lambda x: days_of_week[x % 7])      # Example mapping
    
    # Define time intervals
    time_ranges = {
        'weekdays': [
            (datetime.time(0, 0), datetime.time(10, 0), 0.8),
            (datetime.time(10, 0), datetime.time(18, 0), 1.2),
            (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8)
        ],
        'weekends': [
            (datetime.time(0, 0), datetime.time(23, 59, 59), 0.7)
        ]
    }

    # Apply time-based toll rates
    for index, row in df.iterrows():
        if row['start_day'] in days_of_week[:5]:  # Monday to Friday
            for start_time, end_time, discount in time_ranges['weekdays']:
                if row['start_time'] >= start_time and row['start_time'] < end_time:
                    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                        df.at[index, vehicle_type] *= discount
                    break
        else:  # Saturday and Sunday
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.at[index, vehicle_type] *= time_ranges['weekends'][0][2]

    # Define start_time and end_time as time objects
    df['start_time'] = datetime.time(0, 0)  # Starting at midnight
    df['end_time'] = datetime.time(23, 59, 59)  # Ending at the last second of the day

    return df
