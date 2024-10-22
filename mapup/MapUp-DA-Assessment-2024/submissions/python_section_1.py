from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    for i in range(0, len(lst), n):
        # Reverse the group manually within lst
        group_end = min(i + n, len(lst))
        for j in range((group_end - i) // 2):
            lst[i + j], lst[group_end - 1 - j] = lst[group_end - 1 - j], lst[i + j]
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    dict = {}  
    for string in lst:
        length = len(string)
        if length not in dict:
            dict[length] = []
        dict[length].append(string)

    # Sort the dictionary by keys (lengths)
    dict = dict(sorted(dict.items()))
    return dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    dict = {}  

    def flatten(current_dict, parent_key=''):
        if isinstance(current_dict, dict):
            for key, value in current_dict.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                flatten(value, new_key)
        elif isinstance(current_dict, list):
            for index, item in enumerate(current_dict):
                new_key = f"{parent_key}[{index}]"
                flatten(item, new_key)
        else:
            dict[parent_key] = current_dict

    flatten(nested_dict)
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    from itertools import permutations

    # Use a set to store unique permutations
    unique_perms = set(permutations(nums))

    # Convert each permutation tuple back to a list
    return [list(perm) for perm in unique_perms]
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pass
    import re
    from typing import List
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    return re.findall(date_pattern, text)


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    import polyline
    import math
     def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points 
        on the Earth using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of the first point in decimal degrees.
            lat2, lon2: Latitude and longitude of the second point in decimal degrees.
        
        Returns:
            Distance in meters.
        """
        R = 6371000  # Radius of the Earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Create a list to hold the data
    data = []
    
    # Calculate distances
    previous_coordinate = None
    for lat, lon in coordinates:
        if previous_coordinate is None:
            distance = 0  # Distance for the first point
        else:
            distance = haversine(previous_coordinate[0], previous_coordinate[1], lat, lon)
        
        # Append the data
        data.append({
            'latitude': lat,
            'longitude': lon,
            'distance': distance
        })
        
        previous_coordinate = (lat, lon)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return pd.Dataframe()
    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Step 2: Create the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column in the rotated matrix
            row_sum = sum(rotated_matrix[i])  # Sum of the current row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the current column
            
            # The value for final_matrix[i][j] is the sum of row_sum and col_sum minus the current element
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]
    
    return final_matrix
    


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Create a multi-index from id and id_2
    grouped = df.groupby(['id', 'id_2'])

    # Function to check completeness for each group
    def check_group(group):
        # Convert start and end times to datetime
        group['start'] = pd.to_datetime(group['startDay'] + ' ' + group['startTime'])
        group['end'] = pd.to_datetime(group['endDay'] + ' ' + group['endTime'])
        
        # Get the unique days and times in the group
        unique_days = group['start'].dt.dayofweek.unique()  # 0=Monday, 6=Sunday
        min_time = group['start'].min().time()
        max_time = group['end'].max().time()
        
        # Check if all 7 days are covered
        days_covered = len(unique_days) == 7
        
        # Check if times cover a full 24 hours
        time_covered = min_time == pd.Timestamp('00:00:00').time() and max_time == pd.Timestamp('23:59:59').time()
        
        return not (days_covered and time_covered)

    # Apply the check_group function to each group and return a boolean series
    result = grouped.apply(check_group)
    
    # Convert the result into a boolean series with a multi-index (id, id_2)
    return result.astype(bool)
    
