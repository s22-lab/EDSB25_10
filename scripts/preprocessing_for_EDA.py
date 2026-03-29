# preprocessing_for_EDA.py

def preprocess_for_eda(df):
    """
    Preprocesses the HR dataset for EDA.
    
    Parameters:
    df: pandas DataFrame - the raw HR dataset
    
    Returns:
    pandas DataFrame - the cleaned dataset
    """
    
    # Set the index to EmployeeNumber (if the numbers are unique)
    if 'EmployeeNumber' in df.columns:
        if df['EmployeeNumber'].is_unique:
            # Use EmployeeNumber directly as index
            df = df.set_index('EmployeeNumber')
        else:
            # If not unique, create a new index based on the order of appearance
            df = df.reset_index(drop=True)
            df.index.name = 'EmployeeNumber'
    
    # Handle missing values by filling them with 'Missing' label
    df = df.fillna('Missing')
    
    # Remove duplicate rows (if any)
    df = df.drop_duplicates()
    
    # Remove unvaluable features
    columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    cleaned_for_eda = df.drop(columns=columns_to_drop)
    
    return cleaned_for_eda