# Import Pandas
import pandas as pd

def get_total_modules(df, weighted):
    """
        Return the total number of modules in the provided DataFrame
        NOTE: All rows marked as 'System' with a null value in 'No.modules' are left out
        
        Args:
            df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
            weighted (bool): If true, count all modules in a system
                             If false, count a system as one module
        Returns:
            int: Total modules
    """
    if weighted:
        module_count = df[df['System/module'] == 'Module'].shape[0]
        
        specified = df[(df['System/module'] != 'System') | (df['No.modules'].notnull())]
        system_count = specified[specified['System/module'] != 'Module']['No.modules'].sum()
        
        total_modules = module_count + system_count
        return total_modules
    else:
        return df.shape[0]
    
def get_weighted_mode_quantity(df, mode):
    """
    Return the quantity of modules with a reported degradation mode
    NOTE: All modules in systems of modules will be counted as degrading in this calculation
          Furthermore, all rows marked as 'System' with a null value in 'No.modules' are left out
    
    Args:
        df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
        mode (string): Degradation mode to find in the DataFrame
    Returns:
        int: The quantity of modules with specified degradation mode
    """
    single_modules = len(df[(df['System/module'] == 'Module') & (df[mode] == 1)])
    
    specified = df[(df['System/module'] != 'System') | (df['No.modules'].notnull())]
    systems = specified[(specified['System/module'] != 'Module') & (specified[mode] == 1)]['No.modules'].sum()
    
    return single_modules + systems

def get_mode_percentage(df, mode, weighted):
    """
    Return the percentage of modules with a reported degradation mode
    
    Args:
        df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
        mode (string): Degradation mode to find in the DataFrame
        weighted (bool): If true, count all modules in a system as degrading
                         If false, count a system as one degrading module
    Returns:
        float: The percentage of modules with specified degradation mode
    """
    total_modules = get_total_modules(df, weighted)
    if total_modules == 0:
        return 0
    
    if weighted:
        return float(get_weighted_mode_quantity(df, mode)) / total_modules
    else:
        return float(df[mode].sum()) / total_modules
    
