import os, sys, collections

# Import Pandas
import pandas as pd

# Import numpy
import numpy as np

# Ignore warnings for regex with match groups
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

# List of all columns that need to be typed to numeric
numeric_columns = ['System size (kW)', 'Years', 'Begin.Year',
                  'Ref year', 'Elevation (m)', 'Latitude',
                  'Longitude', 'Rd+/Pmax+', 'Isc+', 'Voc+',
                  'FF+', 'Imax+', 'Vmax+', 'No.modules', 'Tilt']

# Dictionary of all corrections to make to State/Town column entries
corrected_locations = {'Phoenix, AZ':'AZ, Phoenix',
                       'CA, LAKEWOOD':'CO, Lakewood',
                       'Toldedo, OH':'OH, Toldedo',
                       'Ledong, Hainan':'Hainan, Ledong'}

# Dictionary of all corrections to make to Cause column entries
mode_dictionary = {
    'Encapsulant discoloration':'Discoloration|EVA browning|EVA browing|yellow color|discoloration|discolor|haze|'+
                                'light discoloration|Discolor|Discolr|1512 EVA discolor|4 discolor|doscolor|37discolor',
    'Major delamination':'Delamination|delamination|delam|6 delam|25 delam|9 delam|'+
                         'bar graph corrosion|Data sheet describes major bar graph corrosion',
    'Minor delamination':'minimal bar graph corrosion',
    'Backsheet insulation compromise':'Ignore',
    'Backsheet other':'backsset|chalking',
    'Internal circuitry discoloration': 'Grid corrosion|'+
                                        'grid corrosion|Grid and cell corrosion|Busbar|interconnect corrosion|'+
                                        'Busbar oxidation|corrosion|Cell disconnect|Interconnect failure|interconnescts|corrosion',
    'Internal circuitry failure':'Ignore',
    'Hot spots':'Hot spot|hot-spots|hotspots|5 hot spots|2 hotspots 1diode|3 hotspots|htospots caused by  busbar problem|'+
                'hotspots casued by cracked cells|1hotspot',
    'Fractured cells':'some cracked cells|cracked glass & cell|crakced cells|cracked cells|hotspots casued by cracked cells|'+
                      'Snail tracks visible',
    'Diode/J-box problem':'J-box|j-box|1 diode|2 hotspots 1diode|diode',
    'Glass breakage':'cracked glass|3 modules cracked glass|cracks|cracked glass & cell|glass cracks|scratches on module surface|'+
                     'Scratch|glass',
    'Permanent soiling':'soiling|Soiling could be potential degradation accelerator for systems with shallow tilt angles|'+
                        'output terminals are corroded and soiled|Soiling|soiling 3-4%|algae',
    'Potential induced degradation':'Ignore',
    'Frame deformation':'frame|warping|30warping'
}

def encode_list(l):
    """
    In order to remove u'' unicode tag from entries in a string array once printed, this function converts a
    given list into a list of strings
    
    Args:
        l (List) - Given list to be converted
    Return:
        List - Converted list of strings
    """
    if not isinstance(l, collections.Sequence) and pd.isnull(l):
        return np.nan
    else:
        result = []
        for s in l:
            s = str(s)
            result.append(s)
        return result

def clean_data(df):   
    """Cleans up data using certain criteria for a given data frame
    
    Will modify this data frame in place
    
    Args:
        df (data frame): Data frame to be modified
    """
    # Changing some values to NaN ('nan', '*', 'Unknown', 'unknown')
    df.replace(to_replace='(\A)(\*|Unknown|unknown|nan)(\Z)', value=np.nan, regex=True, inplace=True)
    
    # Dropping empty rows
    df.dropna(how='all', inplace=True)
    
    # Removing dupliacte rows
    # df.drop_duplicates(keep='first', inplace=True)
    
    # Data typing the columns
    for c in df.columns:
        if c in numeric_columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # Removing inconsistencies from State/Town column
    df.replace(to_replace={'State/Town':corrected_locations}, inplace=True)
    
    # Applying normal mix capitalization to State/Town column
    # df.replace(to_replace=PUT_REGEX_IDENTIFYING_STATE_TOWN, value=PUT_REGEX_APPLYING_CAPITALIZATION)
    
    # Make a new column for the cleaned version of the Cause column
    col_location = df.columns.get_loc('Cause') + 1
    df.insert(col_location, 'Cause (Cleaned)', df['Cause'])
    
    # Remove any spaces on either side of semi-colon seperators in Cause (Cleaned) column for more accuracte string matching
    # Ex. a; b ; c  ->  a;b;c
    df['Cause (Cleaned)'] = df['Cause (Cleaned)'].replace(to_replace='(; )|( ;)', value=';', regex=True)
    
    # Replace all known representations of degredation modes in Cause (Cleaned) column with their designated values
    for mode in mode_dictionary.keys():
        df['Cause (Cleaned)'] = df['Cause (Cleaned)'].replace(
           to_replace='(\A|;)(' + mode_dictionary[mode] + ')(\Z|;)',
           value=';' + mode + ';', regex=True)
        
    # Strip all leading and trailing semi-colons in Cause (Cleaned) column
    df['Cause (Cleaned)'] = df['Cause (Cleaned)'].replace(to_replace='\A;|;\Z', value='', regex=True)
    
    # Transform values in Cause (Cleaned) column into lists of strings, split on the semi-colons
    df['Cause (Cleaned)'] = df['Cause (Cleaned)'].str.split(pat=';')
    
    # Set column location for new dummy variables columns
    col_location = df.columns.get_loc('Cause (Cleaned)') + 1
    
    # Create dummy variable columns for each of the degradation modes
    for mode in mode_dictionary.keys():
        col_name = mode
        df.insert(col_location, col_name, df['Cause'])
        df[col_name] = df[col_name].replace(to_replace='(; )|( ;)', value=';', regex=True)
        df[col_name] = df[col_name].str.contains('(\A|;)(' + mode_dictionary[mode] + ')(\Z|;)', na=False, regex=True)
        df[col_name] = df[col_name].astype(int)
    
    # Clean the System/module column for all instances of "system", change to "System"
    df['System/module'] = df['System/module'].replace(to_replace='system', value='System')
    
    # Clean the System/module column for all instance of "Module" when the No.Modules column has a value > 1
    df.loc[(df['System/module'] == 'Module') & (df['No.modules'] > 1), ['System/module']] = 'System'

def clean_and_print_data(df):   
    """Cleans up data using certain criteria for a given data frame
    
    Will modify this data frame in place
    
    After, prints the cleaned data into a new CSV file
    
    Args:
        df (data frame): Data frame to be cleaned and printed
    """
    clean_data(df)
    
    # Print the cleaned DataFrame to a CSV in the project directory
    df_print = df
    
    for c in df_print.columns:
        if c == 'Cause (Cleaned)':
            df[c] = df[c].apply(encode_list)
        elif c not in numeric_columns:
            df_print[c] = df_print[c].str.encode('utf-8')
            
    df_print.to_csv('../data/NREL/cleaned_dirks_sheet.csv')