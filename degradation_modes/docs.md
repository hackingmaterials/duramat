# Full code documentation

## utils/process_data.py
```python
def encode_list(l):
    """
    In order to remove u'' unicode tag from entries in a string array once printed, this function converts a
    given list into a list of strings
    
    Args:
        l (List) - Given list to be converted
    Return:
        List - Converted list of strings
    """
    
def clean_data(df):   
    """Cleans up data using certain criteria for a given data frame
    
    Will modify this data frame in place
    
    Args:
        df (data frame): Data frame to be modified
    """

def clean_and_print_data(df):   
    """Cleans up data using certain criteria for a given data frame
    
    Will modify this data frame in place
    
    After, prints the cleaned data into a new CSV file
    
    Args:
        df (data frame): Data frame to be cleaned and printed
    """
```

## utils/degradation_utils.py
```python
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

```

## classification_models_analysis.ipynb
```python
def get_classification_model_data(df, modes):
    """
    Build a DataFrame that holds all of the information for building naive bayes models
    
    Args:
        df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
        modes (list of string): List of degradation modes to add as dummy variable columns
    
    Returns:
        DataFrame: Holds the features and target necessary for building naive bayes
    """
    
def visualize_tree(tree, class_name, feature_names, dot_name, png_name):
    """Create tree png using graphviz
    NOTE: Will export trees into an output folder located on the same level as this notebook

    Args:
        tree -- scikit-learn DecsisionTree
        feature_names -- list of feature names
    """
    
def generate_decision_tree_models(df, modes, features, params_dict):
    """
    Generate dot and png files for decision tree models for each specified degradation mode
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        modes (list of string): List of degradation modes to build models for
        features (list of string): List of features (columns in the dataframe) to use for the model
        params_dict (dict): A dictionary containing the min_samples_leaf and max_depth parameters of each degradation mode
                            Format {Key: Degradation mode,Value:{Key:min_samples_leaf,Value:int}
                                                                {Key:max_depth,Value:int}}
    """

def dt_grid_search(df, mode, feature_names, lr, dr):
    """
    Perform grid search on the given DataFrame to find the optimal parameters for each decision tree
    NOTE: Leaf range is searched in a range of 2000 to 3000 at intervals of 100
          Depth range is searched in a range of 1 and 9 at intervals of 1
          
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        mode (string): Degradation mode to run grid search for
        feature_names (list of string): List of features (columns in the dataframe) to use for the model
        lr (list of integers or None): Integers to try for the grid search for the min_samples_leaf param
                                       Defaulted to 2000-3000 for every 100
        dr (list of integers or None): Integers to try for the grid search for the max_depth param
                                       Defaulted to 1-9 for every 1
    """

def dt_cross_val_score(df, m, feature_names, msl, md):
    """
    Performs a 10-fold cross validation on a decisition tree built with the specified parameters
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        m (string): Degradation mode to build the model on
        msl (int): min_samples_leaf parameter for the model
        md (int): max_depth pramater for the model
    Returns:
        float: The average of accuracies from each of the 10 folds
    """
    
def generate_naive_bayes_models(df, features, modes):
    """
    Build a dictionary that holds all Bernoulli Naive Bayes models for specified degradation modes
    NOTE: This function will print the score provided by Scikit of each model
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        features (list of string): List of desired columns to include in the model as features
        modes (list of string): List of degradation modes to build models for
    Returns:
        dict: Dictionary to hold all Bernoulli Naive Bayes models
              Format is {Key:Degradation mode, Value:respective Naive Bayes model}
    """
```

## climate_and_mounting_analysis.ipynb
```python
def get_mounting_graph_data(df, modes, mountings, weighted):
    """
    Generate a DataFrame that holds percentage of modules with degradation modes based on mounting types
    NOTE: If weighted, all rows marked as 'System' with a null value in 'No.modules' are ignored
    
    Args:
        df (DataFrame) - A cleaned DataFrame (through the clean_data function) that holds module degradation data
        modes (List of String) - List of Strings that contain all of the degradation modes to search for in df
        mountings (List of String) - List of Strings containing the mounting types to select from
        weighted (bool): If true, count all modules in a system as degrading
                         If false, count a system as one degrading module
    Returns:
        DataFrame - Table holding weighted percentages for given mountings
    """
    
def generate_mounting_graph(df, modes, mountings, filename, plotname):
    """
    Generate a grouped bar chart to visualize the degradation mode percentages found in modules with
    different mounting types
    
    Args:
        df (DataFrame) - DataFrame holding percentages of degraded modules
        modes (List of Strings) - List of all modes that were checked for in generating df
        mountings (List of Strings) - List of all mountings that were filtered with in generating df
        filename (string) - Name of the file to place the generated graph in
        plotname (string) - Title appearing at the top of the graph
    """

def generate_mounting_categorical_dot_plot(modes, rack, one_axis, roof, single_axis):
    """
    Create a categorical dot plot showing the percentage of degradation for four different mountings
    
    Args:
        modes (list of string): List of all degradation modes to be included
        rack (list of float): List of percentages of degradation for rack (in order found in 'modes')
        one_axis (list of float): List of percentages of degradation for one_axis (in order found in 'modes')
        roof (list of float): List of percentages of degradation for roof (in order found in 'modes')
        single_axis (list of float): List of percentages of degradation for single_axis (in order found in 'modes')
    """
```

## correlating_modes_analysis.ipynb
```python
def get_mode_correlation_percent(df, mode_1, mode_2, weighted):
    """
    Return the percent of rows where two modes are seen together
    
    Args:
        df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
        mode_1 (string): Degradation mode to find in the DataFrame in pairing with mode_2
        mode_2 (string): Degradation mode to find in the DataFrame in pairing with mode_1
        weighted (bool): If true, count all modules in a system as degrading
                         If false, count a system as one degrading module
    Returns:
        float: The percentage of modules with both specified degradation modes
    """
    
def get_heatmap_data(df, modes, weighted):
    """
    Returns a DataFrame used to construct a heatmap based on frequency of two degradation modes appearing together
    
    Args:
        df (DataFrame): A *cleaned* DataFrame containing the data entries to check modes from
        modes (List of String): A list of all modes to check for in the DataFrame
        weighted (bool): If true, count all modules in a system as degrading
                         If false, count a system as one degrading module
    Returns:
        heatmap (DataFrame): DataFrame containing all of degradation modes correlation frequency results
    """
```