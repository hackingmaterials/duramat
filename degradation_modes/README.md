# Degradation modes analysis

## Setup

1. Git clone the duramat repository
```
$ git clone https://github.com/hackingmaterials/duramat.git
```
2. Download the dependencies for the project, run the following command in the duramat/ directory
```
$ pip install -r requirements.txt
```
3. Set the following environment variables for Plotly:
	* 'PLOTLY_USERNAME' : Username from [Plotly](https://plot.ly/accounts/login/?action=signup)
    * 'PLOTLY_KEY' : API key from [Plotly](https://plot.ly/accounts/login/?action=signup)
4. If not provided already, obtain Dirk Jordan's Rd_review_8.csv file and replace the following code at the top of each Jupyter notebook:

Code to be replaced:
```python
sys.path.append('../data/NREL')
import retrieve_data as rd
solar = rd.retrieve_dirks_sheet()
```
Replace with:
```python
solar = pd.read_csv('/path/to/file.csv', encoding='utf-8')
```

Once these steps have been completed, each of the three Jupyter analysis notebooks can be run.


## Data
These notebooks have all been designed around a data set provided by NREL that contains information on solar modules/systems and their observed degradation modes. Here is a summary of the data's various fields focused on in these studies:

* Begin.Year - Installation year of the module/system
* System/module - Indicator between singular or multiple modules
	* The values in this column are: System, Module, String
	* Some Systems are missing No.modules data
	* Some Modules have > 1 in No.modules
	
* Years - Years of exposure/operation
* Begin.Year - Installation year of the module/system
* Ref year - Reference year of the report for module/system
* Climate3 - Simplified categorization of climates for modules
* Cause - Reported degradation modes for module/system
	* Requires proper cleaning into correct categories before use; this functionality is seen in the cleaning process
* No.modules - Number of modules accounted for in report
* Mounting - Mounting type of the module/system

## Documentation
Pydocs for all functions in Jupyter notebooks and utils modules can be found in the docs file.






