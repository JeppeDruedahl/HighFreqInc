import warnings

from IPython.display import display
import pandas as pd

warnings.filterwarnings('ignore')

def load_info(projectid):
    """ load information from the documentation

    Args:

        projectid (int): the project id

    Return:

        allyearsdict (dict): dictionary with dataset names as keys and list of first and last year as values
        allvarsset (set): set of all available datasets and variables
        years (pd.df): dataframe with datasets and years
        allvars (pd.df): dataframe with datasets and variables

    """
    
    # a. load excel file for years
    years = pd.read_excel(f'H:/Documentation/{projectid}/{projectid}.Years.xlsx')

    # b. create dictionary for years
    allyearsdict = {dataset:years.loc[years.Register == dataset, ['From','To']].values.tolist()[0] 
        for dataset in years.Register.unique()}

    # c. load excel file for variables
    allvars = pd.read_excel(f'H:/Documentation/{projectid}/{projectid}.Variables.xlsx',sheet_name='AllVariables')

    # d. create set of combinations of datasets and variables
    allvars = allvars.loc[allvars.X == 'x',['Register','Name']]
    allvarsset = set()
    for register,name in zip(allvars.Register,allvars.Name):
        allvarsset.add((register.upper(),name.upper()))
    for register in allvars.Register:
        allvarsset.add((register.upper(),f'{register.upper()}SOURCEYEAR'))

    return allyearsdict,allvarsset,years,allvars

def show_variables(years,allvars,dataset):
    """ show years and variables in dataset

    Args:

        years (pd.df): dataframe with datasets and years
        allvars (pd.df): dataframe with datasets and variables

    """

    display(years[years.Register == dataset])
    display(allvars[allvars.Register == dataset])

def check_and_update_dataset(projectid,datasets):
    """ check that all datasets are correctly specified and 

    Args:

        projectid (int): the project id
        datasets (dict): dictionary of datasets 

    """

    # a. load infomation
    allyearsdict,allvarsset,_years,_allvars = load_info(projectid)

    # b. make assertions
    for dataset,datasetspec in datasets.items():

        # i. all keys
        assert 'vars' in datasetspec
        assert 'years' in datasetspec     
        assert 'overwrite' in datasetspec

        # ii. variables available
        for var in datasetspec['vars']:
            x = (dataset.upper(),var[0].upper())
            assert x in allvarsset,x
        
        # iii. years available
        assert type(datasetspec['years']) == list or datasetspec['years'] == 'all'
        years = allyearsdict[dataset.upper()]

        if datasetspec['years'] == 'all':         

            datasetspec['years'] = years
            
        else:

            # update for first and last
            if datasetspec['years'][0] == 'first':
                datasetspec['years'][0] = years[0]
            elif datasetspec['years'][1] == 'last':
                datasetspec['years'][1] = years[1]
            
            assert type(datasetspec['years'][0]) == int
            assert datasetspec['years'][0] >= years[0],(datasetspec['years'],years)
            
            assert type(datasetspec['years'][1]) == int
            
            # this assertion could be uncommented once DST data is updated
            # assert datasetspec['years'][1] <= years[1],(datasetspec['years'],years)