import os
import warnings
import glob
import pickle
import joblib

import pandas as pd

warnings.filterwarnings('ignore')

from .info import check_and_update_dataset

def _fetch(projectid,dataset,vars,year,datafolder):
    """ fetch dataset with given variables for given year and save it to the datafolder

    Args:
    
        projectid (int): the project id
        dataset (str): name of the dataset
        vars (list): list of variables [(original name, new name),]
        year (int): year
        datafolder (str): path to where results are saved 

    """
    
    assert type(year) == int or year == 'all', 'choose specific year or all of them'
    
    # a. filename
    sasfilename = f'{dataset}_{year}.sas'
    if os.path.exists('logfile'):
        os.remove('logfile')
    
    # b. write sas program
    with open(sasfilename,'w') as sasfile:
        
        sasfile.write(f'libname raw "H:/rawdata/{projectid}/views/";\n')
        sasfile.write(f'libname data "{datafolder}";\n')
        
        sasfile.write('proc sql;\n')
        sasfile.write('create table\n')
        sasfile.write(f'data.{dataset}_{year}\n')
        sasfile.write('as select\n')
        for i,var in enumerate(vars):
            if len(var) == 1:
                sasfile.write(var[0])
            else:
                sasfile.write(f'{var[0]} as {var[1]}')
            if i < len(vars)-1:
                sasfile.write(',')
            sasfile.write('\n')

        if type(year) == int:
            sasfile.write(f'from raw.{dataset}v where {dataset}SourceYear={year};\n')           
        else:
            sasfile.write(f'from raw.{dataset}v;\n')                       
        sasfile.write('quit;\n')
        
    # c. run sas program
    sasexe = '"C:/Program Files/SASHome/SASFoundation/9.4/sas.exe"'
    out = os.system(f'{sasexe} -sysin {sasfilename} -ICON -NOSPLASH')
    if out == 0:
        print(f'{sasfilename} terminated succesfully')
    else:
        warnings.warn(f'{sasfilename} did not terminate succesfully', Warning)

    # d. clean-up
    os.remove(f'{dataset}_{year}.sas')
    if not os.path.isdir('logs'): os.mkdir('logs')
    os.replace(f'{dataset}_{year}.log',f'logs/{dataset}_{year}.log')

def load_datasets_disk(datafolder):
    """ load which datasets are on the disk

    Args:

        datafolder (str): path to where results are saved 

    Return:

        dataset (dict): dictionary of datasets on disk        

    """

    if os.path.exists(f'{datafolder}/datasets.p'):
        with open(f'{datafolder}/datasets.p','rb') as p:
            datasets_disk = pickle.load(p) 
    else:
        datasets_disk = []

    return datasets_disk

def _do_task_fetch(dataset,datasetspec,year,datasets_disk):
    """ check if dataset is on disk for the same variables and year

    Args:

        dataset (str): name of dataset
        datasetspec (dict): dataset specification
        year (int): year
        dataset (dict): dictionary of datasets on disk

    Return:

        (bool): true if dataset not on disk

    """

    # a. check if dataset exists at all
    if not dataset in datasets_disk:
        return True

    # b. check if variables are unchanged
    if not datasetspec['vars'] == datasets_disk[dataset]['vars']:
        return True

    # c. check if year exist
    if not (year >= datasets_disk[dataset]['years'][0] and year <= datasets_disk[dataset]['years'][1]):
        return True

    return False

def fetch(projectid,datasets,datafolder='data',threads=20):
    """ fetch all datasets and save them to the datafolder

    Args:

        projectid (int): the project id
        datasets (dict): dictionary of datasets
        datafolder (str): path to where results are saved
        threads (int): number of threads to use       

    """

    # a. test input
    check_and_update_dataset(projectid,datasets)

    # b. load whats on disk
    datasets_disk = load_datasets_disk(datafolder)

    # c. definition of task
    task = lambda dataset,datasetspec,year: _fetch(projectid,dataset,datasetspec['vars'],year,datafolder)

    # d. task generator
    tasks = (joblib.delayed(task)(dataset,datasetspec,year) 
             for dataset,datasetspec in datasets.items()
             for year in range(datasetspec['years'][0],datasetspec['years'][1]+1) 
             if (datasetspec['overwrite'] or _do_task_fetch(dataset,datasetspec,year,datasets_disk)))

    # e. compute in parallel
    joblib.Parallel(n_jobs=threads)(tasks)

    # f. save
    with open(f'{datafolder}/datasets.p','wb') as p:
        pickle.dump(datasets,p)    

