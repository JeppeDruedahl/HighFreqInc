import os
import warnings
import glob
import pickle
import joblib

from IPython.display import display
import pandas as pd

warnings.filterwarnings('ignore')

from .info import check_and_update_dataset

def _combine_years(dataset,years,datafolder):
    """ combine all years for given dataset

    Args:

        dataset (str): name of the dataset
        years (list): list of first and last years
        datafolder (str): path to where results are saved 

    """

    # a. find all datasets
    files = [f'{datafolder}/{dataset}_{year}.parquet' 
            for year in range(years[0],years[1]+1)]
    datasets = [pd.read_parquet(f) for f in files]
    
    # b. concatenate
    df = pd.concat(datasets)

    # c. sort values
    df = df.sort_values(['pnr','year'])

    # d. save
    df.to_parquet(f'{datafolder}/{dataset}_all.parquet')

    # e. sub-samples

    # 1 percent sample
    p1 = pd.read_parquet(f'{datafolder}/pnrs_p1.parquet')
    df_p1 = pd.merge(df,p1,how='inner',on='pnr')
    df_p1.to_parquet(f'{datafolder}/{dataset}_p1.parquet')

    # 5 percent sample
    p5 = pd.read_parquet(f'{datafolder}/pnrs_p5.parquet')
    df_p5 = pd.merge(df,p5,how='inner',on='pnr')
    df_p5.to_parquet(f'{datafolder}/{dataset}_p5.parquet')

    print(f'years combined for {dataset} succesfully')

def _do_task_combine_years(dataset,years,datafolder):
    """ check if dataset file exist or is older than all year files

    Args:

        dataset (str): name of dataset
        datafolder (str): path to where results are saved 

    Return:

        (bool): true if parquet file does not exist or is older than all year files

    """

    # a. check if dataset file exists
    dataset_file = f'{datafolder}/{dataset}_all.parquet'
    if not os.path.exists(dataset_file):
        return True
    dataset_mtime = os.path.getmtime(dataset_file)

    # b. check if datset file is older than all year files
    files = [f'{datafolder}/{dataset}_{year}.parquet' 
            for year in range(years[0],years[1]+1)]
    year_files_max_mtime = max([os.path.getmtime(f) for f in files])
    
    if dataset_mtime < year_files_max_mtime:
        return True

    return False

def combine_years(projectid,datasets,datafolder='data',sub_samples=True,threads=20):
    """ combine all years

    Args:
    
        projectid (int): the project id
        datasets (dict): dictionary of datasets
        datafolder (str): path to where results are saved    
        sub_samples (bool,optional): save sub-samples based on previous random draws    
        threads (int): number of threads to use
    
    """

    # a. test input
    check_and_update_dataset(projectid,datasets)    

    # b. definition of task
    task = lambda dataset,years: _combine_years(dataset,years,datafolder)

    # c. task generator
    tasks = ( joblib.delayed(task)(dataset,datasetspec['years'])
              for dataset,datasetspec in datasets.items()
              if (datasetspec['overwrite'] or _do_task_combine_years(dataset,datasetspec['years'],datafolder)))
    
    # d. compute in parallel
    joblib.Parallel(n_jobs=threads)(tasks)