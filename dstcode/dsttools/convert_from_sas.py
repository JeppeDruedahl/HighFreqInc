import os
import warnings
import glob
import pickle

import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from .info import check_and_update_dataset

def draw_random_sub_samples(random_sub_samples,datafolder='data',force_overwrite=False):
    """ draw random sub-sample of pnrs

    Args:

        random_subsamples (dict): keys 'dataset' (str), 'year' (int), and 'seed' to random number generator
        datafolder (str): path to where results are saved        

    """

    # a. load whats on disk
    if not force_overwrite:
        if os.path.exists(f'{datafolder}/random_sub_samples.p'):
            with open(f'{datafolder}/random_sub_samples.p','rb') as p:
                random_sub_samples_disk = pickle.load(p) 
                if random_sub_samples_disk == random_sub_samples:
                    return

    # b. load sas-file
    dataset = random_sub_samples['dataset']
    year = random_sub_samples['year']
    df = pd.read_sas(f'{datafolder}/{dataset}_{year}.sas7bdat')
    
    # c. destring pnr
    df = df[pd.notnull(df.pnr)]
    pnrs = df.pnr.astype('int64')

    # d. random samples
    p1 = pnrs.sample(frac=0.01,random_state=random_sub_samples['seed'])
    p5 = pnrs.sample(frac=0.05,random_state=random_sub_samples['seed'])
    
    # e. save parquet-files
    p1.to_frame().to_parquet(f'{datafolder}/pnrs_p1.parquet')
    p5.to_frame().to_parquet(f'{datafolder}/pnrs_p5.parquet')

    # f. save random sub-sample dictionary
    with open(f'{datafolder}/random_sub_samples.p','wb') as p:
        pickle.dump(random_sub_samples,p) 



def _convert_to_parquet(dataset,year,datafolder):
    """ convert sas dataset in given year to parquet format 

    Args:

        dataset (str): name of dataset
        year (int): year
        datafolder (str): path to where results are saved
 
    """
    
    # a. load sas data
    df = pd.read_sas(f'{datafolder}/{dataset}_{year}.sas7bdat')

    # change PNR to pnr
    if 'PNR' in df.columns:
        df.rename(columns={'PNR':'pnr'}, inplace=True)
        
    # b. destring pnr
    df = df[pd.notnull(df.pnr)]
    
    df['pnr'] = df.pnr.astype('int64')
    df['year'] = df.year.astype('int')

    #stop
    # d. save parquet-file
    df.to_parquet(f'{datafolder}/{dataset}_{year}.parquet')

    print(f'{dataset}_{year} converted succesfully')

def _do_task_convert_to_parquet(dataset,year,datafolder):
    """ check if parquet-file exist or is older than sas-file

    Args:

        dataset (str): name of dataset
        year (int): year
        datafolder (str): path to where results are saved 

    Return:

        (bool): true if parquet file does not exist or is older than sas file

    """

    # a. check if parquet file exists
    parquet_file = f'{datafolder}/{dataset}_{year}.parquet'
    if not os.path.exists(parquet_file):
        return True

    # b. check if parquet file is older
    sas_mtime = os.path.getmtime(f'{datafolder}/{dataset}_{year}.sas7bdat')
    parquet_mtime = os.path.getmtime(parquet_file)
    if parquet_mtime < sas_mtime:
        return True

    return False

def convert_to_parquet(projectid,datasets,datafolder='data',threads=20):
    """ convert all sas datasets to parquet format 

    Args:
    
        projectid (int): the project id
        datasets (dict): dictionary of datasets
        datafolder (str): path to where results are saved  
        threads (int): number of threads to use
    
    """

    # a. test input
    check_and_update_dataset(projectid,datasets)  

    # b. definition of task
    task = lambda dataset,year: _convert_to_parquet(dataset,year,datafolder)
    
    # c. task generator
    tasks = (joblib.delayed(task)(dataset,year)
             for dataset,datasetspec in datasets.items()
             for year in range(datasetspec['years'][0],datasetspec['years'][1]+1) 
             if (datasetspec['overwrite'] or _do_task_convert_to_parquet(dataset,year,datafolder)))
    
    # d. compute in parallel
    joblib.Parallel(n_jobs=threads)(tasks)

