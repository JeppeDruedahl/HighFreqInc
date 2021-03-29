# -*- coding: utf-8 -*-
"""HighFreqIncProcess

Calculate moments for, simulates and estimates a high frequency income process.

"""

##############
# 1. imports #
##############

import os
import psutil
import glob
import warnings
import time
import ctypes as ct
import joblib
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import boolean, int64, double, njit, prange

from consav import ModelClass, jit # baseline model class

# local modules
import theoretical_moments
import moments
import simulate
import estimate
from labels import latex

############
# 2. model #
############

class HighFreqIncProcessClass(ModelClass):
    
    ############
    # 3. setup #
    ############
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
    
        # b. other attributes
        self.other_attrs = ['latexname','specs','theta','data','datamoms','moms','est']

        # d. list not-floats for safe type inference
        self.not_floats = ['simT','simQ','simN','Nmoms','Ndata','use_equal_weighting','use_theoretical','est_max_iter','est_max_iter_ini',
                           'N_guesses','N_identification','N_multistart','seed','seed_sim','do_d1ky','do_d12ky','kmax','Nomegas_cond']

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. horizon
        par.simT = 12*30 # number of simulation periods
        par.simN = 100_000 # number of individuals
        
        # b. frequency
        par.p_psi = 0.05
        par.p_xi = 0.08
        par.p_phi = 0.10
        par.p_eta = 0.15
        
        # c. persistence
        par.rho = 0.90

        # d. dispersion
        par.sigma_psi = 0.15
        par.sigma_xi = 0.20
        par.sigma_phi = 0.05
        par.sigma_eta = 0.02
        par.sigma_epsilon = 0.00

        # e. means
        par.mu_phi = 0.02
        par.mu_xi = 0.30
        par.mu_eta = 0.00

        # f. estimation
        par.Nmoms = 0 # number of moments
        par.Ndata = 0 # number of observational units in the data
        
        par.use_equal_weighting = False # use equal weighting
        par.use_theoretical = True # use theoretical formulas
        
        par.est_method = 'Nelder-Mead+BFGS' # method for numerical optimization
        par.est_tol = 1e-8 # tolerance numerical optimization
        par.est_max_iter = 10_000 # maximum number of iteration for numerical optimization
        par.est_max_iter_ini = 500 # maximum number of iteration for numerical optimization in initial step (if used)
        
        par.N_guesses = 500 # number of values of all parameters when drawn completely at random
        par.wtrue = 0.99 # weight on true paramer values in multi-start

        par.N_identification = 500 # number of parameter sets drawn when testing identification
        par.N_multistart = 50 # number of multi-starts used when estimating

        # g. misc
        par.seed = 2019 # seed for random numbers (except simulation)
        par.seed_sim = 17 # seed used for simulation (only to check)
        
        par.do_d1ky = True # monthly growth rates
        par.do_d12ky = True # 12-month growth rates
        par.kmax = 6 # growth rates for 12,24,...,12*kmax are calculated 
        par.wmin = 1e-6 # faster computation of conditional CDF

        par.omega_cond_midrange = 0.01 # range for conditional shares
        par.omegas_cond = np.array([-0.1,0.05,-0.01,0.01,0.05,0.10]) # range for conditional shares
        
        # h. empty containers
        par.Nomegas_cond  = 0
        par.omegas_cdf = np.zeros((0,))

        par.cov_moms = np.zeros((0,0))
        par.W = np.zeros((0,0))
        par.Ypsilon = np.zeros((0,0))

        par.w_psi = np.zeros((0,))
        par.w_xi = np.zeros((0,))
        par.w_phi = np.zeros((0,))
        par.w_eta = np.zeros((0,))

    def allocate(self):
        """ allocate simluation arrays """

        par = self.par
        sim = self.sim

        par.Nomegas_cond = par.omegas_cond.size

        sim.y = np.zeros((0,0))
        sim.p = np.zeros((0,0))
        sim.z = np.zeros((0,0))

        sim.p_psi = np.zeros((0,0))
        sim.p_xi = np.zeros((0,0))
        sim.p_phi = np.zeros((0,0))
        sim.p_eta = np.zeros((0,0))
        
        sim.psi_raw = np.zeros((0,0))
        sim.xi_raw = np.zeros((0,0))
        sim.phi_raw = np.zeros((0,0))
        sim.eta_raw = np.zeros((0,0))
        sim.epsilon_raw = np.zeros((0,0))    
        sim.p0_raw = np.zeros((0,))    
        
        sim.d1ky = np.zeros((0,0,0))
        sim.d1kyw = np.zeros((0,0,0))
        sim.d1ky_lag = np.zeros((0,0,0))

        sim.d12ky = np.zeros((0,0,0))
        sim.d12kyw = np.zeros((0,0,0))
        sim.d12ky_lag = np.zeros((0,0,0))

    def load_data(self,datafolder='data',sample='all',do_cov=True,do_print=False):
        """ load actual data  """

        self.datamoms = {}

        # a. load moments
        self.par.Ndata = np.genfromtxt(f'{datafolder}/N_{sample}.txt',delimiter=',').reshape((1,))[0]
        datamoms_df = pd.read_excel(f'{datafolder}/moments_{sample}.xls')

        # b. load bootstrap
        files = glob.glob(f'{datafolder}/moments_bootstrap*_{sample}.txt')
        moms_boots = np.concatenate([np.genfromtxt(file,delimiter=',') for file in files],axis=0)

        # c. convert to dicts
        datamoms_dict = {}
        moms_boots_dict = {}

        # convert string to number
        def num(s):
            try:
                return int(s)
            except ValueError:
                return float(s)

        # loop through data
        iboot = 0
        for row in datamoms_df.iterrows():
            
            # i. unpack
            momname = row[1]['momname']
            if type(row[1]['args']) is int:
                args = row[1]['args']
            else:
                args = row[1]['args'].replace('(','').replace(')','').split(',')
                args = (num(args[0]),num(args[1]))
            
            # ii. save value
            self.datamoms[(momname,args)] = row[1]['value']

            # iii. load bootstraps
            if not 'cdf' in momname: # cdf not bootstraped
                moms_boots_dict[(momname,args)] = moms_boots[:,iboot]
                iboot += 1

        # d. find chosen moments
        for momname,infolist in self.specs.items():
            for info in infolist:
                
                # i. unpack
                args = info['args']
                key = (momname,args)

                # ii. weight
                if 'weight' in info:
                    self.datamoms[(key,'weight')] = info['weight']
                else:
                    self.datamoms[(key,'weight')] = 1

                # iii. print
                if do_print:
                    str_key = f'{momname} [{args}]'
                    print(f'{str_key:40s}: {datamoms_dict[key]:7.4f}')    

                # iv. load boostraped moments
                if 'cdf' not in momname: # cdf not bootstraped
                    self.datamoms[(key,'var')] = np.var(moms_boots_dict[key])  

                if not do_cov or 'cdf' in momname: continue

                # compute covariance
                for momname_cov,infolist_cov in self.specs.items():
                    for info_cov in infolist_cov:

                        if 'cdf' in momname_cov: continue 

                        # o. unpack
                        args_cov = info_cov['args']
                        key_cov = (momname_cov,args_cov)

                        # oo. selected                            
                        a = moms_boots_dict[key]
                        b = moms_boots_dict[key_cov]
                        
                        # ooo. covariance matrix
                        if key == key_cov:
                            self.datamoms[(key,key_cov,'cov')] = np.var(a)
                        else:
                            self.datamoms[(key,key_cov,'cov')] = np.cov(a,b)[0,1]

    def load_datamoms(self,data):
        """ load data moments """
        
        # a. copy moments
        self.datamoms = data.moms.copy()

        # b. add weights
        for momname,infolist in self.specs.items():
            for info in infolist:

                # i. unpack
                args = info['args']
                key = (momname,args)
                
                # ii. weight
                if 'weight' in info:
                    self.datamoms[(key,'weight')] = info['weight']
                else:
                    self.datamoms[(key,'weight')] = 1

    def set_specs(self,ws=''):
        """ set moment specification """

        # a. lags for moment specifications
        ks = [1,2,3,4,5,6]

        # b. x-values for unconditional cdf and conditional cdf (i.e. midrange)
        etas = [0.50,0.40,0.30,0.20,0.10,0.05,0.04,0.03,0.02,0.01,5*1e-3,1e-3,1e-4]
        etas = np.concatenate((etas,-np.flip(etas)))

        etas_midrange = [0.50,0.30,0.10,0.05,0.01,1e-3,1e-4]
        etas_midrange = np.concatenate((etas_midrange,-np.flip(etas_midrange)))

        etas_d1ky = etas_midrange
        etas_d1ky_midrange = etas_midrange

        # c. construct sub-specs
        _base = { 
            
            f'mean_d12ky{ws}':[{'args':k} for k in ks],
            f'var_d12ky{ws}':[{'args':k} for k in ks],
            f'auto_cov_d12y{ws}12l':[{'args':l} for l in [1,2,3,4,5]],
            f'frac_auto_cov_d12y{ws}1l': [{'args':l} for l in range(1,12)], 
            
        }

        _higherorder = {
            f'kurt_d12ky{ws}':[{'args':k} for k in ks],    
        }

        _shares = {
            
            'leq_d1ky': [{'args':(1,eta)} for eta in etas],
            'leq_d1ky_midrange': [{'args':(k,eta)} for k,eta in it.product([1],etas_d1ky_midrange)],
            
            'leq_d12ky': [{'args':(k,eta)} for k,eta in it.product(ks,etas)],
            'leq_d12ky_midrange': [{'args':(k,eta)} for k,eta in it.product([1],etas_midrange)],
            
        }

        _level = { 
            'var_y_d12_diff': [{'args':k} for k in [1,2,3,4,5,6]],
            'cov_y_y_d12_diff': [{'args':k} for k in [1,2,3,4,5]]
        }

        # d. set specs
        self.specs = {**_base, **_higherorder, **_shares, **_level}

    def set_theta(self,use_closed_form=False):
        """ set estimation parameter specification """

        self.theta = {

            'p_phi': {'guess':np.nan,'lower':0.0001,'upper':0.40},
            'p_psi': {'guess':np.nan,'lower':0.0001,'upper':0.40},
            'p_eta': {'guess':np.nan,'lower':0.0001,'upper':0.40},    
            'p_xi': {'guess':np.nan,'lower':0.0001,'upper':0.40},    

            'sigma_phi': {'guess':np.nan,'lower':0.0,'upper':0.5,'closed_form':use_closed_form},    
            'sigma_psi': {'guess':np.nan,'lower':0.0,'upper':0.5,'closed_form':use_closed_form},    
            'sigma_eta': {'guess':np.nan,'lower':0.0,'upper':2.0,'closed_form':use_closed_form},    
            'sigma_xi': {'guess':np.nan,'lower':0.0,'upper':2.0},
            
            'rho': {'guess':np.nan,'lower':0.0,'upper':1.0,'closed_form':use_closed_form},

            'mu_phi': {'guess':np.nan,'lower':-0.05,'upper':0.05,'closed_form':use_closed_form},
            'mu_xi': {'guess':np.nan,'lower':-2.00,'upper':2.00},
            
        }

    ###############    
    # 4. simulate #
    ###############

    setup_simulation = simulate.setup
    
    def simulate(self):
        with jit(self) as model:
            simulate.all(model)

    ##############    
    # 5. moments #
    ##############

    def calc_moments(self,do_timing=False):
        """ calculate moments """
        
        self.moms = {}

        for momname,infolist in self.specs.items():

            for info in infolist:
                args = info['args']

                if self.par.use_theoretical:
                    module = 'theoretical_moments'
                else:
                    module = 'moments'

                # i. skip if already calculated
                if (momname,args) in self.moms: continue
                
                # ii. potentially fast calculations    
                t0 = time.time()
                
                found = getattr(eval(module),'fast')(self,momname,args)   
                
                t1 = time.time()
                if do_timing and found: print(f'{momname:30s}: {t1-t0:2f} secs')

                if found: continue                    

                # iii. calculate   
                t0 = time.time()

                with jit(self) as model:

                    if args == None: # no arguments                
                        model.moms[(momname,args)] = getattr(eval(module),momname)(model.par,model.sim)                
                    elif np.isscalar(args): # single argument                
                        model.moms[(momname,args)] = getattr(eval(module),momname)(model.par,model.sim,args)                
                    else: # multiple arguments                
                        model.moms[(momname,args)] = getattr(eval(module),momname)(model.par,model.sim,*args)

                t1 = time.time()
                if do_timing: print(f'{momname:30s}: {t1-t0:2f} secs')

    def show_moments(self):
        """ show moments """

        for key in self.moms.keys():
            
            mom = self.moms[key]   
            str_key = f'{key[0]} [{key[1]}]'
            print(f'{str_key:40s}: {mom:12.8f}')

    def compare_moments(self,other=None,str_self='model',str_other='data'):
        """ compare model moments with data moments or other model moments """
        
        for i,key in enumerate(self.moms.keys()):
            
            mom_self = self.moms[key]   
            
            if other is None:
                mom_other = self.datamoms[key]
            else:
                mom_other = other.moms[key]
                
            str_key = f'{key[0]} [{key[1]}]'
            print(f'{str_key:40s}: {str_self} = {mom_self:7.4f}, {str_other} = {mom_other:7.4f} ',end='')
            if self.par.use_equal_weighting:
                print(f'diff = {mom_self-mom_other:7.4f}')
            else:
                print(f'contribution = {(mom_self-mom_other)**2*self.par.W[i,i]:7.4f}')

    ###############    
    # 6. estimate #
    ###############

    def prepare_estimation(self):
        """ calculate Nmoms, cov_moms, W and Ypsilon """

        # a. calculate moments
        self.calc_moments()

        # b. count moments
        self.par.Nmoms = len(self.moms)

        # c. extract covariances
        self.par.cov_moms = np.zeros((self.par.Nmoms,self.par.Nmoms))
        self.par.W = np.zeros((self.par.Nmoms,self.par.Nmoms))
        
        for i,key_i in enumerate(self.moms.keys()):
            for j,key_j in enumerate(self.moms.keys()):
                
                # i. full
                if (key_i,key_j,'cov') in self.datamoms:
                    self.par.cov_moms[i,j] = self.datamoms[(key_i,key_j,'cov')]
                else:
                    self.par.cov_moms[i,j] = np.nan

                # ii. diagonal
                if i == j:
                    
                    if self.par.use_equal_weighting:
                    
                        self.par.W[i,j] = 1
                    
                    else:
                        
                        self.par.W[i,j] = 1/(self.par.cov_moms[i,j]*self.par.Ndata)

                        key_weight = (key_i,'weight')
                        if key_weight in self.datamoms:
                            self.par.W[i,j] *= self.datamoms[key_weight]

        # d. compute Ypsilon        
        self.par.Ypsilon = self.par.Ndata*self.par.cov_moms

    def update_from_guess_dict(self,guess_dict):
        """ update guess and values from dictionary returned by find_initial_guess* """

        for name,spec in self.theta.items():
            value = guess_dict[name]
            setattr(self.par,name,value)
            spec['guess']= value

    def draw_initial_guesses(self):
        """ draw unscaled initial guesses """

        np.random.seed(self.par.seed)
        self.initial_guesses = np.random.uniform(size=(len(self.theta),self.par.N_guesses))

    def find_initial_guess(self,do_print=False,true=None):
        """ find initial guesses using random search """

        t0 = time.time()

        # a. random guesses 
        random_guesses = {}
        for i,(name,spec) in enumerate(self.theta.items()):
            random_guesses[name] = spec['lower'] + self.initial_guesses[i,:]*(spec['upper']-spec['lower'])

        # b. grid search
        initial_guesses = {} 
        names_closed_form = [key for key,value in self.theta.items() if ('closed_form' in value and value['closed_form'])]        
        for i in range(self.par.N_guesses):

            # i. set parameter
            for name in self.theta.keys():

                guess = random_guesses[name][i]
                if not true is None:
                    guess = true[name]*self.par.wtrue + (1.0-self.par.wtrue)*guess

                setattr(self.par,name,guess)

            # ii. use closed form
            if len(names_closed_form) > 0:
                estimate.update_closed_form(self,names_closed_form)

            # iii. calculate objective
            obj = self.calc_obj()
            
            # iv. record  
            initial_guesses[(obj,i)] = {key:getattr(self.par,key) for key in self.theta.keys()}

        # c. find best
        initial_guesses = sorted(initial_guesses.items(), key=lambda k: k[0])
        
        best_guess = initial_guesses[0]
        best_obj = best_guess[0][0]

        self.update_from_guess_dict(best_guess[1])

        # d. print
        if do_print:

            t1 = time.time()
            minutes = np.int((t1-t0)/60)
            seconds = np.int(t1-t0-minutes*60) 

            print(f'initial guess found in {minutes}min {seconds}s')
            print(f'obj. = {best_obj:.8f}')

    def estimate(self,do_se=False,do_print=False):
        """ estimate model """

        self.est = {}

        # a. estimate
        estimate.estimate(self,do_print=do_print)
            
        # b. standard errors
        if do_se: 
            estimate.std_error(self)
        else:
            for name in self.theta.keys(): self.est[(name,'se')] = np.nan

    def calc_obj(self):
        """ calculate objective value """
    
        return estimate.calc_obj(self,W=self.par.W)

    def est_vs_true(self,data):
        """ compare estimated parameters with true parameters """

        for name in self.theta.keys():
            true = getattr(data.par,name)
            print(f'{name:14s}: {self.est[name]:7.4f} [true: {true:7.4f}]')   

    def est_results(self):
        """ print estimation results to screen and table """

        # a. print to screen
        for name in self.theta.keys():
            
            est = self.est[name]
            se = self.est[(name,'se')]
            print(f'{name:14s} estimated to be {est:7.4f} ({se:7.4f})')

        print('')

        # b. print to latex table
        if not os.path.isdir('figs'):
            os.mkdir('figs')

        with open(f'figs\\est_{self.name}.tex','w') as file:

            file.write('\\begin{tabular}{lccc} \n')
            file.write('\\toprule \n')
            file.write('Parameter & & Estimate & S.E. \\\\ \n')
            file.write('\\midrule \n')
            for name in self.theta.keys():
                
                # i. name
                if name in latex:
                    latex_name = latex[name](self.par)
                else:
                    latex_name = 'Unknown, '
                name_1 = latex_name[0]
                name_2 = latex_name[1]
                
                # ii. estimate and standard deviation
                est = self.est[name]
                se = self.est[(name,'se')]

                # iii. print row
                file.write(f'{name_1} & {name_2} & {est:7.4f} & {se:7.4f} \\\\ \n')
            
            file.write('\\bottomrule\n')
            file.write('\\end{tabular}\n')

    def multistart_estimation(self,n_jobs=50,load=False,save=True):
        """ run multi-start estimation"""

        self.est = {}
        model_dict = self.as_dict()
        
        # folder for saving
        if not os.path.exists(f'saved/{self.name}_multistart'):
            os.makedirs(f'saved/{self.name}_multistart')

        t0 = time.time()
        psutil.cpu_percent(interval=None)

        # a. random numbers
        np.random.seed(self.par.seed)

        initial_guessess = np.zeros((self.par.N_multistart,len(self.theta),self.par.N_guesses))

        for i in range(self.par.N_multistart):
            initial_guessess[i] = np.random.uniform(size=(len(self.theta),self.par.N_guesses))

        # b. tasks
        tasks = (
                joblib.delayed(estimate.estimate_for_multistart)
                (i,initial_guessess[i],self.name,model_dict,load) 
                for i in range(self.par.N_multistart)                
                )

        # c. compute
        self.est['ests'] = joblib.Parallel(n_jobs=n_jobs)(tasks)

        # d. save
        if save: self.save()

        t1 = time.time()
        minutes = np.int((t1-t0)/60)
        seconds = np.int(t1-t0-minutes*60) 
        cpu_percent = psutil.cpu_percent(interval=None)
        print(f'total time: {minutes}min {seconds}s (avg. cpu load: {cpu_percent})')

    def apply_multistart(self):
        """ apply results from multi-start estimation """

        # a. find best
        best_obj = np.inf
        for est in self.est['ests']:
            if est['obj'] < best_obj:
                best_obj = est['obj']
                for name in self.theta.keys():
                    setattr(self.par,name,est[name])
                    self.est[name] = est[name]
                    self.est['obj'] = est['obj']
        
        print(f'best obj. = {best_obj:12.8f}')
        
        # b. calculate moments
        self.calc_moments()

        # c. claculate standard errors
        self.prepare_estimation()
        estimate.std_error(self)

    def show_multistart(self,N=-1):
        """ show N multistart results """

        sorted_ests = sorted(self.est['ests'],key=lambda k:k['obj'])
        cutoff_obj = [est['obj'] for est in sorted_ests][N]
        print(f'{"obj":>15s}',end='')
        print(' ',end='')
        for name in self.theta.keys():
            print(f'{name:>10s} ',end='')
        print(f'{"nits":>7s}')
        for i,est in enumerate(sorted_ests):
            if est['obj'] < cutoff_obj:
                print(f'{i:3d}: {est["obj"]:12.8f} ',end='')
                for name in self.theta.keys():
                    print(f'{est[name]:10.3f} ',end='')
                print(f' [{est["nits"]:5d}]')
        
        self.sorted_ests = sorted_ests

    ##################
    # 7. monte carlo #
    ##################

    def identification_estimation(self,do_print=False,true=None):
        """ fuld estimation of model """

        self.est = {}

        # a. from standard initial guess
        if not true is None and do_print: print('### from random guess ###\n')

        # i. initial guess
        self.find_initial_guess(do_print=do_print)
        if do_print: print('')

        # ii. estimate
        self.estimate(do_print=do_print)

        # b. from improved initial guess
        if not true is None:

            if do_print: print('### from improved guess ###\n')

            # i. remember
            est_A = self.est.copy()
            
            # ii. initial guess
            self.find_initial_guess(do_print=do_print,true=true)
            if do_print: print('')

            # iii. estimate
            self.estimate(do_print=do_print)
                          
            # iv. find best
            est_B = self.est.copy()
            self.est = est_B if est_B['obj'] < est_A['obj'] else est_A
            
            self.est['est_A'] = est_A
            self.est['est_B'] = est_B

            # v. finalize
            if do_print: print('### final results ###\n')

            for name in self.theta.keys(): setattr(self.par,name,self.est[name])
            self.calc_moments()

    def test_identification(self,n_jobs=50,load=False):
        """ run monte carlo """

        self.est = {}
        model_dict = self.as_dict()

        # folder for saving
        if not os.path.exists(f'saved/{self.name}_identification'):
            os.makedirs(f'saved/{self.name}_identification')

        t0 = time.time()
        psutil.cpu_percent(interval=None)
        
        # a. random numbers
        np.random.seed(self.par.seed)

        truepars = np.zeros((self.par.N_identification,len(self.theta)))
        initial_guessess = np.zeros((self.par.N_identification,len(self.theta),self.par.N_guesses))

        for i in range(self.par.N_identification):

            truepars[i,:] = np.random.uniform(size=len(self.theta))
            initial_guessess[i,:] = np.random.uniform(size=(len(self.theta),self.par.N_guesses))

        # b. tasks
        tasks = (
                joblib.delayed(estimate.identification)
                (i,truepars[i],initial_guessess[i],self.name,model_dict,load) 
                for i in range(self.par.N_identification)
                )

        # c. compute
        self.est['ests'] = joblib.Parallel(n_jobs=n_jobs)(tasks)

        # d. save
        self.save()

        t1 = time.time()
        minutes = np.int((t1-t0)/60)
        seconds = np.int(t1-t0-minutes*60) 
        cpu_percent = psutil.cpu_percent(interval=None)
        print(f'total time: {minutes}min {seconds}s (avg. cpu load: {cpu_percent})')

    def identification_results(self,cutoff=np.inf):
        """ print identification results to screen """

        # a. collect
        max_abs_diff = 0.0
        for name in self.theta.keys():

                diff = np.array([est['true'][name]-est['est'][name] for est in self.est['ests'] if est['obj'] < cutoff])
                self.est[(name,'mean')] = np.mean(np.abs(diff))
                self.est[(name,'min')] = np.min(diff)
                self.est[(name,'max')] = np.max(diff)
                max_abs_diff = np.fmax(max_abs_diff,np.max(np.abs(diff)))

        # c. print
        print(f'{"":12s} {"min-max diff.":11s}')
        for name in self.theta.keys():
            line = f'{name:11s}:'
            line += f' {self.est[(name,"min")]:6.2f} {self.est[(name,"max")]:6.2f}' 
            print(line)
        
        print('')

        max_obj = max([est['obj'] for est in self.est['ests'] if est['obj'] < cutoff])
        print(f'max obj.: {max_obj:.8f}')

        share = np.mean(np.array([est['obj'] < cutoff for est in self.est['ests']]))
        print(f'share < cutoff: {share:.3f}')

        print(f'max diff. in par: {max_abs_diff:.8f}')