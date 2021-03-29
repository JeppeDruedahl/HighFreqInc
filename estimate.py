import os
import time
import pickle
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from consav.smd_std_error import compute_std_error
import HighFreqIncProcess

#########################
# 1. objective function #
#########################

# 1. diff_moms_vec() does not call other functions
# 2. g() calls diff_moms_vec()
# 3. calc_obj() calls g()
# 4. obj() calls calc_obj()

def diff_moms_vec(moms,datamoms,data,model):
    """ difference between model and data moments """

    diff_moms_vec = np.zeros(len(moms))

    for i,(key,value) in enumerate(moms.items()):
        if key in datamoms:
            diff_moms_vec[i] = datamoms[key]-value

    return diff_moms_vec

def g(x,names,names_closed_form,model):
    """ calculate difference between model and data moments """

    # a. update par
    for i in range(x.size):
        setattr(model.par,names[i],x[i])

    if len(names_closed_form) > 0:
        update_closed_form(model,names_closed_form)
        
    # b. simulate and calculate moments
    if not model.par.use_theoretical:
        model.simulate()

    model.calc_moments()
        
    # c. differences to data moments
    return diff_moms_vec(model.moms,model.datamoms,model.data,model)

def calc_obj(model,x=None,names=None,names_closed_form=None,W=None):
    """ calculate objective value """

    # a. differences to data moments
    if x is None:
        diff = g(np.zeros(0),'','',model)
    else:
        diff = g(x,names,names_closed_form,model)
    
    # b. weighting
    diff = np.expand_dims(diff,axis=1)
    try:
        objval = diff.T @ W @ diff
        objval = objval[0,0]
    except:
        objval = -np.inf
    
    return objval

def obj(x,model,W,names,names_closed_form,lower,upper,do_print,**kwargs):
    """ full objective function """

    global nfuns
    global minfunval

    # a. setup print progress
    if do_print:
        if nfuns == 0:
            minfunval = np.inf
        nfuns += 1

    # b. implement bounds and penalty
    penalty = 0
    x_clipped = x.copy()
    
    for i in range(x.size):

        # i. clip
        if (lower[i] != None) or (upper[i] != None):
            x_clipped[i] = np.clip(x_clipped[i],lower[i],upper[i])
        
        # ii. penalty
        penalty += 10_000*(x[i]-x_clipped[i])**2

    # c. calcualte objective
    objval = calc_obj(model,x=x_clipped,names=names,names_closed_form=names_closed_form,W=W)
    objval += penalty
    
    # d. print progress
    if do_print:
        minfunval = np.fmin(objval,minfunval)

    return objval
              
###########
# 2. misc #
###########

def progress(xk,do_print,names):
    """ print progress when estimating """

    global nits
    global nfuns
    global minfunval
    global tic

    if do_print:

        # a. since last
        if nits > 0 and nits%10 == 0:
            print(f'  obj. = {minfunval:.8f}, {time.time()-tic:.1f} secs, {nfuns} func. evals')

        # b. new parameters
        if nits%10 == 0:
            print(f' iteration: {nits}')
            for x,name in zip(xk,names):
                print(f'  {name} = {x:.4f}')

    # d. update
    nits += 1
    nfuns = 0
    tic = time.time()

def construct_closed_form(par,datamoms):
    """ calculate closed form results """
    
    # a. rho
    rho = 1-(1-(datamoms[('auto_cov_d12y12l',3)]/datamoms[('auto_cov_d12y12l',2)])**(1/12))/par.p_psi
    
    if rho < -0.999:
        rho = -0.999
    elif rho > 0.999: 
        rho = 0.999

    rho_tilde = lambda k: (1-par.p_psi*(1-rho))**(12*k)

    # b. mean of permanent shock
    mu_phi = datamoms[('mean_d12ky',1)]/(12*par.p_phi)
    mu_phi_tilde = lambda k: 12*k*par.p_phi*(1-par.p_phi)*mu_phi**2
    
    # c. variance of persistent shock   
    var_psi = 2*(datamoms[('var_d12ky',2)]-mu_phi_tilde(2)) - (datamoms[('var_d12ky',1)]-mu_phi_tilde(1)) - (datamoms[('var_d12ky',3)]-mu_phi_tilde(3))
    var_psi *= (1-rho**2)
    var_psi /= 2*(rho_tilde(1)+rho_tilde(3)-2*rho_tilde(2))
    
    if var_psi > 0:
        sigma_psi = np.sqrt(var_psi)
    else:
        sigma_psi = 0

    # d. variance of permanent shock   
    var_phi = (datamoms[('var_d12ky',2)]-mu_phi_tilde(2)) - (datamoms[('var_d12ky',1)]-mu_phi_tilde(1)) - 2*sigma_psi**2/(1-rho**2)*(rho_tilde(1)-rho_tilde(2))
    var_phi /= 12*par.p_phi
    
    if var_phi > 0:
        sigma_phi = np.sqrt(var_phi)
    else:
        sigma_phi = 0

    # e. eta
    var_eta = datamoms[('auto_cov_d12y12l',1)] + var_psi/(1-rho**2)*(1-rho_tilde(1))**2 + par.p_xi*par.sigma_xi**2 + par.sigma_epsilon**2 + par.p_xi*(1-par.p_xi)*par.mu_xi**2               # Last term is xi_mu_tilde and is taken from eq. mu_xi_t in 'IncomeProcess.lyx' (currently eq. 18)
    var_eta /= -par.p_eta

    if var_eta > 0:
        sigma_eta = np.sqrt(var_eta)
    else:
        sigma_eta = 0    

    # d. dictionary
    closed_form = {}
    closed_form['rho'] = rho
    closed_form['mu_phi'] = mu_phi
    closed_form['sigma_psi'] = sigma_psi
    closed_form['sigma_phi'] = sigma_phi
    closed_form['sigma_eta'] = sigma_eta

    return closed_form

def update_closed_form(model,names):
    """ update using closed form results """

    closed_form = construct_closed_form(model.par,model.datamoms)
    for name in names:
        setattr(model.par,name,closed_form[name])

###############
# 3. estimate #
###############

def estimate(model,do_print=False):
    """ estimate model """

    global nits
    global theta

    par = model.par

    # a. theta (without those in closed form)
    _theta = model.theta
    model.theta = {key:value for key,value in model.theta.items() if (not 'closed_form' in value or not value['closed_form'])}

    # b. parameter names
    names = [key for key in model.theta.keys()]
    names_closed_form = [key for key,value in _theta.items() if ('closed_form' in value and value['closed_form'])]
    names_full = [key for key in _theta.keys()]

    # c. initial parameter guesses
    xk0 = [spec['guess'] for name,spec in model.theta.items()]

    # d. paramter bounds
    lower = np.array([spec['lower'] for name,spec in model.theta.items()])
    upper = np.array([spec['upper'] for name,spec in model.theta.items()])

    # e. setup print progress
    theta = model.theta
    nits = 0
    callback_func = lambda xk: progress(xk,do_print,names)

    # f. optimizer
    if do_print: 

        if par.est_method == 'Nelder-Mead+BFGS': 
            print('running optimizaition with Nelder-Mead:')
        else:
            print(f'running optimization with {par.est_method}:')

    if do_print: t0 = time.time()
    
    callback_func(xk0)

    if par.est_method == 'Nelder-Mead':

        res = optimize.minimize(obj,xk0,args=(model,model.par.W,names,names_closed_form,lower,upper,do_print),
                                method=par.est_method,
                                callback=callback_func,
                                options={'maxiter':par.est_max_iter,'xatol':model.par.est_tol,'fatol':model.par.est_tol})
    
    elif par.est_method == 'BFGS':

        res = optimize.minimize(obj,xk0,args=(model,model.par.W,names,names_closed_form,lower,upper,do_print),
                                method='BFGS',
                                callback=callback_func,
                                options={'maxiter':par.est_max_iter,'gtol':par.est_tol})
                                        
    elif par.est_method == 'CG':

        res = optimize.minimize(obj,xk0,args=(model,model.par.W,names,names_closed_form,lower,upper,do_print),
                                method='CG',
                                callback=callback_func,
                                options={'maxiter':par.est_max_iter,'gtol':par.est_tol})

    elif par.est_method == 'Nelder-Mead+BFGS':

        res = optimize.minimize(obj,xk0,args=(model,model.par.W,names,names_closed_form,lower,upper,do_print),
                                method='Nelder-Mead',
                                callback=callback_func,
                                options={'maxiter':par.est_max_iter_ini})

        xk0 = res.x

        if do_print: print('\nrunning optimization with BFGS:')
        nits = 0

        res = optimize.minimize(obj,xk0,args=(model,model.par.W,names,names_closed_form,lower,upper,do_print),
                                method='BFGS',
                                callback=callback_func,
                                options={'maxiter':par.est_max_iter,'gtol':par.est_tol})

    else:

        raise Exception('unknown solver method')    
    
    if do_print:
        
        print('')
        t1 = time.time()
        minutes = np.int((t1-t0)/60)
        seconds = np.int(t1-t0-minutes*60) 
        print(f'total time: {minutes}min {seconds}s')
        print(f'iterations = {res.nit}')
        print(f'obj. = {res.fun:.4f}')
        print('')

    # g. update with estimation results

    # i. objective 
    model.est['obj'] = res.fun
    model.est['nits'] = nits
    model.est['res'] = res

    # ii. parameters
    for name,value in zip(names,res.x):
        setattr(model.par,name,value)

    # # iii. closed form parameters
    model.theta = _theta # reset theta
    if len(names_closed_form) > 0: update_closed_form(model,names_closed_form)

    # iv. set estimates
    for name in names_full: model.est[name] = getattr(model.par,name)

    # v. calculate moments
    model.calc_moments()
    model.est['moms'] = model.moms

def std_error(model):
    """ compute standard error and objective with optimal weighting matrix """

    # a. find names and values
    names = [key for key in model.theta.keys()]
    names_closed_form = [key for key,value in model.theta.items() if ('closed_form' in value and value['closed_form'])]
    x = np.array([getattr(model.par,name) for name in names])

    # b. objective with optimal weighting matrix
    W_opt = np.linalg.inv(model.par.Ypsilon)
    model.est['obj_opt'] = calc_obj(model,W=W_opt)

    # c. compute standard errors
    se = compute_std_error(g,x,model.par.W,model.par.Ypsilon,model.par.Ndata,
                           Nsim=1e10,step=1e-5,args=(names,names_closed_form,model)) # Nsim=1e10 -> nu simulation error

    # d. reset par
    for i in range(x.size):
        setattr(model.par,names[i],x[i])
        model.est[(names[i],'se')] = se[i]
    for name in names_closed_form:
        model.est[(name,'se')] = np.nan

def estimate_for_multistart(num,initial_guesses,name,model_dict,load=False):
    """ estimation function for use in multistart """

    # load ? -> return
    filename = f'saved/{name}_multistart/{num}.p'
    if load and os.path.isfile(filename):
        with open(filename, 'rb') as f:
            est_dict = pickle.load(f)
        return est_dict

    # a. model
    model = HighFreqIncProcess.HighFreqIncProcessClass(name=f'{name}_multistart',from_dict=model_dict)
    model.initial_guesses = initial_guesses

    # b. estimate
    model.find_initial_guess()
    model.estimate()
    est_dict = model.est

    # c. save and return
    with open(filename, 'wb') as f:
        pickle.dump(est_dict, f)

    return model.est

#####################
# 4. Identification #
#####################

def identification(num,truepar,initial_guesses,name,model_dict,load=False):
    """ run verify identification experiment """

    # load ? -> return
    filename = f'saved/{name}_identification/{num}.p'
    if load and os.path.isfile(filename):
        with open(filename, 'rb') as f:
            est_dict = pickle.load(f)
        return est_dict

    # a. data 
    data = HighFreqIncProcess.HighFreqIncProcessClass(name=f'{name}_identification_data',from_dict=model_dict)

    names = [key for key in data.theta.keys()] # parameters to be estimated

    # b. draw true parameters
    for i,name in enumerate(names):
        spec = data.theta[name]
        value = spec['lower'] + truepar[i]*(spec['upper']-spec['lower'])
        setattr(data.par,name,value)

    # c. calculate true moments
    data.calc_moments()
    
    # d. estimate
    true = {name:getattr(data.par,name) for name in names}

    model = HighFreqIncProcess.HighFreqIncProcessClass(name=f'{name}__identification',from_dict=model_dict)
    model.initial_guesses = initial_guesses
    model.load_datamoms(data)
    model.identification_estimation(true=true)

    # e. pack
    est = {name:getattr(model.par,name) for name in names}
    est_dict = {'est':est,'true':true,'obj':model.est['obj'],'est_A':model.est['est_A'],'est_B':model.est['est_B']}

    # f. save and return
    with open(filename, 'wb') as f:
        pickle.dump(est_dict, f)

    return est_dict