import numpy as np
from numba import njit

###########
# 1. base #
###########

@njit
def mean_var_skew_kurt(x):
    """ calculate mean, variance, skewness and kurtosis """

    # a. allocate memory and initialize
    out = np.zeros(4)

    mean = out[0:]
    var = out[1:]
    skew = out[2:]
    kurt = out[3:]

    Nactive = 0

    # b. determine sum and active
    for i in range(x.size):
        if ~np.isnan(x[i]):
            Nactive += 1
            mean[0] += x[i]

    # c. means
    if Nactive == 0:
        mean[0] = np.nan
    else:
        mean[0] /= Nactive

    # d. variance, skewness and kurtosis
    for i in range(x.size):
        if Nactive == 0: continue
        if ~np.isnan(x[i]):
            diff = x[i]-mean[0]
            diff2 = diff*diff
            var[0] += diff2
            skew[0] += diff2*diff
            kurt[0] += diff2*diff2
    
    # e. results
    if Nactive > 0:
        var[0] /= Nactive-1
    else:
        var[0] = np.nan
    
    if Nactive > 2:
        cor_fac = Nactive/((Nactive-1)*(Nactive-2))
        skew[0] *= cor_fac
        skew[0] /= var[0]**(3/2)
    else:
        skew[0] = np.nan

    if Nactive > 3:
        cor_fac =  (((Nactive-1)/Nactive)*((Nactive-2)/(Nactive+1))*(Nactive-3))
        cor_sub = 3*(Nactive-1)*(Nactive-1) / ((Nactive-2)*(Nactive-3))
        kurt[0] /= cor_fac
        kurt[0] /= var[0]*var[0]
        kurt[0] -= cor_sub
    else: 
        kurt[0] = np.nan
    
    return out
    
@njit
def cov(a,b):
    """ calculate covariance """

    # a. initialize
    mean_a = 0.0
    mean_b = 0.0
    Nactive = 0

    # b. determine sums and active
    for i in range(a.size):

        if ~np.isnan(a[i]) and ~np.isnan(b[i]):

            Nactive += 1
            mean_a += a[i]
            mean_b += b[i]       

    # c. means
    if Nactive == 0:
        return np.nan

    mean_a /= Nactive
    mean_b /= Nactive

    # d. covariance
    cov = 0.0
    for i in range(a.size):
        
        if ~np.isnan(a[i]) and ~np.isnan(b[i]):
            cov += (a[i]-mean_a)*(b[i]-mean_b)

    # e. result
    cov /= (Nactive-1)

    return cov

@njit
def share_in_range(x,omegas_low,omegas_high):
    """ calculate share in range """

    # a. allocate memory and inialize
    Nomegas = omegas_low.size
    Ntrue = np.zeros(Nomegas)

    Nactive = 0

    # b. compute
    for i in range(x.size):
        if ~np.isnan(x[i]):
            Nactive += 1
            for h in range(Nomegas):
                if (x[i] >= omegas_low[h]) and (x[i] <= omegas_high[h]):
                    Ntrue[h] += 1
                else:
                    break # assuming ordered from high to low
    
    # c. result
    out = np.zeros(Nomegas)
    if Nactive > 0:
        for h in range(Nomegas):
            out[h] = Ntrue[h]/Nactive
    else:
        for h in range(Nomegas):
            out[h] = np.nan

    return out

@njit
def share_in_range_cond(x,y,omegas_low,omegas_high,cond_low,cond_high):
    """ calculate conditional share in range """

    # a. allocate memory and inialize
    Nomegas = omegas_low.size
    Ntrue = np.zeros(Nomegas)
    Ntrue_l = np.zeros(Nomegas)
    Ntrue_u = np.zeros(Nomegas)

    Nactive = 0
    Nactive_l = 0
    Nactive_u = 0

    # b. compute
    for i in range(x.size):

        if ~np.isnan(x[i]) and (y[i] >= cond_low) and (y[i] <= cond_high):
            Nactive += 1
            for h in range(Nomegas):
                if (x[i] >= omegas_low[h]) and (x[i] <= omegas_high[h]):
                    Ntrue[h] += 1
                else:
                    break # assuming ordered from high to low

        if ~np.isnan(x[i]) and y[i] < cond_low:
            Nactive_l += 1
            for h in range(Nomegas):
                if (x[i] >= omegas_low[h]) and (x[i] <= omegas_high[h]):
                    Ntrue_l[h] += 1
                else:
                    break # assuming ordered from high to low

        if ~np.isnan(x[i]) and y[i] > cond_high:
            Nactive_u += 1
            for h in range(Nomegas):
                if (x[i] >= omegas_low[h]) and (x[i] <= omegas_high[h]):
                    Ntrue_u[h] += 1
                else:
                    break # assuming ordered from high to low

    # c. result
    out = np.zeros((3,Nomegas))
    if Nactive > 0:
        for h in range(Nomegas):
            out[0,h] = Ntrue[h]/Nactive
            out[1,h] = Ntrue_l[h]/Nactive_l
            out[2,h] = Ntrue_u[h]/Nactive_u
    else:
        for h in range(Nomegas):
            out[0,h] = np.nan
            out[1,h] = np.nan
            out[2,h] = np.nan

    return out

######################
# 2. autocovariances #
######################

def auto_cov_d1y1l(par,sim,l):
    assert l > 0
    return cov(sim.d1ky[0,:,l:].ravel(),sim.d1ky[0,:,:-l].ravel())

def auto_cov_d12y12l(par,sim,l):
    assert l > 0
    return cov(sim.d12ky[0,:,12*l:].ravel(),sim.d12ky[0,:,:-12*l].ravel())

def frac_auto_cov_d12y1l(par,sim,l):
    assert l > 0 and l < 12
    return cov(sim.d12ky[0,:,l:].ravel(),sim.d12ky[0,:,:-l].ravel())

# windsorized
def auto_cov_d1yw1l(par,sim,l):
    assert l > 0
    return cov(sim.d1kyw[0,:,l:].ravel(),sim.d1kyw[0,:,:-l].ravel())

def auto_cov_d12yw12l(par,sim,l):
    assert l > 0
    return cov(sim.d12kyw[0,:,12*l:].ravel(),sim.d12kyw[0,:,:-12*l].ravel())

def frac_auto_cov_d12yw1l(par,sim,l):
    assert l > 0 and l < 12
    return cov(sim.d12kyw[0,:,l:].ravel(),sim.d12kyw[0,:,:-l].ravel())

###########
# 4. fast #
###########

def get_omegas(model,momname,k):
    """ unpack omegas from moment specification """

    if momname in model.specs:
        return np.array([info['args'][1] for info in model.specs[momname] if info['args'][0] == k])
    else:
        return np.array([])

def fast(model,momname,args):
    """ function for calculating moments fast """

    # a. mean_var_skew_kurt
    _d12ky_list = ['mean_d12ky','var_d12ky','skew_d12ky','kurt_d12ky']
    _d12kyw_list = ['mean_d12kyw','var_d12kyw','skew_d12kyw','kurt_d12kyw']

    if (momname in _d12ky_list) or (momname in _d12kyw_list):

        # i. compute
        k = args
        if momname in _d12ky_list:
            x = model.sim.d12ky[k-1].ravel()
        else:
            x = model.sim.d12kyw[k-1].ravel()
        out = mean_var_skew_kurt(x)
        
        # ii. unpack
        if momname in _d12ky_list:
            _d12ky_list_now = _d12ky_list
        else:
            _d12ky_list_now = _d12kyw_list

        for i,momname_ in enumerate(_d12ky_list_now):
            if momname_ in model.specs:
                model.moms[(momname_,args)] = out[i]

        return True

    _d1ky_list = ['mean_d1ky','var_d1ky','skew_d1ky','kurt_d1ky']
    _d1kyw_list = ['mean_d1kyw','var_d1kyw','skew_d1kyw','kurt_d1kyw']

    if (momname in _d1ky_list) or (momname in _d1kyw_list):

        # i. compute
        k = args
        if _d1ky_list:
            x = model.sim.d1ky[k-1].ravel()
        else:
            x = model.sim.d1kyw[k-1].ravel()

        out = mean_var_skew_kurt(x)
        
        # ii. unpack
        if momname in _d1ky_list:
            _d1ky_list_now = _d1ky_list
        else:
            _d1ky_list_now = _d1kyw_list

        for i,momname_ in enumerate(_d1ky_list_now):
            if momname_ in model.specs:
                model.moms[(momname_,args)] = out[i]

        return True

    # b. leq
    if momname in ['leq_d12ky','cdf_d12ky']:
        
        # i. k and omegas
        k = args[0]

        if momname == 'leq_d12ky':
            omegas_high = get_omegas(model,'leq_d12ky',k)
        else:
            omegas_high = model.par.omegas_cdf

        omegas_low = -np.inf*np.ones(omegas_high.size)

        # ii. compute
        x = model.sim.d12ky[k-1].ravel()
        out = share_in_range(x,omegas_low,omegas_high)
        
        # iii. unpack
        for i,omega in enumerate(omegas_high):
            if momname == 'leq_d12ky':
                model.moms[(momname,(k,omega))] = out[i]
            else:
                model.moms[(momname,(k,i))] = out[i]

        return True  

    if momname in ['leq_d1ky','cdf_d1ky']:
        
        # i. k and omegas
        k = args[0]

        if momname == 'leq_d1ky':
            omegas_high = get_omegas(model,'leq_d1ky',k)
        else:
            omegas_high = model.par.omegas_cdf

        omegas_low = -np.inf*np.ones(omegas_high.size)

        # ii. compute
        x = model.sim.d1ky[k-1].ravel()
        out = share_in_range(x,omegas_low,omegas_high)
        
        # iii. unpack
        for i,omega in enumerate(omegas_high):
            if momname == 'leq_d1ky':
                model.moms[(momname,(k,omega))] = out[i]
            else:
                model.moms[(momname,(k,i))] = out[i]

        return True  

    # c. leq_d*ky_midrange
    if momname in  ['leq_d12ky_midrange','cdf_d12ky_midrange']:
        
        # i. k and omegas
        k = args[0]

        if momname == 'leq_d12ky_midrange':
            omegas_high = get_omegas(model,'leq_d12ky_midrange',k)
        else:
            omegas_high = model.par.omegas_cdf

        omegas_low = -np.inf*np.ones(omegas_high.size)
        
        # ii. compute
        x = model.sim.d12ky[k-1].ravel()
        y = model.sim.d12ky_lag[k-1].ravel()

        cond_low = -model.par.omega_cond_midrange
        cond_high = model.par.omega_cond_midrange

        out = share_in_range_cond(x,y,omegas_low,omegas_high,cond_low,cond_high)

        # iii. unpack
        for i,omega in enumerate(omegas_high):
            if momname == 'leq_d12ky_midrange':
                model.moms[('leq_d12ky_midrange',(k,omega))] = out[0,i]
                if ('leq_d12ky_midrange_l',(k,omega)) in model.specs: model.moms[('leq_d12ky_midrange_l',(k,omega))] = out[1,i]
                if ('leq_d12ky_midrange_u',(k,omega)) in model.specs: model.moms[('leq_d12ky_midrange_u',(k,omega))] = out[2,i]
            else:
                model.moms[(momname,(k,i))] = out[0,i]

        return True

    if momname in  ['leq_d1ky_midrange','cdf_d1ky_midrange']:
        
        # i. k and omegas
        k = args[0]

        if momname == 'leq_d1ky_midrange':
            omegas_high = get_omegas(model,'leq_d1ky_midrange',k)
        else:
            omegas_high = model.par.omegas_cdf

        omegas_low = -np.inf*np.ones(omegas_high.size)
        
        # ii. compute
        x = model.sim.d1ky[k-1].ravel()
        y = model.sim.d1ky_lag[k-1].ravel()

        cond_low = -model.par.omega_cond_midrange
        cond_high = model.par.omega_cond_midrange

        out = share_in_range_cond(x,y,omegas_low,omegas_high,cond_low,cond_high)

        # iii. unpack
        for i,omega in enumerate(omegas_high):
            if momname == 'leq_d1ky_midrange':
                model.moms[('leq_d1ky_midrange',(k,omega))] = out[0,i]
                if ('leq_d1ky_midrange_l',(k,omega)) in model.specs: model.moms[('leq_d1ky_midrange_l',(k,omega))] = out[1,i]
                if ('leq_d1ky_midrange_l',(k,omega)) in model.specs: model.moms[('leq_d1ky_midrange_u',(k,omega))] = out[2,i]
            else:
                model.moms[(momname,(k,i))] = out[0,i]

        return True

    if momname == 'var_y_d12_diff':

        var_y = np.var(model.sim.y,axis=0)
        for k in range(1,model.par.kmax):
            model.moms[('var_y_d12_diff',k)] = np.mean(var_y[12*k:]-var_y[:-12*k])

        return True

    if momname == 'cov_y_y_d12_diff':

        cov_y_d12_diff_ = np.nan*np.ones((model.par.kmax-2,model.par.simT))
        for t in range(model.par.simT-12):
            cov_short = cov(model.sim.y[:,t],model.sim.y[:,t+12])
            for k in range(1,model.par.kmax-1):
                if t+12+12*k < model.par.simT:
                    cov_long = cov(model.sim.y[:,t],model.sim.y[:,t+12+12*k])
                    cov_y_d12_diff_[k-1,t] = cov_long-cov_short

        for k in range(1,model.par.kmax-1):
            model.moms[('cov_y_y_d12_diff',k)] = np.nanmean(cov_y_d12_diff_[k-1])

        return True

    # did not find anything
    return False