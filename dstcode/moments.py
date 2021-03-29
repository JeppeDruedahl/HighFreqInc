import numpy as np
from numba import njit

@njit
def mean_var_skew_kurt_ages(x,age,cond,ages,periods=12):
    """ calcualte mean, variance, skewness and kurtosis """
    
    # a. allocate memory
    N = x.shape[0]
    T = x.shape[1]
    Nages = ages.size

    Nactive = np.zeros(ages.size,dtype=np.int64)
    out = np.zeros((4,Nages))

    mean = out[0,:]
    var = out[1,:]
    skew = out[2,:]
    kurt = out[3,:]

    # b. determine sums and sample
    for i in range(N):
        for j in range(Nages):

            t0 = (ages[j]-age[i,0])*periods
            if t0 < 0 or t0+periods > T: continue

            for t in range(t0,t0+periods):
                if cond[i,t] and (not np.isnan(x[i,t])):
                    Nactive[j] += 1
                    mean[j] += x[i,t]

    # c. means
    for j in range(Nages):
        if Nactive[j] == 0:
            mean[j] = np.nan
        else:
            mean[j] /= Nactive[j]

    # d. variance and kurtosis
    for i in range(N):        
        for j in range(Nages):

            if Nactive[j] == 0: continue
            t0 = (ages[j]-age[i,0])*periods
            if t0 < 0 or t0+periods > T: continue
        
            for t in range(t0,t0+periods):
                if cond[i,t] and (not np.isnan(x[i,t])):
                    
                    diff = x[i,t]-mean[j]
                    diff2 = diff*diff

                    var[j] += diff2
                    skew[j] += diff2*diff
                    kurt[j] += diff2*diff2
    
    # e. result
    for j in range(Nages):

        if Nactive[j] > 0:
            var[j] /= Nactive[j]-1
        else:
            var[j] = np.nan
        
        if Nactive[j] > 2:
            cor_fac = Nactive[j]/((Nactive[j]-1)*(Nactive[j]-2))
            skew[j] *= cor_fac
            skew[j] /= var[j]**(3/2)
        else:
            skew[j] = np.nan

        if Nactive[j] > 3:
            cor_fac =  (((Nactive[j]-1)/Nactive[j])*((Nactive[j]-2)/(Nactive[j]+1))*(Nactive[j]-3))
            cor_sub = 3*(Nactive[j]-1)*(Nactive[j]-1) / ((Nactive[j]-2)*(Nactive[j]-3))
            kurt[j] /= cor_fac
            kurt[j] /= var[j]*var[j]
            kurt[j] -= cor_sub
        else: 
            kurt[j] = np.nan
    
    return out.ravel()

@njit
def cov_ages(a,b,offset,age,cond,ages,periods=12):
    """ calculate covariance """

    # a. allocate memory
    N = a.shape[0]
    T = a.shape[1]
    Nages = ages.size

    Nactive = np.zeros(Nages,dtype=np.int64)
    mean_a = np.zeros(Nages)
    mean_b = np.zeros(Nages)
    cov = np.zeros(Nages)

    # b. determine sums and sample
    for i in range(N):
        for j in range(Nages):

            t0 = (ages[j]-age[i,0])*periods
            if t0 < 0: continue
            if t0+periods > T: continue
            if t0-offset < 0: continue
            
            for t in range(t0,t0+periods):

                if cond[i,t] and (not np.isnan(a[i,t])) and (not np.isnan(b[i,t-offset])):
                    Nactive[j] += 1
                    mean_a[j] += a[i,t]
                    mean_b[j] += b[i,t-offset]

    # c. means
    for j in range(Nages):
        if Nactive[j] == 0:
            mean_a[j] = np.nan
            mean_b[j] = np.nan
        else:
            mean_a[j] /= Nactive[j]
            mean_b[j] /= Nactive[j]

    # d. covariance
    for i in range(N):
        
        for j in range(Nages):

            if Nactive[j] == 0: continue

            t0 = (ages[j]-age[i,0])*periods
            if t0 < 0: continue
            if t0+periods > T: continue
            if t0-offset < 0: continue
            
            for t in range(t0,t0+periods):
                if cond[i,t] and (not np.isnan(a[i,t])) and (not np.isnan(b[i,t-offset])):
                    cov[j] += (a[i,t]-mean_a[j])*(b[i,t-offset]-mean_b[j])
    
    # e. result
    for j in range(Nages):

        if Nactive[j] > 0:
            cov[j] /= Nactive[j]-1
        else:
            cov[j] = np.nan
    
    return cov   

@njit
def share_in_range(x,etas_low,etas_high,age,cond,ages,periods=12):
    """ calculate share in range """

    # a. allocate memory
    N = x.shape[0]
    T = x.shape[1]
    Nages = ages.size
    Netas = etas_low.size

    Nactive = np.zeros((Nages))
    Ntrue = np.zeros((Nages,Netas))

    # b. compute
    for i in range(N):
        for j in range(Nages):

            t0 = (ages[j]-age[i,0])*periods
            if t0 < 0 or t0+periods > T: continue

            for t in range(t0,t0+periods):

                if cond[i,t] and (not np.isnan(x[i,t])):

                    Nactive[j] += 1
                    for h in range(Netas):
                        if (x[i,t] >= etas_low[h]) and (x[i,t] <= etas_high[h]):
                            Ntrue[j,h] += 1
                        else:
                            break
    
    # c. result
    out = np.zeros((Netas,Nages))
    for j in range(Nages):
        if Nactive[j] > 0:
            for h in range(Netas):
                out[h,j] = Ntrue[j,h]/Nactive[j]
        else:
            for h in range(Netas):
                out[h,j] = np.nan

    return out.ravel()