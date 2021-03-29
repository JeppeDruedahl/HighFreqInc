import itertools as it
import numpy as np
import scipy.stats as ss
from scipy.stats import norm
import math

from numba import njit
from consav import jit

########################
# 1. mean and variance #
########################

@njit(fastmath=True)
def aux(par):
    """ combined parameters often used """

    rho_tilde = (1.0 - par.p_psi*(1.0-par.rho))
    long_run_fac = (par.sigma_psi**2)/(1.0-par.rho**2)
    
    return rho_tilde,long_run_fac

@njit(fastmath=True)
def mean_d1ky(par,sim,k):
    return par.mu_phi*k*par.p_phi

@njit(fastmath=True)
def mean_d12ky(par,sim,k):
    return mean_d1ky(par,sim,12*k)

@njit(fastmath=True)
def var_d1ky(par,sim,k):
    
    # a. psi + phi
    if par.rho < 1:
        rho_tilde, long_run_fac = aux(par)
        psi_phi_term = 2.0*long_run_fac*(1-rho_tilde**k)
    else:
        psi_phi_term = par.p_psi*k*par.sigma_psi**2

    psi_phi_term += par.p_phi*k*par.sigma_phi**2

    # b. xi + eta + epsilon
    xi_eta_eps_term = 2.0*(par.p_xi*par.sigma_xi**2 + par.p_eta*par.sigma_eta**2 + par.sigma_epsilon**2)

    # c. mu's
    mu_xi_term = 2*par.p_xi*(1-par.p_xi)*par.mu_xi**2
    mu_eta_term = 2*par.p_eta*(1-par.p_eta)*par.mu_eta**2
    mu_phi_term = k*par.p_phi*(1-par.p_phi)*par.mu_phi**2

    return psi_phi_term + xi_eta_eps_term + mu_xi_term + mu_phi_term + mu_eta_term
  
@njit(fastmath=True)
def var_d12ky(par,sim,k):
    return var_d1ky(par,sim,12*k)

# windsorized
def mean_d12kyw(par,sim,k):
    return mean_d12ky(par,sim,k)

def var_d12kyw(par,sim,k):
    return var_d12ky(par,sim,k)

def mean_d1kyw(par,sim,k):
    return mean_d1ky(par,sim,k)

def var_d1kyw(par,sim,k):
    return var_d1ky(par,sim,k)

######################
# 2. autocovariances #
######################

@njit(fastmath=True)
def auto_cov_dyl(par,sim,l,k=1):
    
    # assert l > 0
    
    # no contribution from phi, permanent shock

    if l == 1:
        
        # a. psi
        if par. rho < 1:
            rho_tilde, long_run_fac = aux(par)
            psi_term = -long_run_fac*(1-rho_tilde**k)**2
        else:
            psi_term = 0

        # xi + eta + eps term
        xi_eta_eps_term = (par.p_xi*par.sigma_xi**2 + par.p_eta*par.sigma_eta**2 + par.sigma_epsilon**2)

        # mu term
        mu_xi_term = par.p_xi*(1-par.p_xi)*par.mu_xi**2
        mu_eta_term = par.p_eta*(1-par.p_eta)*par.mu_eta**2
        
        return  psi_term - xi_eta_eps_term - mu_xi_term - mu_eta_term

    else: # l > 1

        if par.rho < 1:
            rho_tilde, long_run_fac = aux(par)
            return -long_run_fac*(1-rho_tilde**k)**2*rho_tilde**(k*(l-1))
        else:
            return 0

@njit(fastmath=True)
def auto_cov_d1y1l(par,sim,l):
    return auto_cov_dyl(par,sim,l,k=1)

@njit(fastmath=True)
def auto_cov_d12y12l(par,sim,l):
    return auto_cov_dyl(par,sim,l,k=12)

@njit(fastmath=True)
def frac_auto_cov_dky1l(par,sim,l,k):

    assert l > 0 and l < k

    # no contribution from transitory shocks

    if par.rho < 1:

        rho_tilde, long_run_fac = aux(par)
        psi_phi_term = 2*rho_tilde**l
        psi_phi_term -= rho_tilde**(k-l)
        psi_phi_term -= rho_tilde**(k+l)
        psi_phi_term *= long_run_fac
    
    else:

        psi_phi_term = (k-l)*par.sigma_psi**2*par.p_psi

    psi_phi_term += (k-l)*par.sigma_phi**2*par.p_phi

    # b. mu
    mu_phi_term = ((k-l)*par.p_phi*(1-par.p_phi))*par.mu_phi**2

    return psi_phi_term + mu_phi_term

@njit(fastmath=True)
def frac_auto_cov_d12y1l(par,sim,l):
    return frac_auto_cov_dky1l(par,sim,l,k=12)

# windsorized
@njit(fastmath=True)
def auto_cov_d1yw1l(par,sim,l):
    return auto_cov_d1y1l(par,sim,l)

@njit(fastmath=True)
def auto_cov_d12yw12l(par,sim,l):
    return auto_cov_d12y12l(par,sim,l)

@njit(fastmath=True)
def frac_auto_cov_d12yw1l(par,sim,l):
    return frac_auto_cov_d12y1l(par,sim,l)

#################
# 3. normal cdf #
#################

import math
inv_sqrt2 = 1/math.sqrt(2) # precompute

@njit(fastmath=True)
def _norm_cdf(z):  
    """ raw normal cdf """

    return 0.5*math.erfc(-z*inv_sqrt2)

@njit(fastmath=True)
def norm_cdf(z,mean,std):
    """ normal cdf with scaling """

    # a. check
    if std <= 0:
        if z > mean: return 1
        else: return 0

    # b. scale
    z_scaled = (z-mean)/std

    # c. return
    return _norm_cdf(z_scaled)

@njit(fastmath=True)
def norm_cdf_vec(ps,zs,mean,std):
    """ vectorized normal cdf """

    # a. check
    if std <= 0:
        I = zs > mean
        ps[I] = 1
        ps[~I] = 0
        return

    # b. scale
    z_scaled = (zs-mean)/std

    for i in range(zs.size):
        ps[i] = 0.5*math.erfc(-z_scaled[i]*inv_sqrt2)

# nodes and weights for bivriate normal cdf
w1 = np.array([0.1713244923791705,0.3607615730481384,0.4679139345726904])
x1 = np.array([0.9324695142031522,0.6612093864662647,0.2386191860831970])
w1 = np.concatenate((w1,w1))
x1 = np.concatenate((1-x1,1+x1))

w2 = np.zeros(6)
w2[:3] = np.array([0.04717533638651177,0.1069393259953183,0.1600783285433464])
w2[3:] = np.array([0.2031674267230659,0.2334925365383547,0.2491470458134029])
x2 = np.zeros(6)
x2[:3] = np.array([0.9815606342467191,0.9041172563704750,0.7699026741943050])
x2[3:] = np.array([0.5873179542866171,0.3678314989981802,0.1252334085114692])
w2 = np.concatenate((w2,w2))
x2 = np.concatenate((1-x2,1+x2))

w3 = np.zeros(10)
w3[:3] = np.array([.01761400713915212,0.04060142980038694,0.06267204833410906])
w3[3:6] = np.array([.08327674157670475,0.1019301198172404,0.1181945319615184])
w3[6:9] = np.array([0.1316886384491766,0.1420961093183821,0.1491729864726037])
w3[9] = 0.1527533871307259
x3 = np.zeros(10)
x3[:3] = np.array([0.9931285991850949,0.9639719272779138,0.9122344282513259])
x3[3:6] = np.array([0.8391169718222188,0.7463319064601508,0.6360536807265150])
x3[6:9] = np.array([0.5108670019508271,0.3737060887154196,0.2277858511416451])
x3[9] = 0.07652652113349733
w3 = np.concatenate((w3,w3))
x3 = np.concatenate((1-x3,1+x3))

@njit(fastmath=True)
def binorm_cdf(z1,z2,mean1,mean2,std1,std2,cov):
    """ bi-variate normal cdf with scaling """ 

    # a. scale (and make negative) 
    if std1 > 0:
        h = -(z1-mean1)/std1
    else:
        h = -np.sign(z1-mean1)*np.inf

    if std2 > 0:
        k = -(z2-mean2)/std2
    else:
        k = -np.sign(z1-mean2)*np.inf

    if std1 > 0 and std2 > 0:
        r = cov/(std1*std2)
    else:
        r = 0

    # b. cases with infinity or zero correlation
    if (h > 0 and np.isinf(h)) or (k > 0 and np.isinf(k)): return 0
    elif (h < 0 and np.isinf(h)):
        if (k < 0 and np.isinf(k)): return 1
        else: return _norm_cdf(-k)
    elif (k < 0 and np.isinf(k)): return _norm_cdf(-h)
    elif r == 0: return _norm_cdf(-k)*_norm_cdf(-h)

    # c. baseline calculations
    tp = 2*math.pi
    hk = h*k
    bvn = 0
        
    if np.abs(r) < 0.3:
        x = x1
        w = w1
    elif np.abs(r) < 0.75:
        x = x2
        w = w2
    else:
        x = x3
        w = w3
        
    # d. simple case
    if abs(r) < 0.925:

        hs = (h*h+k*k)/2 
        asr = math.asin(r)/2  
        sn = np.sin(asr*x) 
        bvn = np.exp((sn*hk-hs)/(1-sn*sn))@w.T
        bvn = bvn*asr/tp+_norm_cdf(-h)*_norm_cdf(-k)

        return np.fmax(0,np.fmin(1,bvn))
        
    # e. general case     
    if r < 0:
        k = -k
        hk = -hk
                           
    if abs(r) < 1:

        # i. block I       
        as_ = 1-r**2
        a = np.sqrt(as_) 
        bs = (h-k)**2
        asr = -(bs/as_+hk)/2 
        c = (4-hk)/8 
        d = (12-hk)/80

        # ii. block II
        if asr > -100:   
            bvn = a*np.exp(asr)*(1-c*(bs-as_)*(1-d*bs)/3+c*d*as_**2) 

        # iii. block III
        if hk  > -100:   

            # o. block IIIa 
            b = np.sqrt(bs)
            sp = np.sqrt(tp)*_norm_cdf(-b/a)
            bvn = bvn - np.exp(-hk/2)*sp*b*( 1 - c*bs*(1-d*bs)/3 )                       
            a = a/2
            xs = (a*x)**2
            ast_vec = -(bs/xs+hk)/2            

            # oo. block IIIb
            num_xs_ = np.sum(ast_vec > -100)
            xs_ = np.zeros(num_xs_)
            ast_vec_ = np.zeros(num_xs_)
            w_ = np.zeros(num_xs_)

            j = 0
            for i in range(xs.size):
                if ast_vec[i] > -100:
                    xs_[j] = xs[i]
                    ast_vec_[j] = ast_vec[i]
                    w_[j] = w[i]
                    j += 1

            # ooo. block IIIc
            sp_ = (1+c*xs_*(1+5*d*xs_)) 
            rs = np.sqrt(1-xs_)
            ep_ = np.exp(-(hk/2)*xs_/(1+rs)**2)/rs          
            q1 = np.exp(ast_vec_)*(sp_-ep_)
            q2 = w_.T        
            bvn = (a*q1@q2-bvn)/tp

    # e. final            
    if r > 0:
        bvn =  bvn + _norm_cdf(-np.fmax(h,k))
    elif h >= k:
        bvn = -bvn
    else:               
        if h < 0:
            L = _norm_cdf(k)-_norm_cdf(h)
        else:
            L = _norm_cdf(-h)-_norm_cdf(-k) 
        bvn =  L - bvn
    
    return np.fmax(0,np.fmin(1,bvn))

######################
# 4. mixture results #
######################

# weights for psi and xi
def create_weights(par,k):

    par.w_phi = np.zeros(k+1)
    par.w_psi = np.zeros(k+1)
    par.w_eta = np.zeros(2)
    par.w_xi = np.zeros(2)
    
@njit(fastmath=True)
def fill_weights(par,k):
    """ weights for psi (binominal) and xi (bernoulli) """

    # a. phi
    if par.p_phi > 0 and par.p_phi < 1:
        
        if k == 1:

            par.w_phi[0] = 1-par.p_phi
            par.w_phi[1] = par.p_phi

        else:

            for s in range(k+1):
                if s == 0:
                    par.w_phi[s] = (1-par.p_phi)**k
                else:
                    fac1 = 1
                    for i,j in zip(range(k-s+1,k+1),range(1,s+1)):
                        fac1 *= i/j
                    fac2 = par.p_phi**s*(1-par.p_phi)**(k-s)
                    par.w_phi[s] = fac1*fac2
    
    elif par.p_phi == 0:

        par.w_phi[0] = 1

    else:
        
        par.w_phi[-1] = 1

    # b. psi
    if par.p_psi > 0 and par.p_psi < 1:
    
        if k == 1:
            
            par.w_psi[0] = 1-par.p_psi
            par.w_psi[1] = par.p_psi
        
        else:

            for s in range(k+1):
                if s == 0:
                    par.w_psi[s] = (1-par.p_psi)**k
                else:
                    fac1 = 1
                    for i,j in zip(range(k-s+1,k+1),range(1,s+1)):
                        fac1 *= i/j
                    fac2 = par.p_psi**s*(1-par.p_psi)**(k-s)
                    par.w_psi[s] = fac1*fac2
    
    elif par.p_psi == 0:

        par.w_psi[0] = 1

    else:
        
        par.w_psi[-1] = 1

    # c. xi
    if par.p_xi < 1:
        par.w_xi[:] = np.array([1-par.p_xi,par.p_xi])
    else:
        par.w_xi[:] = np.array([0.0,1.0])

    # d. eta
    if par.p_eta < 1:
        par.w_eta[:] = np.array([1-par.p_eta,par.p_eta])
    else:
        par.w_eta[:] = np.array([0.0,1.0])

# mean and variances
@njit(fastmath=True)
def mean_cond(par,n_phi,n_xi1,n_xi2,n_eta1,n_eta2):
    """ conditional mean given number of shocks """
    
    mean = n_phi*par.mu_phi
    if n_xi1 == 1: mean -= par.mu_xi
    if n_xi2 == 1: mean += par.mu_xi 
    if n_eta1 == 1: mean -= par.mu_eta
    if n_eta2 == 1: mean += par.mu_eta 

    return mean
    
@njit(fastmath=True)    
def var_cond(par,n_phi,n_psi,n_xi1,n_xi2,n_eta1,n_eta2):
    """ conditional mean given number of shocks """

    var = (n_xi1+n_xi2)*par.sigma_xi**2 + (n_eta1+n_eta2)*par.sigma_eta**2 + 2*par.sigma_epsilon**2

    if n_psi > 0:

        if par.rho < 1:
            var += 2*(1-par.rho**n_psi)*par.sigma_psi**2/(1-par.rho**2)
        else:
            var += n_psi*par.sigma_psi**2
    
    if n_phi > 0:

        var += n_phi*par.sigma_phi**2

    return var

# unconditional
def main_mixture_results(model,k,omegas):

    global skew, kurt, leq

    create_weights(model.par,k)

    with jit(model) as model:

        par = model.par
        sim = model.sim

        fill_weights(par,k)
        skew, kurt, leq = main_mixture_results_(par,sim,k,omegas)
    
    return skew, kurt, leq

@njit(fastmath=True)
def main_mixture_results_(par,sim,k,omegas):
    """ mixture results for skew, kurt and cdf """

    # a. calculate total moments
    mean_total = mean_d1ky(par,sim,k)
    var_total = var_d1ky(par,sim,k)

    # b. allocate and initialize
    skew = 0
    kurt = 0

    probs = np.zeros(omegas.size)
    leq = np.zeros(omegas.size)

    # c. loop
    for n_phi in range(k+1):
        for n_psi in range(k+1):
            for n_xi1 in range(2):
                for n_xi2 in range(2):
                    for n_eta1 in range(2):
                        for n_eta2 in range(2):

                            # i. weight
                            w = par.w_phi[n_phi]*par.w_psi[n_psi]*par.w_xi[n_xi1]*par.w_xi[n_xi2]*par.w_eta[n_eta1]*par.w_eta[n_eta2]
                            if w <= par.wmin: continue

                            # ii. mean and variance
                            mean = mean_cond(par,n_phi,n_xi1,n_xi2,n_eta1,n_eta2)
                            var = var_cond(par,n_phi,n_psi,n_xi1,n_xi2,n_eta1,n_eta2)
                            std = np.sqrt(var)
                            
                            # iii. skew and kurt
                            skew += w*(mean-mean_total)*(3*var+(mean-mean_total)**2)
                            kurt += w*(3*var**2 + 6*(mean-mean_total)**2*var + (mean-mean_total)**4)
                            
                            if omegas.size == 0: continue

                            # iv. shares
                            norm_cdf_vec(probs,omegas,mean,std)
                            leq += w*probs                         

    # d. return
    if var_total > 0:
        skew = skew/var_total**(3/2)
        kurt = kurt/var_total**2-3
    else:
        skew = 0
        kurt = 0

    return skew, kurt, leq

# conditional
def cond_mixture_results(model,k,omegas):

    leq = np.nan*np.ones((3,omegas.size))
    create_weights(model.par,k)

    try:

        with jit(model) as model:

            par = model.par
            sim = model.sim

            fill_weights(par,k)
            leq[:,:] = cond_mixture_results_(par,sim,k,omegas)

    except:

        pass

    return leq

@njit(fastmath=True)
def cond_mixture_results_(par,sim,k,omegas):
    """ mixture results conditional on lagged growth """ 

    # a. allocate and initialize
    midrange_lag = 0
    midrange_l_lag = 0
    midrange_u_lag = 0
    leq_cur = np.zeros(omegas.size)
    leq_cur_l = np.zeros(omegas.size)
    leq_cur_u = np.zeros(omegas.size)
    
    # b. loop
    for n_phi1 in range(k+1):
        for n_psi1 in range(k+1):
            for n_xi1 in range(2):
                for n_xi2 in range(2):
                    for n_eta1 in range(2):
                        for n_eta2 in range(2):   

                            # i. weight for lagged
                            w_lag = par.w_phi[n_phi1]*par.w_psi[n_psi1]*par.w_xi[n_xi1]*par.w_xi[n_xi2]*par.w_eta[n_eta1]*par.w_eta[n_eta2]
                            if w_lag <= par.wmin: continue

                            # ii. mean and var for lagged
                            mean_lag = mean_cond(par,n_phi1,n_xi1,n_xi2,n_eta1,n_eta2)
                            var_lag = var_cond(par,n_phi1,n_psi1,n_xi1,n_xi2,n_eta1,n_eta2)
                            std_lag = np.sqrt(var_lag)

                            # iii. lagged in range
                            std_lag = np.fmax(std_lag,1e-8)
                            prob_high_lag = norm_cdf(par.omega_cond_midrange,mean_lag,std_lag)
                            prob_low_lag = norm_cdf(-par.omega_cond_midrange,mean_lag,std_lag)
                            midrange_lag += w_lag*(prob_high_lag-prob_low_lag)
                            midrange_l_lag += w_lag*prob_low_lag
                            midrange_u_lag += w_lag*(1-prob_high_lag)

                            # iv. current
                            for n_psi2 in range(k+1):
                                for n_phi2 in range(k+1):
                                    for n_xi3 in range(2):
                                        for n_eta3 in range(2):

                                            # v. cov
                                            cov = -(n_xi2*par.sigma_xi**2 + n_eta2*par.sigma_eta**2 + par.sigma_epsilon**2)
                                            if par.rho < 1:
                                                cov += (par.rho**n_psi2-1)*(1-par.rho**n_psi1)/(1-par.rho**2)*par.sigma_psi**2

                                            # vi. total weight
                                            w = w_lag*par.w_phi[n_phi2]*par.w_psi[n_psi2]*par.w_xi[n_xi3]*par.w_eta[n_eta3]
                                            if w <= par.wmin: continue

                                            # vii. mean and var for current
                                            mean_cur = mean_cond(par,n_phi2,n_xi2,n_xi3,n_eta2,n_eta3)
                                            var_cur = var_cond(par,n_phi2,n_psi2,n_xi2,n_xi3,n_eta2,n_eta3)
                                            std_cur = np.sqrt(var_cur)
                                            
                                            # viii. shares
                                            std_lag = np.fmax(std_lag,1e-8)
                                            std_cur = np.fmax(std_cur,1e-8)
                                            
                                            for i in range(omegas.size):
                                                omega = omegas[i]
                                                prob_high_cur = binorm_cdf(par.omega_cond_midrange,omega,mean_lag,mean_cur,std_lag,std_cur,cov)
                                                prob_low_cur = binorm_cdf(-par.omega_cond_midrange,omega,mean_lag,mean_cur,std_lag,std_cur,cov)
                                                leq_cur[i] += w*(prob_high_cur-prob_low_cur)
                                                leq_cur_l[i] += w*prob_low_cur
                                                leq_cur_u[i] += w*(binorm_cdf(np.inf,omega,mean_lag,mean_cur,std_lag,std_cur,cov)-prob_high_cur)

    # c. return
    out = np.zeros((3,omegas.size))
    for i in range(omegas.size):
        if midrange_lag > 0:
            out[0,i] = leq_cur[i]/midrange_lag
            out[1,i] = leq_cur_l[i]/midrange_l_lag
            out[2,i] = leq_cur_u[i]/midrange_u_lag
        else:
            out[0,i] = 0
            out[1,i] = 0
            out[2,i] = 0
    
    return out

############
# 5. level #
############

def var_y_d12_diff(par,sim,k):
    """ difference in variance of levels """

    phi_term = 12*k*(par.p_phi*par.sigma_phi**2 + par.p_phi*(1-par.p_phi)*par.mu_phi**2)
    
    if np.isclose(par.rho,1.0):
        psi_term = 12*k*par.p_psi*par.sigma_psi**2
    else:
        psi_term = 0.0

    return phi_term+psi_term

def cov_y_y_d12_diff(par,sim,k):
    """ difference in variance of levels """

    if par.rho < 1:

        base = par.sigma_psi**2/(1-par.rho**2)
        fac_long = (1-par.p_psi*(1-par.rho))**(12+12*k)
        fac_short = (1-par.p_psi*(1-par.rho))**12

        return (fac_long-fac_short)*base
    
    else:
        
        return 0.0

###########
# 6. fast #
###########

def get_omegas(model,momname,k): 
    """ unpack omegas from moment specification """

    if momname in model.specs:
        return np.array([info['args'][1] for info in model.specs[momname] if info['args'][0] == k])
    else:
        return np.array([])

def fast(model,momname,args):
    """ function for calculating moments fast """
    
    # a. mixture results
    mixture_list = ['skew_d12ky','kurt_d12ky','skew_d12kyw','kurt_d12kyw','leq_d12ky']
    if momname in mixture_list:

        # i. k
        if momname in ['skew_d12ky','kurt_d12ky','skew_d12kyw','kurt_d12kyw']:
            k = args
        else:
            k = args[0]
        
        # ii. find omegas
        omegas = get_omegas(model,'leq_d12ky',k)

        # iii. compute
        skew, kurt, leq = main_mixture_results(model,12*k,omegas)

        # iv. unpack
        if 'skew_d12ky' in model.specs: model.moms[('skew_d12ky',k)] = skew
        if 'kurt_d12ky' in model.specs: model.moms[('kurt_d12ky',k)] = kurt
        if 'skew_d12kyw' in model.specs: model.moms[('skew_d12kyw',k)] = skew
        if 'kurt_d12kyw' in model.specs: model.moms[('kurt_d12kyw',k)] = kurt

        if 'leq_d12ky' in model.specs:
            for i,omega in enumerate(omegas):            
                model.moms[('leq_d12ky',(k,omega))] = leq[i]

        return True
    
    mixture_list = ['skew_d1ky','kurt_d1ky','skew_d1kyw','kurt_d1kyw','leq_d1ky']
    if momname in mixture_list:

        # i. k
        if momname in ['skew_d1ky','kurt_d1ky','skew_d1kyw','kurt_d1kyw']:
            k = args
        else:
            k = args[0]
        
        # ii. find omegas
        omegas = get_omegas(model,'leq_d1ky',k)

        # iii. compute
        skew, kurt, leq = main_mixture_results(model,1*k,omegas)

        # iv. unpack
        if 'skew_d1ky' in model.specs: model.moms[('skew_d1ky',k)] = skew
        if 'kurt_d1ky' in model.specs: model.moms[('kurt_d1ky',k)] = kurt
        if 'skew_d1kyw' in model.specs: model.moms[('skew_d1kyw',k)] = skew
        if 'kurt_d1kyw' in model.specs: model.moms[('kurt_d1kyw',k)] = kurt

        if 'leq_d1ky' in model.specs:
            for i,omega in enumerate(omegas):            
                model.moms[('leq_d1ky',(k,omega))] = leq[i]

        return True

    # b. cdf
    if momname == 'cdf_d12ky':

        # i. compute
        k = args[0]    
        omegas = model.par.omegas_cdf
        skew, kurt, leq = main_mixture_results(model,12*k,omegas)

        # ii. unpack
        for i,_omega in enumerate(omegas):
            model.moms[('cdf_d12ky',(k,i))] = leq[i]

        return True
    
    if momname == 'cdf_d1ky':

        # i. compute
        k = args[0]    
        omegas = model.par.omegas_cdf
        skew, kurt, leq = main_mixture_results(model,k,omegas)

        # ii. unpack
        for i,_omega in enumerate(omegas):
            model.moms[('cdf_d1ky',(k,i))] = leq[i]

        return True

    # c. leq_midrange
    if 'leq_d12ky_midrange' in momname:

        # i. compute
        k = args[0]
        omegas = get_omegas(model,momname,k).copy()
        leq = cond_mixture_results(model,12*k,omegas)
        
        # ii. unpack
        for i,omega in enumerate(omegas):
            model.moms[('leq_d12ky_midrange',(k,omega))] = leq[0,i]
            if ('leq_d12ky_midrange_l',(k,omega)) in model.specs: model.moms[('leq_d12ky_midrange_l',(k,omega))] = leq[1,i]
            if ('leq_d12ky_midrange_u',(k,omega)) in model.specs: model.moms[('leq_d12ky_midrange_u',(k,omega))] = leq[2,i]

        return True

    if 'leq_d1ky_midrange' in momname:

        # i. compute
        k = args[0] 
        omegas = get_omegas(model,momname,k).copy()
        leq = cond_mixture_results(model,1*k,omegas)
        
        # ii. unpack
        for i,omega in enumerate(omegas):
            model.moms[('leq_d1ky_midrange',(k,omega))] = leq[0,i]
            if ('leq_d1ky_midrange_l',(k,omega)) in model.specs: model.moms[('leq_d1ky_midrange_l',(k,omega))] = leq[1,i]
            if ('leq_d1ky_midrange_u',(k,omega)) in model.specs: model.moms[('leq_d1ky_midrange_u',(k,omega))] = leq[2,i]

        return True

    # d. cdf_midrange
    if momname == 'cdf_d12ky_midrange':

        # i. compute
        k = args[0] 
        omegas = np.flip(model.par.omegas_cdf).copy()
        leq = cond_mixture_results(model,12*k,omegas)
        leq0 = np.flip(leq[0,:])
        leq1 = np.flip(leq[1,:])
        leq2 = np.flip(leq[2,:])
        omegas = np.flip(omegas)

        # ii. unpack
        for i,_omega in enumerate(omegas):
            model.moms[('cdf_d12ky_midrange',(k,i))] = leq0[i]
            if ('cdf_d12ky_midrange_l',(k,i)) in model.specs: model.moms[('cdf_d12ky_midrange_l',(k,i))] = leq1[i]
            if ('cdf_d12ky_midrange_u',(k,i)) in model.specs: model.moms[('cdf_d12ky_midrange_u',(k,i))] = leq2[i]

        return True

    if momname == 'cdf_d1ky_midrange':

        # i. compute
        k = args[0] 
        omegas = np.flip(model.par.omegas_cdf).copy()
        leq = cond_mixture_results(model,1*k,omegas)
        leq0 = np.flip(leq[0,:])        
        leq1 = np.flip(leq[1,:])        
        leq2 = np.flip(leq[2,:])
        omegas = np.flip(omegas)

        # ii. unpack
        for i,_eta in enumerate(omegas):
            model.moms[('cdf_d1ky_midrange',(k,i))] = leq0[i]
            if ('cdf_d1ky_midrange_u',(k,i)) in model.specs: model.moms[('cdf_d1ky_midrange_l',(k,i))] = leq1[i]
            if ('cdf_d1ky_midrange_u',(k,i)) in model.specs: model.moms[('cdf_d1ky_midrange_u',(k,i))] = leq2[i]

        return True

    # did not find anything
    return False