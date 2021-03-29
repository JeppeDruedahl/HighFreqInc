import numpy as np

######################
# 1. latex - moments #
######################

latex = {}

latex['mean_d1ky'] = r'$\mathrm{E}[\Delta_{k}y_{t}]$'
latex['var_d1ky'] = r'$\mathrm{Var}[\Delta_{k}y_{t}]$'
latex['skew_d1ky'] = r'$\mathrm{Skew}[\Delta_{k}y_{t}]$'
latex['kurt_d1ky'] = r'$\mathrm{Kurt}[\Delta_{k}y_{t}]$'

latex['auto_cov_d1y1l'] = r'$\mathrm{Cov}[\Delta y_{t},\Delta y_{t-\ell}]$'

latex['leq_d1ky'] = r'$\mathrm{Pr}[\Delta y_{t} < \eta]$'
latex['leq_d1ky_midrange'] = r'$\mathrm{Pr}[\Delta_{k}y_{t} < \eta | |\Delta_{k}y_{t-k}| \leq 0.01]$'

latex['mean_d12ky'] = r'$\mathrm{E}[\Delta_{12k}y_{t}]$'
latex['var_d12ky'] = r'$\mathrm{Var}[\Delta_{12k}y_{t}]$'
latex['skew_d12ky'] = r'$\mathrm{Skew}[\Delta_{12k}y_{t}]$'
latex['kurt_d12ky'] = r'$\mathrm{Kurt}[\Delta_{12k}y_{t}]$'

latex['auto_cov_d12y12l'] = r'$\mathrm{Cov}[\Delta_{12}y_{t},\Delta_{12}y_{t-12\ell}]$'
latex['auto_cov_d24y24l'] = r'$\mathrm{Cov}[\Delta_{24}y_{t},\Delta_{24}y_{t-24\ell}]$'
latex['auto_cov_d36y36l'] = r'$\mathrm{Cov}[\Delta_{36}y_{t},\Delta_{36}y_{t-36\ell}]$'
latex['frac_auto_cov_d12y1l'] = r'$\mathrm{Cov}[\Delta_{12}y_{t},\Delta_{12}y_{t-\ell}]$'
latex['frac_auto_cov_d24y1l'] = r'$\mathrm{Cov}[\Delta_{24}y_{t},\Delta_{24}y_{t-\ell}]$'
latex['frac_auto_cov_d36y1l'] = r'$\mathrm{Cov}[\Delta_{36}y_{t},\Delta_{36}y_{t-\ell}]$'

latex['leq_d12ky'] = r'$\mathrm{Pr}[\Delta_{12k}y_{t} < \eta]$'
latex['leq_d12ky_midrange'] = r'$\mathrm{Pr}[\Delta_{12k}y_{t} < \eta | |\Delta_{12k}y_{t-12k}| \leq 0.01]$'

latex['var_y_d12_diff'] = r'$\mathrm{Var}[y_{t+k}]-\mathrm{Var}[y_t]$'
latex['cov_y_y_d12_diff'] = r'$\mathrm{Cov}[y_t,y_{t+12+12k}]-\mathrm{Cov}[y_t,y_{t+12}]$'

#########################
# 2. latex - parameters #
#########################

latex['obj'] = lambda par: ('','objective')

latex['p_psi'] = lambda par: ('Prob. of persistent shock',f'$p_{{\\psi}}$')
latex['p_xi'] = lambda par: ('Prob. of transitory shock',f'$p_{{\\xi}}$')
latex['p_phi'] = lambda par: ('Prob. of permanent shock',f'$p_{{\\phi}}$')
latex['p_eta'] = lambda par: ('Prob. of mean-zero transitory shock',f'$p_{{\\eta}}$')

latex['sigma_psi'] = lambda par: ('Std. of persistent shock',f'$\\sigma_{{\\psi}}$')
latex['sigma_xi'] = lambda par: ('Std. of transitory shock',f'$\\sigma_{{\\xi}}$')
latex['sigma_phi'] = lambda par: ('Std. of permanent shock',f'$\\sigma_{{\\phi}}$')
latex['sigma_eta'] = lambda par: ('Std. of mean-zero transitory shock',f'$\\sigma_{{\\eta}}$')
latex['sigma_epsilon'] = lambda par: ('Std. of ever-present shock',r'$\sigma_{\epsilon}$')

latex['rho'] = lambda par: ('Persistence',f'$\\rho$')

latex['mu_phi'] = lambda par: ('Mean of permanent shock',f'$\\mu_{{\\phi}}$')
latex['mu_xi'] = lambda par: ('Mean of transitory shock',f'$\\mu_{{\\xi}}$')
latex['mu_eta'] = lambda par: ('Mean of mean-zero transitory shock',f'$\\mu_{{\\eta}}$')