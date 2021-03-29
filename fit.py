import os
import time
import itertools as it
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
colors = [x['color'] for x in plt.style.library['seaborn']['axes.prop_cycle']]
markers = ['s','P','D','v','^','*']

from labels import latex

latex['auto_cov_d12y12l'] = r'$\mathrm{Cov}[\Delta_{12}y_{t},\Delta_{12}y_{t-12\ell}]$'
latex['auto_cov_d24y24l'] = r'$\mathrm{Cov}[\Delta_{24}y_{t},\Delta_{24}y_{t-24\ell}]$'
latex['auto_cov_d36y36l'] = r'$\mathrm{Cov}[\Delta_{36}y_{t},\Delta_{36}y_{t-36\ell}]$'
latex['frac_auto_cov_d12y1l'] = r'$\mathrm{Cov}[\Delta_{12}y_{t},\Delta_{12}y_{t-\ell}]$'
latex['frac_auto_cov_d24y1l'] = r'$\mathrm{Cov}[\Delta_{24}y_{t},\Delta_{24}y_{t-\ell}]$'
latex['frac_auto_cov_d36y1l'] = r'$\mathrm{Cov}[\Delta_{36}y_{t},\Delta_{36}y_{t-\ell}]$'

##############
# 1. figures #
##############

def unpack(models):
    """ unpack based on whether input is list or not """

    if type(models) is list:
        model = models[0]
    else:
        model = models
        models = [model]

    return model,models

def mean_var_skew_kurt(models,momname,show=True,savefig=False,prefix='',ext='pdf',include_non_windsor=True):

    model,models = unpack(models)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    do_data = (momname,1) in model.datamoms
    do_var = ((momname,1),'var') in model.datamoms

    # a. unpack data
    ks = np.array([info['args'] for info in model.specs[momname]])

    if do_data:

        ydata = np.array([model.datamoms[(momname,k)] for k in ks]).ravel()
    
        if do_var:

            ydata_var = np.array([model.datamoms[((momname,k),'var')] for k in ks]).ravel()
    
        # b. plot - data
        ax.plot(ks,ydata,'-o',color='black',label='data')
            
        if do_var:
            [ax.plot(ks,ydata+sign*1.96*np.sqrt(ydata_var),'--',color='black',alpha=0.5,label='',zorder=0) for sign in [-1,1]]
        
        if ('yw' in momname or 'yaw' in momname) and include_non_windsor:
            ydata = np.array([model.datamoms[(momname.replace('yw','y').replace('yaw','ya'),k)] for k in ks]).ravel()
            ax.plot(ks,ydata,':',color='gray',label='data (raw)')

    # c. plot - model
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname,k)] for k in ks]).ravel()    
        ax.plot(ks,y,color=colors[i],marker=markers[i],label=model.latexname,zorder=len(models)-i)

    # d. details
    ax.set_xlabel('$k$')
    ax.set_xticks(ks)
    ylabel = latex[momname.replace('yw','y').replace('yaw','ya')]
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True,ncol=2,loc='best')

    ylims = ax.get_ylim()

    if not ('yaw' in momname or 'ya' in momname or 'skew' in momname):
        ax.set_ylim([0,ylims[1]])
    
    fig.tight_layout()
    if savefig: fig.savefig(f'figs/{prefix}fit_{momname}.{ext}')
    if show: plt.show()
    plt.close('all')

def auto_cov(models,momname,zoom=False,show=True,savefig=False,prefix='',ext='pdf',include_non_windsor=True):

    model,models = unpack(models)
    momname = f'auto_cov_{momname}'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    do_data = (momname,2) in model.datamoms
    do_var = ((momname,1),'var') in model.datamoms

    # a. unpack data
    ls = np.array([info['args'] for info in model.specs[momname]])
    if zoom:
        ls = ls[1:]

    if do_data:

        ydata = np.array([model.datamoms[(momname,l)] for l in ls]).ravel()
        
        if do_var:
            ydata_var = np.array([model.datamoms[((momname,l),'var')] for l in ls]).ravel()

        # b. plot - data
        ax.plot(ls,ydata,'-o',color='black',label='data')

        if do_var:
            [ax.plot(ls,ydata+sign*1.96*np.sqrt(ydata_var),'--',color='black',alpha=0.5,label='',zorder=0) for sign in [-1,1]]

        if ('yw' in momname or 'yaw' in momname) and include_non_windsor:
            ydata = np.array([model.datamoms[(momname.replace('yw','y').replace('yaw','ya'),l)] for l in ls]).ravel()
            ax.plot(ls,ydata,':',color='gray',label='data (raw)')

    # c. plot - models
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname,l)] for l in ls]).ravel()
        ax.plot(ls,y,color=colors[i],marker=markers[i],label=model.latexname,zorder=len(models)-i)

    # d. details
    ax.set_xlabel(r'$\ell$')
    ax.set_xticks(ls)
    ylabel = latex[momname.replace('yw','y').replace('yaw','ya')]
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True,ncol=2)

    fig.tight_layout()
    if not zoom:
        if savefig: fig.savefig(f'figs/{prefix}fit_{momname}.{ext}')
    else:
        if savefig: fig.savefig(f'figs/{prefix}fit_{momname}_zoom.{ext}')
    
    if show: plt.show()
    plt.close('all')

def frac_auto_cov(models,momname,show=True,savefig=False,prefix='',ext='pdf',include_non_windsor=True):

    model,models = unpack(models)
    momname = f'frac_auto_cov_{momname}'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    do_var = ((momname,1),'var') in model.datamoms

    # a. upack
    x = ls = np.array([info['args'] for info in model.specs[momname]])
    y = np.array([model.moms[(momname,l)] for l in ls]).ravel()
    ydata = np.array([model.datamoms[(momname,l)] for l in ls]).ravel()

    if do_var:
        ydata_var = np.array([model.datamoms[((momname,l),'var')] for l in ls]).ravel()

    # b. plot - data
    ax.plot(x,ydata,'-o',color='black',label='data')

    if do_var:
        [ax.plot(x,ydata+sign*1.96*np.sqrt(ydata_var),'--',color='black',alpha=0.5,label='',zorder=0) for sign in [-1,1]]

    if ('yw' in momname or 'yaw' in momname) and include_non_windsor:
        ydata = np.array([model.datamoms[(momname.replace('yw','y').replace('yaw','ya'),l)] for l in ls]).ravel()
        ax.plot(x,ydata,':',color='gray',label='data (raw)')

    # c. plot - model
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname,l)] for l in ls]).ravel()
        ax.plot(x,y,color=colors[i],marker=markers[i],label=model.latexname,zorder=len(models)-i)

    # d. details
    ax.set_xlabel(r'$\ell$')
    ylabel = latex[momname.replace('yw','y').replace('yaw','ya')]
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True,ncol=2)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/{prefix}fit_{momname}.{ext}')
    if show: plt.show()
    plt.close('all')

def var_diff_level(models,momname,show=True,savefig=False,prefix='',ext='pdf',include_non_windsor=True):

    model,models = unpack(models)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    do_data = (momname,1) in model.datamoms
    do_var = ((momname,1),'var') in model.datamoms

    # a. unpack data
    ks = np.array([info['args'] for info in model.specs[momname]])

    if do_data:

        ydata = np.array([model.datamoms[(momname,k)] for k in ks]).ravel()
        if do_var:

            ydata_var = np.array([model.datamoms[((momname,k),'var')] for k in ks]).ravel()
    
        # b. plot - data
        ax.plot(ks,ydata,'-o',color='black',label='data')
            
        if do_var:
            [ax.plot(ks,ydata+sign*1.96*np.sqrt(ydata_var),'--',color='black',alpha=0.5,label='',zorder=0) for sign in [-1,1]]
        
        if ('yw' in momname or 'yaw' in momname) and include_non_windsor:
            ydata = np.array([model.datamoms[(momname.replace('yw','y').replace('yaw','ya'),k)] for k in ks]).ravel()
            ax.plot(ks,ydata,':',color='gray',label='data (raw)')

    # c. plot - model
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname,k)] for k in ks]).ravel()    
        ax.plot(ks,y,color=colors[i],marker=markers[i],label=model.latexname,zorder=len(models)-i)

    # d. details
    ax.set_xlabel('$k$')
    ax.set_xticks(ks)
    ylabel = latex[momname.replace('yw','y').replace('yaw','ya')]
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True,ncol=2,loc='best')

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/{prefix}fit_{momname}.{ext}')
    if show: plt.show()
    plt.close('all')

###########
# 2. cdfs #
###########

def prepare_cdf(models,ks,load_data=True,sample='all'):
    """ add cdf moments to list of moments, re-calculate and re-load data """

    islist = type(models) is list
    model,models = unpack(models)
    
    # a. setup
    if islist:
        cdfmodels = []

    # b. compute
    for model in models:

        # i. copy
        cdfmodel = model.copy()
        cdfmodel.name = model.name
        cdfmodel.latexname = model.latexname

        # ii. omegas for cdf (same as on DST server)
        _omegas_cdf = np.logspace(-4,np.log(150)/np.log(10),50)
        cdfmodel.par.omegas_cdf = np.flip(np.concatenate((-np.flip(_omegas_cdf),_omegas_cdf)))/100

        # iii. extend moment specification
        Nomegas = cdfmodel.par.omegas_cdf.size

        if model.par.do_d12ky:

            _cdf_specs = {
                'cdf_d12ky': [{'args':(k,j)} for k,j in it.product(ks,range(Nomegas))],
                'cdf_d12ky_midrange': [{'args':(1,j)} for j in range(Nomegas)],
            }
            cdfmodel.specs = {**cdfmodel.specs,**_cdf_specs}

        if model.par.do_d1ky:

            _cdf_specs = {
                'cdf_d1ky': [{'args':(k,j)} for k,j in it.product(ks,range(Nomegas))],
                'cdf_d1ky_midrange': [{'args':(1,j)} for j in range(Nomegas)],
            }
            cdfmodel.specs = {**cdfmodel.specs,**_cdf_specs}

        # iv. load data again
        if load_data:
            cdfmodel.load_data(sample=sample,do_cov=False)

        # v. calculate moments
        cdfmodel.calc_moments()

        if islist:
            cdfmodels.append(cdfmodel)

    # c. return
    if islist:
        return cdfmodels
    else:
        return cdfmodel

def cdf(models,momname,k,show=True,savefig=False,prefix='',ext='pdf',showdata=True):

    model,models = unpack(models)
    momname = f'cdf_{momname}'
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # a. unpack
    x = model.par.omegas_cdf
    if showdata:
        ydata = np.array([model.datamoms[(momname,(k,j))] for j in range(model.par.omegas_cdf.size)])

    # b. plot - data
    if showdata:
        ax.plot(100*x,ydata,'-',lw=3,color='black',label='data')

    # c. plot - model
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname,(k,j))] for j in range(model.par.omegas_cdf.size)])
        ax.plot(100*x,y,color=colors[i],marker=markers[i],label=model.latexname,markersize=2,zorder=len(models)-i)

    # d. details
    if momname == 'cdf_d1ky':
        ax.set_xlabel(fr'$100 \cdot \Delta_{{{k}}} y_{{t}}$')
    elif momname == 'cdf_d12ky':
        ax.set_xlabel(fr'$100 \cdot \Delta_{{{12*k}}} y_{{t}}$')
    elif momname == 'cdf_d1kya':
        ax.set_xlabel(fr'$100 \cdot \Delta_{{{12*k}}} \overline{{y}}_{{s}}$')

    ax.set_xscale('symlog')
    ax.set_ylim([0,1])
    ax.set_ylabel('cdf')
    ax.legend(frameon=True,ncol=2)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/{prefix}fit_{momname}_{k}.{ext}')

    if show: plt.show()
    plt.close('all')

def cdf_midrange(models,momname,postfix='',show=True,savefig=False,prefix='',ext='pdf',showdata=True):

    model,models = unpack(models)
    momname = f'cdf_{momname}_midrange{postfix}'
    k = 1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    do_data = (momname,(k,0)) in model.datamoms

    # a. unpack
    x = model.par.omegas_cdf
    if do_data:

        if showdata:
            ydata = np.array([model.datamoms[(momname,(k,j))] for j in range(model.par.omegas_cdf.size)])

        # b. plot - data
        if showdata:
            ax.plot(100*x,ydata,'-',lw=3,color='black',label='data')
        
    # c. plot - model
    for i,model in enumerate(models):
        y = np.array([model.moms[(momname ,(k,i))] for i in range(model.par.omegas_cdf.size)])
        ax.plot(100*x,y,color=colors[i],marker=markers[i],label=model.latexname,markersize=2,zorder=len(models)-i)

    # d. details
    if momname in ['cdf_d1ky_midrange','cdf_d1ky_midrange_l','cdf_d1ky_midrange_u']:
        ax.set_xlabel(fr'$100 \cdot \Delta_{{{k}}} y_{{t}}$')
    elif momname in ['cdf_d12ky_midrange','cdf_d12ky_midrange_l','cdf_d12ky_midrange_u']:
        ax.set_xlabel(fr'$100 \cdot \Delta_{{{12*k}}} y_{{t}}$')
    
    ax.set_xscale('symlog')
    ax.set_ylim([0,1])
    if momname in ['cdf_d1ky_midrange','cdf_d1ky_midrange_l','cdf_d1ky_midrange_u']:
        ax.set_ylabel(fr'cdf cond. on $|\Delta_{{{k}}}y_{{t-{k}}}| \leq {model.par.omega_cond_midrange:.2f}$')
    elif momname in ['cdf_d12ky_midrange','cdf_d12ky_midrange_l','cdf_d12ky_midrange_u']:
        ax.set_ylabel(fr'cdf cond. on $|\Delta_{{{12*k}}}y_{{t-{12*k}}}| \leq {model.par.omega_cond_midrange:.2f}$')

    ax.legend(frameon=True,ncol=2)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/{prefix}fit_{momname}_{k}.{ext}')
    if show: plt.show()
    plt.close('all')

#########################
# 3. estimation results #
#########################

def estimation_results(modelbase,models,tab_name='combined'):

    # unpack
    par = modelbase.par
    theta = modelbase.theta
    Nmodels = len(models)
       
    # print to latex table
    with open(f'figs\\est_{tab_name}.tex','w') as file:

        # a. header
        file.write(f'\\begin{{tabular}}{{lc*{{{Nmodels}}}{{c}}}} \n')
        file.write('\\toprule \n')
        file.write(f' & & \\multicolumn{{{Nmodels}}}{{c}}{{Estimates}} \\\\ \n')
        file.write(f' \cmidrule(lr){{3-{Nmodels+2}}}  Parameters & ')
        for model in models:
            latexname = model.latexname.split(',')[0]
            file.write(f'& {latexname}')
        file.write('\\\\ \\midrule \\addlinespace \n')
        
        # b. main
        j = 0
        for key in theta.keys():
            
            # i. parameter name
            j+=1
            if key in latex:
                latex_name = latex[key](par)
            else:
                latex_name = 'Unknown, '

            name_1 = latex_name[0]
            name_2 = latex_name[1]
            
            if ( j > 1) & (name_1 != ''):
                file.write(f'[2mm]{name_1} & {name_2}')
            else:
                file.write(f'{name_1} & {name_2}')

            # ii. estimates
            for model in models:
                if key in model.theta:
                    value = model.est[key]
                    if np.abs(value) < 1e-4: value = 0.0 
                    file.write(f'& ${value:7.3f}$')
                else:
                    value = getattr(model.par,key)
                    if np.abs(value) < 1e-4: value = 0.0 
                    file.write(f'& ${value:7.3f}\dagger$')
                            
            # iii. standard errors
            file.write(f'\\\\ \n & ')
            for model in models:
                if key in model.theta:
                    value = model.est[(key,'se')]
                    file.write(f' & $({value:7.3f})$')
                else:
                    file.write(f' & $$')

            file.write('\\\\ \n')
        
        # c. footer
        file.write('\\addlinespace \midrule \\addlinespace \n')

        file.write(f'\\multicolumn{{2}}{{l}}{{Objective function}}')
        for model in models:
            value = model.est['obj']
            file.write(f'& ${value:7.4f}$')

        file.write('\\\\ \\addlinespace  \\bottomrule\n')
        file.write('\\end{tabular}\n')