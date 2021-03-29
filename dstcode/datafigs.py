import calendar
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
colors = [x['color'] for x in plt.style.library['seaborn']['axes.prop_cycle']]

def lifecycle(ys,ages,name=None,ylabel=None):
    
    # a. setup
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    
    # b. means
    y = ys.mean()    
    ax.plot(ages,y,'o',color='black',label=f'means',zorder=99)
    
    meanval = y.mean()
    ax.plot(ages,np.repeat(meanval,ages.size),lw=1,color='black',label='life-cycle mean',zorder=99)
    
    # d. by cohort
    for birthyear in ys.index.get_level_values('birthyear'):

        y = ys.loc[birthyear]
        ax.plot(ages,y,label='')

    # e. save
    ax.set_xlabel('age')
    if not ylabel == None: ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    
    fig.tight_layout()
    if not name==None: fig.savefig(f'figs/{name}.pdf')
    plt.show() 
    
def cdf(varlist,varnamelist,name=None,xlabel=None):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # a. cdf for each variable
    for var,varname in zip(varlist,varnamelist):
        
        I = ~np.isnan(var)
        x = var[I]
        x = x.sort_values()
        cdf = x.rank(method='average',pct=True)
        ax.plot(x*100,cdf,label=varname,lw=1)

    # b. details
    ax.set_xscale('symlog')
    ax.set_ylim([0,1])
    if not xlabel==None: ax.set_xlabel(xlabel)
    ax.set_ylabel('cdf')
    ax.legend(frameon=True)

    fig.tight_layout()
    if not name==None: fig.savefig(f'figs/{name}.pdf')
    plt.show()

def lifecycle_dist(df,varname,name=None,ylabel=None,ages=[35,40,45,50,55],ylim=(-1,1),clip=(-1.5,1.5)):

    # a. plot
    I = df.age.isin(ages)
    ax = sns.violinplot(x=df['age'][I],y=df[varname][I].clip(clip[0],clip[1]))

    # b. details
    if not ylabel==None: ax.set_ylabel(ylabel)
    plt.ylim(ylim)

    plt.tight_layout()
    if not name==None: plt.savefig(f'figs/{name}.pdf')
    plt.show()

def quantiles_time(df,varname,groupbyvar,name=None,xlabel=None,ylabel=None):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # a. calcualte quantiles
    quantiles = df.groupby(groupbyvar)[varname].quantile([0.1,0.25,0.5,0.75,0.9])
    
    # b. get x
    _y = quantiles.xs(0.1,level=1)
    x = _y.index.get_level_values(groupbyvar)

    # c. plot
    ax.fill_between(x,quantiles.xs(0.1,level=1),quantiles.xs(0.9,level=1),label='10th-90th')
    ax.fill_between(x,quantiles.xs(0.25,level=1),quantiles.xs(0.75,level=1),label='25th-75th')
    ax.plot(x,quantiles.xs(0.5,level=1),color='black',label='median')
    
    # d. details
    if not xlabel == None: ax.set_xlabel(xlabel)
    if not ylabel == None: ax.set_ylabel(ylabel)
    ax.legend(frameon=True)

    fig.tight_layout()
    if not name==None: fig.savefig(f'figs/{name}.pdf')
    plt.show()
        
def abs_leq_month(df,etas,use_months,ages):

    for eta in etas:

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        # a. name
        name = f'd1y_abs_leq_{eta}'
        
        # b. calculate share
        ys = df[df.age.between(ages[0],ages[-1])].groupby(['age','month']).d1y.apply(lambda x: (x.between(-eta/100,eta/100)).mean())
        
        # c. average (for selected months)
        y = ys[ys.index.get_level_values('month').isin(use_months)].groupby('age').mean()
        x = y.index.get_level_values('age')
        ax.plot(x,y,'o',color='black',label='mean',zorder=99)
        
        meanval = y.mean()
        ax.plot(x,np.repeat(meanval,x.size),lw=1,color='black',label='life-cycle mean',zorder=99)   
        
        # d. by month
        for i, month in enumerate(df.index.unique('month')):

            y = ys.xs(month,level='month').groupby('age').mean()
            x = y.index.get_level_values('age')

            monthname = calendar.month_abbr[month]
            if month in use_months:
                ax.plot(x,y,label=f'{monthname}')
            else:
                ax.plot(x,y,ls='--',label=f'{monthname}')

        # f. save
        ax.set_ylim([0,1.0])
        ax.set_xlabel('age')
        ax.set_ylabel(f'$|\Delta y_t| < {eta/100}$')
        ax.legend(ncol=4,frameon=True)
        fig.tight_layout()
        fig.savefig(f'figs/{name}.pdf')    