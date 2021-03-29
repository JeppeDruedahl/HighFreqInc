from IPython.display import display
import pandas as pd

class SampleSelectionTable():
    
    def __init__(self,obs_fac=1):
        
        self.lines = {}
        self.obs_fac = obs_fac
    
    def add(self,rowlabel,df):
        
        N = df.index.unique(0).size
        obs = df.shape[0]*self.obs_fac
        
        self.lines[rowlabel] = [N,obs]

    def create_dataframe(self):

        lines = {}
        for i,(key,value) in enumerate(self.lines.items()):
            lines[f'{i}. {key}'] = value

        df = pd.DataFrame.from_dict(lines,orient='index',columns=['N','obs'])
        return df.rename(columns={'N':'Individuals','obs':'Observations'})

    def show(self):

        df = self.create_dataframe()
        df['Avg. obs. per individual'] = df['Observations']/df['Individuals']
        df['Percent of individuals'] = df['Observations']/df['Observations'].values[0]*100
        display(df)

    def latex(self,filename):

        df = self.create_dataframe()
        format_func = lambda x: f'{x:,d}'
        df.to_latex(filename,formatters=[format_func,format_func])