from typing import List, Tuple 
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import utils

US_CASES_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
US_DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"

GLOBAL_CASES_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"   
GLOBAL_DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"    
    
class CovidDataset(Dataset):
    '''Base class'''
    def __init__(self, root: Path, download: bool = True, 
                 by: List = ["Combined_Key"], days_range: List = [], 
                 date_range: List = [], Admin2: List = [], 
                 Province_State: List = [],remove_negative_days: bool = False):
        self.root = root 
        self.download = download
        self.by = by
        self.days_range = days_range
        self.date_range = date_range
        self.Admin2 = Admin2
        self.Province_State = Province_State
        self.remove_negative_days = remove_negative_days   
        
    def date_filter_groupby(self, df):        
        if self.days_range:
            start_day, end_day = self.days_range
            dfs = []
            for item in df.groupby(self.by):
                if start_day is not None and end_day is not None:
                    indx = ((item[1]["num_days"]>= start_day) & 
                            (item[1]["num_days"]<= end_day))
                elif start_day is not None:
                    indx = item[1]["num_days"]>= start_day
                elif end_day is not None:
                    indx = item[1]["num_days"]<= end_day
                df_temp = item[1][indx] 
                dfs.append((item[0],df_temp))
        elif self.date_range:
            start_date, end_date = self.date_range
            dfs = []
            for item in df.groupby(self.by):
                if start_date is not None and end_date is not None:
                    indx = ((item[1]["date"]>= start_date) & 
                            (item[1]["date"]<= end_date))
                elif start_date is not None:
                    indx = item[1]["date"]>= start_date
                elif end_date is not None:
                    indx = item[1]["date"]<= end_date
                df_temp = item[1][indx] 
                dfs.append((item[0],df_temp))                                            
        else:
            dfs = [item for item in df.groupby(self.by)]
        
        return dfs
            
    @property
    def filename(self) -> str:
        pass
    
    def _download(self, url_path):        
        df_cases_raw = pd.read_csv(url_path)
        df_cases_raw.to_csv(self.filename, index = False)  
        return df_cases_raw            
    
    def __len__(self):
        return len(self.dfs)
    
    def __getitem__(self, ix):
        return self.dfs[ix]            
    
    
    
class USDataset(CovidDataset):  
    '''Base class'''
    
    def __init__(self, root: Path, download: bool = True, 
                 by: List = ["Combined_Key"], days_range: List = [], 
                 date_range: List = [], Admin2: List = [], 
                 Province_State: List = [],remove_negative_days: bool = False):
        self.root = root 
        self.download = download
        self.by = by
        self.days_range = days_range
        self.date_range = date_range
        self.Admin2 = Admin2
        self.Province_State = Province_State
        self.remove_negative_days = remove_negative_days           
        #CovidDataset.__init__(self,**kwargs)     
                
    def us_geo_filter(self, df):
        if self.Province_State:
            df = df[df.Province_State.isin(self.Province_State)]
        if self.Admin2:
            df = df[df.Admin2.isin(self.Admin2)]               
        if self.remove_negative_days:
            df = df[df.num_days>=0]                
        return df.reset_index(drop=True)
    
    @property
    def filename(self) -> str:
        pass                                     
        
        
    
class USDeathsDataset(USDataset):  
    def __init__(self, **kwargs):
        USDataset.__init__(self,**kwargs)
        if self.download:
            df_deaths_raw = self._download(US_DEATHS_URL)
        else:            
            df_deaths_raw = pd.read_csv(self.filename)            
        df = utils.prep_us_deaths(df_deaths_raw, by=self.by)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
        
    @property
    def filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_DEATHS_URL.rpartition('/')[-1])
        return filename

class USCasesDataset(USDataset):  
    def __init__(self, **kwargs):
        USDataset.__init__(self,**kwargs)
        if self.download:
            df_cases_raw = self._download(US_CASES_URL)
        else:            
            df_cases_raw = pd.read_csv(self.filename)            
        df = utils.prep_us_cases(df_cases_raw, by=self.by)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
        
    @property
    def filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_CASES_URL.rpartition('/')[-1])
        return filename        

class USCasesDeathsDataset(USDataset):  
    def __init__(self, **kwargs):
        USDataset.__init__(self, **kwargs)
        if self.download:
            df_cases_raw = self._download(US_CASES_URL)
            df_deaths_raw = self._download(US_DEATHS_URL)
        else:            
            df_cases_raw = pd.read_csv(self.us_cases_filename)
            df_deaths_raw = pd.read_csv(self.us_deaths_filename)

        df = utils.prep_us_cases_deaths(df_cases_raw, df_deaths_raw, by=self.by)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
                
        
    @property
    def us_cases_filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_CASES_URL.rpartition('/')[-1])
        return filename  
    
    @property
    def us_deaths_filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_DEATHS_URL.rpartition('/')[-1])
        return filename        
    
    
class GlobalDataset(CovidDataset):  
    '''Base class'''
    
    def __init__(self, root: Path, download: bool = True, 
                 by: List = ["Combined_Key"], days_range: List = [], 
                 date_range: List = [], Province_State: List = [], 
                 Country_Region: List = [],remove_negative_days: bool = False):
        self.root = root 
        self.download = download
        self.by = by
        self.days_range = days_range
        self.date_range = date_range
        self.Province_State = Province_State
        self.Country_Region = Country_Region
        self.remove_negative_days = remove_negative_days
                
    def global_geo_filter(self, df):
        if self.Province_State:
            df = df[df['Province/State'].isin(Province_State)]
        if self.Country_Region:
            df = df[df['Country/Region'].isin(self.Country_Region)]               
        if self.remove_negative_days:
            df = df[df.num_days>=0]                
        return df.reset_index(drop=True)
    
    @property
    def filename(self) -> str:
        pass   
    
    
class GlobalDeathsDataset(GlobalDataset):  
    def __init__(self, **kwargs):
        GlobalDataset.__init__(self,**kwargs)
        if self.download:
            df_deaths_raw = self._download(GOBAL_DEATHS_URL)
        else:            
            df_deaths_raw = pd.read_csv(self.filename)            
        df = utils.prep_us_deaths(df_deaths_raw, by=self.by)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
        
    @property
    def filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, GLOBAL_DEATHS.rpartition('/')[-1])
        return filename    
    
class GlobalCasesDataset(GlobalDataset):  
    def __init__(self, **kwargs):
        GlobalDataset.__init__(self,**kwargs)
        if self.download:
            df_cases_raw = self._download(GLOBAL_CASES_URL)
        else:            
            df_cases_raw = pd.read_csv(self.filename)            
        df = utils.prep_us_cases(df_cases_raw, by=self.by)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
        
    @property
    def filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_CASES_URL.rpartition('/')[-1])
        return filename     