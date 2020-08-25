from typing import List, Tuple 
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
#from . import utils

US_CASES_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
US_DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"

GLOBAL_CASES_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"   
GLOBAL_DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"    
    
class CovidDataset(Dataset):
    '''
    root : Path
        where to save/load data
    download: bool (default True)
        download data

    '''    
    def __init__(self, root: Path, download: bool = True, 
                 by: List = ["Combined_Key"], days_range: List = [], 
                 date_range: List = [], Admin2: List = [], 
                 Province_State: List = [], days_since: int = 0, 
                 remove_negative_days: bool = False) -> None:

        self.root = root 
        self.download = download
        self.by = by
        self.days_range = days_range
        self.date_range = date_range
        self.days_since = days_since
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

    def melt_data(self, df: pd.DataFrame):
        df_m = pd.melt(df, id_vars = self.id_vars)
        df_m["variable"] = pd.to_datetime(df_m["variable"])
        df_m.rename(columns = {"variable": "date", "value": self.target_name}, 
                          inplace=True)    
        def num_days_since(df_x):
            indx = df_x[self.target_name] >= self.days_since
            thresh_date = df_x[indx].date.min()
            num_days = (df_x.date - thresh_date).dt.days        
            return num_days.rename("num_days")       
        num_days =  df_m.groupby(self.by).apply(lambda x: num_days_since(x))
        num_days.index = num_days.index.get_level_values(-1)    
        df_m = pd.merge(df_m, num_days, left_index=True, 
                              right_index=True)

        num_new = (df_m.groupby(self.by)[self.target_name].
                         apply(lambda x: x.diff()).
                         rename(self.target_diff))
        num_new.index = num_new.index.get_level_values(-1)    
        df_m = pd.merge(df_m, num_new, left_index=True, 
                              right_index=True)

        return df_m        

        
    
    
    
class USDataset(CovidDataset):  
    '''
    Admin2: List (default empty)
        Filter data to specific list of Admin2 
    Province_State: List (default empty)
        Filter data to specific list of Procince_State
    '''    
    
    def __init__(self,Admin2: List = [], Province_State: List = [], 
                 **kwargs) -> None:
        CovidDataset.__init__(self, **kwargs)    
        self.Admin2 = Admin2
        self.Province_State = Province_State         
        
                
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
    def __init__(self, **kwargs) -> None:
        USDataset.__init__(self,**kwargs)
        if self.download:
            df_deaths_raw = self._download(US_DEATHS_URL)
        else:            
            df_deaths_raw = pd.read_csv(self.filename)   
            
            
        self.id_vars = ['UID',
                        'iso2',
                        'iso3',
                        'code3',
                        'FIPS',
                        'Admin2',
                        'Province_State',
                        'Country_Region',
                        'Lat',
                        'Long_',
                        'Combined_Key',
                        'Population']
        self.target_name = "num_deaths"
        self.target_diff = "new_deaths"            
            
        df = self.melt_data(df_deaths_raw)
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)         
        
    @property
    def filename(self) -> str:
        folder_path = os.path.abspath(self.root)
        os.makedirs(folder_path, exist_ok = True)
        filename = os.path.join(folder_path, US_DEATHS_URL.rpartition('/')[-1])
        return filename

class USCasesDataset(USDataset):  
    def __init__(self, **kwargs) -> None:
        USDataset.__init__(self, **kwargs)
        if self.download:
            df_cases_raw = self._download(US_CASES_URL)
        else:            
            df_cases_raw = pd.read_csv(self.filename)  
            
        self.id_vars = ['UID',
                        'iso2',
                        'iso3',
                        'code3',
                        'FIPS',
                        'Admin2',
                        'Province_State',
                        'Country_Region',
                        'Lat',
                        'Long_',
                        'Combined_Key']   
        self.target_name = "num_cases"
        self.target_diff = "new_cases"            
            

        df = self.melt_data(df_cases_raw)
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
                    
        df = self.merge_cases_deaths(df_cases_raw, df_deaths_raw)    
        df = self.us_geo_filter(df)
        self.dfs = self.date_filter_groupby(df)    
        
        
    def merge_cases_deaths(self, df_cases_raw, df_deaths_raw):
        self.id_vars = ['UID',
                        'iso2',
                        'iso3',
                        'code3',
                        'FIPS',
                        'Admin2',
                        'Province_State',
                        'Country_Region',
                        'Lat',
                        'Long_',
                        'Combined_Key']   
        self.target_name = "num_cases"
        self.target_diff = "new_cases" 
        df_cases = self.melt_data(df_cases_raw)
        
        self.id_vars = ['UID',
                        'iso2',
                        'iso3',
                        'code3',
                        'FIPS',
                        'Admin2',
                        'Province_State',
                        'Country_Region',
                        'Lat',
                        'Long_',
                        'Combined_Key',
                        'Population']
        self.target_name = "num_deaths"
        self.target_diff = "new_deaths"                        
        df_deaths = self.melt_data(df_deaths_raw)      
        
        df_us = pd.merge(df_cases[["Province_State", "Admin2", 
                                     "Combined_Key", "date", "num_cases", 
                                     "num_days", "new_cases"]],
                         df_deaths[["Province_State", "Admin2", 
                                     "Combined_Key", "date", "Population", 
                                     "num_deaths", "new_deaths"]],
                         left_on = ["Province_State", "Admin2", "Combined_Key", 
                                    "date"],
                         right_on =  ["Province_State", "Admin2", "Combined_Key", 
                                      "date"])    
        return df_us        

                
        
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
        CovidDataset.__init__(self,**kwargs)     
        self.Province_State = Province_State
        self.Country_Region = Country_Region
                
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