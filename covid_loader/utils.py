import pandas as pd


def prep_us_deaths(df_death, k = 5, by = ["Combined_Key"]) -> pd.DataFrame:
    death_id_vars = ['UID',
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

    df_death_m = pd.melt(df_death, id_vars=death_id_vars)
    df_death_m["variable"] = pd.to_datetime(df_death_m["variable"])
    df_death_m.rename(columns = {"variable": "date", "value": "num_deaths"}, inplace=True)

    def num_days_since(df_x):
        indx = df_x["num_deaths"] >= k
        thresh_date = df_x[indx].date.min()
        num_days = (df_x.date - thresh_date).dt.days        
        return num_days.rename("num_days")       
    num_days =  df_death_m.groupby(by).apply(lambda x: num_days_since(x))
    num_days.index = num_days.index.get_level_values(-1)    
    df_death_m = pd.merge(df_death_m, num_days, left_index=True, 
                          right_index=True)
    
    num_new_death = (df_death_m.groupby(by)["num_deaths"].
                     apply(lambda x: x.diff()).
                     rename("new_deaths"))
    num_new_death.index = num_new_death.index.get_level_values(-1)    
    df_death_m = pd.merge(df_death_m, num_new_death, left_index=True, 
                          right_index=True)    
    
    return df_death_m

def prep_us_cases(df_cases, k = 10, by = ["Combined_Key"]) -> pd.DataFrame:
    cases_id_vars = ['UID',
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
    df_cases_m = pd.melt(df_cases, id_vars=cases_id_vars)
    df_cases_m["variable"] = pd.to_datetime(df_cases_m["variable"])
    df_cases_m.rename(columns = {"variable": "date", "value": "num_cases"}, 
                      inplace=True)    
    def num_days_since(df_x):
        indx = df_x["num_cases"] >= k
        thresh_date = df_x[indx].date.min()
        num_days = (df_x.date - thresh_date).dt.days        
        return num_days.rename("num_days")       
    num_days =  df_cases_m.groupby(by).apply(lambda x: num_days_since(x))
    num_days.index = num_days.index.get_level_values(-1)    
    df_cases_m = pd.merge(df_cases_m, num_days, left_index=True, 
                          right_index=True)
    
    num_new_cases = (df_cases_m.groupby(by)["num_cases"].
                     apply(lambda x: x.diff()).
                     rename("new_cases"))
    num_new_cases.index = num_new_cases.index.get_level_values(-1)    
    df_cases_m = pd.merge(df_cases_m, num_new_cases, left_index=True, 
                          right_index=True)
    
    return df_cases_m

def prep_us_cases_deaths(df_cases, df_death, k = 10, 
                         by = ["Combined_Key"]) -> pd.DataFrame:
    
    df_cases_m = prep_us_cases(df_cases, k = k, by = by)
    df_death_m = prep_us_deaths(df_death, k = k, by = by)

    df_us = pd.merge(df_cases_m[["Province_State", "Admin2", 
                                 "Combined_Key", "date", "num_cases", 
                                 "num_days", "new_cases"]],
                     df_death_m[["Province_State", "Admin2", 
                                 "Combined_Key", "date", "Population", 
                                 "num_deaths", "new_deaths"]],
                     left_on = ["Province_State", "Admin2", "Combined_Key", 
                                "date"],
                     right_on =  ["Province_State", "Admin2", "Combined_Key", 
                                  "date"])    
    return df_us



def prep_global_deaths(df_death, k = 5, 
                       by = ['Province/State', 'Country/Region']
                      ) -> pd.DataFrame:
    death_id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']

    df_death_m = pd.melt(df_death, id_vars=death_id_vars)
    df_death_m["variable"] = pd.to_datetime(df_death_m["variable"])
    df_death_m.rename(columns = {"variable": "date", "value": "num_deaths"}, inplace=True)

    def num_days_since(df_x):
        indx = df_x["num_deaths"] >= k
        thresh_date = df_x[indx].date.min()
        num_days = (df_x.date - thresh_date).dt.days        
        return num_days.rename("num_days")       
    num_days =  df_death_m.groupby(by).apply(lambda x: num_days_since(x))
    num_days.index = num_days.index.get_level_values(-1)    
    df_death_m = pd.merge(df_death_m, num_days, left_index=True, 
                          right_index=True)
    
    num_new_death = (df_death_m.groupby(by)["num_deaths"].
                     apply(lambda x: x.diff()).
                     rename("new_deaths"))
    num_new_death.index = num_new_death.index.get_level_values(-1)    
    df_death_m = pd.merge(df_death_m, num_new_death, left_index=True, 
                          right_index=True)    
    
    return df_death_m