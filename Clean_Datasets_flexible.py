import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy import interpolate
#from scipy.stats import norm

def clean_option_data(excel_file_path):    
    # file name
    df = pd.read_csv(excel_file_path, sep=',', parse_dates=True)
    columns_names = list(df.iloc[0])
    df.columns = columns_names
    df = df[1:]

    # Separate Calls from Puts
    calls = df.iloc[:,0:4]
    puts = df.iloc[:,7:11]

    # Reset Indices
    calls = calls.reset_index(drop = True)
    puts = puts.reset_index(drop = True)
    
    # Collect Strikes & Maturities (days) -> put them in separate column // FLEXIBLE
    strikes_collect = np.array([])
    mat_days_collect = np.array([])
    for i in calls.Strike:
        try:
            strikes_collect = np.append(strikes_collect, float(i))
            mat_days_collect = np.append(mat_days_collect, days)
        except:
            pass
            strikes_collect = np.append(strikes_collect, 'NaN')
            days = float(i.split()[1][1:-3])
            mat_days_collect = np.append(mat_days_collect, days)

    # Add Columns
    calls['Strike'] = strikes_collect
    calls['Maturity_days'] = mat_days_collect
    calls['Option_type'] = np.repeat('Call', len(calls))
    
    puts['Strike'] = strikes_collect
    puts['Maturity_days'] = mat_days_collect
    puts['Option_type'] = np.repeat('Put', len(puts))
    
    # Drop unwanted rows with NaN´s: former Info row
    calls.dropna(thresh=4, inplace=True)
    puts.dropna(thresh=4, inplace=True)
    
    # Rounding
    calls[['Bid', 'Ask']] = calls[['Bid', 'Ask']].apply(pd.to_numeric).round(5)
    puts[['Bid', 'Ask']] = puts[['Bid', 'Ask']].apply(pd.to_numeric).round(5)

    #calls.shape == puts.shape
    dataset = calls.append(puts, ignore_index=True)
    dataset[["Strike", "Maturity_days"]] = dataset[["Strike", "Maturity_days"]].apply(pd.to_numeric)
    
    return dataset


def clean_merged_data(stock):
    
    if stock == 'AMZN':
        dataset1 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/AMZN1.csv')
        dataset2 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/AMZN2.csv')
        dataset3 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/AMZN3.csv')
    if stock == 'FB': 
        dataset1 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/FB1.csv')
        dataset2 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/FB2.csv')
        dataset3 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/FB3.csv')
    
    if stock == 'MSFT': 
        dataset1 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/MSFT1.csv')
        dataset2 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/MSFT2.csv')
        dataset3 = clean_option_data('/Users/albanzapke/Desktop/optionsData_export/MSFT3.csv')
    
    merged_df = pd.concat([dataset1, dataset2, dataset3])
    return merged_df