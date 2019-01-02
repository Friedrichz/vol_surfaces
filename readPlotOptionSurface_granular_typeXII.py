import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy import interpolate
#from scipy.stats import norm
import Clean_Datasets_flexible

def readNPlot(stock, otype): 
    
    option_type = otype 
    deltaK = 2.5

    # Choose Data & define strikes and maturities -> calibrate for smooth surface
    # The Grids need adjustments to get a clean surface around the end points
    if (stock == 'APPL'):
        excel_file = '/Users/albanzapke/Desktop/Comp_Methods_Hirsa/data_apple.xlsx'
        df = pd.read_excel(excel_file)
        strikes = np.arange(170., 210 + deltaK , deltaK)
        #np.arange(min(df.Strike.unique()), max(df.Strike.unique()) + deltaK , deltaK)
        maturities = np.sort(df.Maturity_days.unique())
    
    # STILL NEED TO THINK ABOUT CLEAN AMZN SURFACE (!)
    elif (stock == 'AMZN'):
        df = Clean_Datasets_flexible.clean_merged_data(stock)
        strikes = np.arange(min(df.Strike.unique()), max(df.Strike.unique()) + deltaK , deltaK)
        maturities = np.sort(df.Maturity_days.unique())
        
    elif (stock == 'FB'):
        df = Clean_Datasets_flexible.clean_merged_data(stock)
        strikes = np.arange(min(df.Strike.unique()), 205 + deltaK , deltaK)
        maturities = np.sort(df.Maturity_days.unique())
        
    elif (stock == 'MSFT'):
        df = Clean_Datasets_flexible.clean_merged_data(stock)
        if option_type == 'Put':
            strikes = np.arange(min(df.Strike.unique()), 142.5 + deltaK , deltaK)
        elif option_type == 'Call':
            strikes = np.arange(85., max(df.Strike.unique()) + deltaK , deltaK)
        maturities = np.sort(df.Maturity_days.unique())
    
    # create the 'Mid' variable
    df['Mid'] = df[['Bid','Ask']].mean(axis=1)
    
    # 1.) FIRST YOU DO INTERPOLATION FOR STRIKES with fixed Maturity
    df_calls = df[df['Option_type'] == option_type][['Maturity_days', 'Strike', 'Mid']]
    df_calls.head()
    
    # define a grid for the surface
    X, Y = np.meshgrid(strikes, maturities)
    callPricesI1 = np.empty([len(maturities), len(strikes)])
    
    # we use linear interpolation for missing strikes
    for i in range(len(maturities)):
        s = df_calls[df_calls.Maturity_days == maturities[i]]['Strike']
        price = df_calls[df_calls.Maturity_days == maturities[i]]['Mid']
        f = interpolate.interp1d(s, price, bounds_error=False, fill_value='extrapolate') 
        callPricesI1[i, :] = f(strikes) 

    # 2.) NOW I WANT TO INTERPOLATE FOR MATURITIES and fixed strikes 
    # Use the matrix interpolated for strikes as departing point
    callPricesI1 = pd.DataFrame(callPricesI1.transpose(), columns = np.sort(df.Maturity_days.unique()))
    
    # Define granularity of grid on the maturity axis: -> here weekly 
    #
    deltaT = 1/52 
    #maturities_years = maturities/365
    #maturities_weeks = maturities/7
    #lenT = int(52*2 - maturities_weeks[0])    # approx 2 Years
    #maturities = (maturities_years[0] + deltaT * np.arange(lenT)[:-4]) 
    # Here in weekly steps! [:-4] because the last 4 entries give 0

     
    # Define granularity of grid on the maturity axis: -> here weekly 
    deltaT = 1
    wm = maturities/7
    lenT = wm[-1]-wm[0]
    maturities_weeks = wm[0] + deltaT * np.arange(lenT+1) 
    maturities_year = maturities_weeks/52

    # However we cannot index floating maturities, thus need integer values!
    int_maturities = []
    for i in maturities_weeks:
        int_maturities = np.append(int_maturities, int(i*7))

    # define the new and final grid for the second more granular surface
    realPrices = pd.DataFrame(np.empty([len(strikes), len(int_maturities)]).fill(np.nan), columns = int_maturities, index=strikes)
    realPrices = realPrices.transpose()
    
    for row in df_calls.itertuples():
            strike = row[2]
            maturity = row[1]
            price = row[3]
            try: 
                realPrices.loc[maturity,strike] = price
            except:
                  pass
    realPrices = realPrices.reindex(columns=sorted(realPrices.columns))
    
    # However we cannot index floating maturities, thus need integer values!
    #int_maturities = []
    #for i in maturities:
    #    int_maturities = np.append(int_maturities, int(i*365))
    
    # define the new and final grid for the second more granular surface
    X, Y = np.meshgrid(strikes, int_maturities)
    callPricesI2 = np.empty([len(strikes), len(int_maturities)])
    
    # we use linear interpolation for missing maturities
    for i in range(len(strikes)):
        s = np.sort(df.Maturity_days.unique())                           # all maturities we have
        price = callPricesI1.iloc[i]
        f = interpolate.interp1d(s, price, bounds_error=False, fill_value='extrapolate')
        callPricesI2[i, :] = f(int_maturities) 
        
    fig = plt.figure(figsize=(12.,12.))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(int_maturities, strikes)
    ax.plot_surface(X, Y, callPricesI2)
    ax.set_xlabel('maturity T')
    ax.set_ylabel('strike K')
    ax.set_zlabel( option_type + ' prices')
    if (option_type == 'Call'): 
        ax.view_init(30, 150)
    elif (option_type == 'Put'): 
        ax.view_init(30, 200)
    #plt.show()
    
    return int_maturities, strikes, callPricesI2, realPrices.transpose().values