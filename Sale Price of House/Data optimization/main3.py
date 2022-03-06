import numpy as np
import pandas as pd

Raw_Data3=pd.read_csv('Raw_Data3.csv')

#  Since Zipcode is a categorical value thus we set object as the dtype

Raw_Data3['Zipcode'] = Raw_Data3['Zipcode'].astype(object)

# The column No. of time visited contains categorical values but representing them in numerical form is more useful
# A dictionary is created to map the values of the column No. of time visited
mapping={'None':'0',
         'Once':'1',
         'Twice':'2',
         'Thrice':'3',
         'Four':'4'}
# The map function is used to map the variables
Raw_Data3['No of Times Visited'] = Raw_Data3['No of Times Visited'].map(mapping)
print(Raw_Data3['No of Times Visited'].unique())

Raw_Data3['Ever renovated'] = np.where(Raw_Data3['Renovated Year'] == 0,'No','Yes')

# Extraction the year part of the variable
Raw_Data3['Purchase Year'] = pd.DatetimeIndex(Raw_Data3['Date House was Sold']).year

# The abs function returns the absolute function of the number eg: abs(-2.5) gives 2.5
Raw_Data3['Years since renovation'] = np.where(Raw_Data3['Ever renovated'] ==  'Yes',
                                               abs(Raw_Data3['Purchase Year']-Raw_Data3['Renovated Year']),0)

# Year since renovation is the only column we need so columns date house was sold, Purchase year,renovated years can
# be dropped
Raw_Data3.drop(columns=['Purchase Year','Date House was Sold','Renovated Year','Unnamed: 0','Unnamed: 0.1'],inplace = True)
Raw_Data3.to_csv('Transformed.csv')

# To see the corelation we use the corr() function
print(Raw_Data3['Sale Price'].corr(Raw_Data3['Flat Area (in Sqft)']))
# To see the corelation of each and every variable
print(Raw_Data3.drop(columns = ['ID']).corr().to_string())

