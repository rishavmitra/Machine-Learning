import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import Graphs as G

Raw_Data2=pd.read_csv('Raw_data2.csv')
Raw_Data2.info()

#Imputing missing values of Independent variables

#This function returns the names of the columns in the dataset having Dtype as int64 and float 64
#It is better than writing each column name again and again
#The eq() function gets equal to dataframe and others.

Raw_data_numcol = Raw_Data2.columns[Raw_Data2.dtypes.eq("float64")].values.tolist()
Raw_data_numcol2=Raw_Data2.columns[Raw_Data2.dtypes.eq("int64")].values.tolist()
Raw_data_numcol.extend(Raw_data_numcol2)
print(Raw_data_numcol)

impute_variable = SimpleImputer(missing_values = np.nan,strategy = 'median')
#Fitting phase calculates the median of all the columns passed as parameter and stores it
#Transformation phase does the actual action of locating the missing values and imputes them using the median strategy

Raw_Data2[Raw_data_numcol] = impute_variable.fit_transform(Raw_Data2[Raw_data_numcol])

#since zipcode is categorical we need to use another strategy
impute_variable = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')

#since we need to provide 2-D array for fit_transform
column = Raw_Data2['Zipcode'].values.reshape(-1,1)

Raw_Data2['Zipcode'] = impute_variable.fit_transform(column)







