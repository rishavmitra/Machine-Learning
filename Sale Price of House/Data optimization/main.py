import pandas as pd
import Graphs as g

#this function is used to treat outliers by imputing them

def imputing(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value

Raw_Data = pd.read_csv('1. Regression - Module - (Housing Prices).csv')

'''print(Raw_Data.head(5).to_string())'''
'''print(Raw_Data['Sale Price'].head(10))'''
'''print(Raw_Data['Sale Price'].describe())'''

'''plt.scatter(y = Raw_Data['Sale Price'],x = Raw_Data['ID'], color='red')
plt.xlabel('Sale Price')
plt.ylabel('ID')
plt.show()'''

'''g.boxplt(Raw_Data['Sale Price'])'''

q1=Raw_Data['Sale Price'].quantile(0.25)
q3=Raw_Data['Sale Price'].quantile(0.75)

iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr

Raw_Data['Sale Price'] = Raw_Data['Sale Price'].apply(imputing)
print(Raw_Data['Sale Price'].describe())

# Treating this missing values(Reduces the performance of the model)
# Deleting the whole row having missing values
# Imputing the missing values by replacing them with the average values or mean values or mode values.

# the dropna dunction is used to delete any row or column having missing values.
Raw_Data.dropna(axis = 0 , subset=['Sale Price'] , inplace = True , how = 'any')
Raw_Data.info()

g.hist_gram(Raw_Data['Sale Price'])

'''Raw_Data.to_csv('Raw_data2.csv')<-------------missing values and outliers have been treated'''
