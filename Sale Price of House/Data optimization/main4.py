import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm

trans_data = pd.read_csv('Transformed.csv')

# value_counts() counts each unique values on the column
print(trans_data['Condition of the House'].value_counts())

'''print(trans_data.groupby('Condition of the House',)['Sale Price'].mean().plot(kind='bar'))'''


# ANOVA
# In python variables should not have spaces in between so we need to rename them

trans_data=trans_data.rename(columns = {'Sale Price':'Sale_Price',
                                        'Condition of the House':'Condition_of_house',
                                        'Ever renovated':'Ever_Renovated',
                                        'Waterfront View':'Waterfront_view'})

mod = ols('Sale_Price ~ Condition_of_house', data = trans_data).fit()
Anova_table = sm.stats.anova_lm(mod , typ=2)
print(Anova_table)

# Dummy Variables
trans_data= pd.get_dummies(trans_data,
                           columns = ['Condition_of_house','Waterfront_view','Ever_Renovated'],
                           drop_first = True)


Zip_table=trans_data.groupby('Zipcode').agg({'Sale_Price':'mean'}).sort_values('Sale_Price',ascending = True)
Zip_table['Zipcode_group']=pd.cut(Zip_table['Sale_Price'],bins=10,
                                  labels=['Zipcode_Group_0',
                                          'Zipcode_Group_1',
                                          'Zipcode_Group_2',
                                          'Zipcode_Group_3',
                                          'Zipcode_Group_4',
                                          'Zipcode_Group_5',
                                          'Zipcode_Group_6',
                                          'Zipcode_Group_7',
                                          'Zipcode_Group_8',
                                          'Zipcode_Group_9'
                                          ])
Zip_table=Zip_table.drop(columns='Sale_Price')
trans_data=pd.merge(trans_data,Zip_table,left_on='Zipcode',
                    how='left',
                    right_index= True)
trans_data=trans_data.drop(columns='Zipcode')


trans_data=pd.get_dummies(trans_data
                          ,columns=['Zipcode_group'],
                          drop_first=True)
print(trans_data.info())

trans_data.to_csv('Transformed2.csv')