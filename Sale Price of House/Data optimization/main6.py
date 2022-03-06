'''Simple Regression Model'''
import numpy as np
import pandas as pd
import Graphs as G
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Transformed2.csv')

data['Mean_Sales']=data['Sale_Price'].mean()

# Predicting the mean Price (Simplest Model)
#G.main6_graph(data)

# There are other factors that the sale price depends on eg: the overall grade
mean_grade = data.pivot_table(values='Sale_Price',columns='Overall Grade',aggfunc=np.mean)
# Storing the above mean grade in the dataframe
data['Grade mean'] = 0

# filling the Grade mean column with mean_grade according to their grade.
for i in mean_grade.columns:
    data['Grade mean'][data['Overall Grade']==i] = mean_grade[i][0]

# Calling the main6_graph2 function for plotting the overall grade mean sales price
#G.main6_graph2(data)

######################## Residual Plot ###############################
residual_mean_sales_difference = data['Mean_Sales']-data['Sale_Price']
residual_mean_grade_difference = data['Grade mean']-data['Sale_Price']

#G.main6_residual_graph(data,residual_mean_sales_difference,residual_mean_grade_difference)
#G.main6_residual_graph(data,residual_mean_sales_difference,residual_mean_grade_difference)
### From the Graphs above we see the second graph is more perfect
### Thus mean regression based on categorical datas are more useful for predicting the model

################ Mean error#####
cost=sum(residual_mean_sales_difference)/len(data)
print(cost)
################ Mean Absolute error#######
Y = data['Sale_Price']
Y_hat1 = data['Mean_Sales']
Y_hat2 = data['Grade mean']

# abs() is used so that the values don't cancel out each other since some have higher values and some lower in the
# negative range
cost1=sum(abs(Y_hat1-Y))/len(data)
cost2=sum(abs(Y_hat2-Y))/len(data)
print(cost1,cost2)
# We van see that cost2 is more efficient as it is categorical values
from sklearn.metrics import mean_absolute_error
cost_grade_mean = mean_absolute_error(Y_hat2,Y)
print(cost_grade_mean)

############# MSE #############
from sklearn.metrics import mean_squared_error
cost_mean = mean_squared_error(Y_hat1,Y)
cost_grade_mean = mean_squared_error(Y_hat2,Y)
print(cost_mean,cost_grade_mean)

############# RMSE ############
from sklearn.metrics import mean_squared_error
cost_mean = mean_squared_error(Y_hat1,Y)**0.5
cost_grade_mean = mean_squared_error(Y_hat2,Y)**0.5
print(cost_mean,cost_grade_mean)

############ R^2 ###############
Y_hat = data['Grade mean']
Y_bar = data['Mean_Sales']
Y = data['Sale_Price']
n = len(data)

mse_mean = mean_squared_error(Y_bar,Y)
mse_model = mean_squared_error(Y_hat,Y)

R2 = 1-(mse_model)/(mse_mean)
print('R^2 of the model=',R2)









































