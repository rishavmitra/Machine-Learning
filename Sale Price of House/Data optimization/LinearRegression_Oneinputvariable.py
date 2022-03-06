'''Linear Regression Model'''
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Transformed2.csv')

sale_price = data['Sale_Price'].head(30)
flat_area = data['Flat Area (in Sqft)'].head(30)
sample_data = pd.DataFrame({'Sale_Price':sale_price,'Flat_Area':flat_area})

## Creating a mean regression model
sample_data['mean_sale_price'] = sample_data.Sale_Price.mean()

plt.figure(dpi=100)
plt.ticklabel_format(style='plain')
plt.scatter(sample_data.Flat_Area.sort_values(),sample_data.Sale_Price.sort_values(),color='red')
plt.plot(sample_data.Flat_Area,sample_data.mean_sale_price,color='yellow',label='Mean sale price')
plt.xlabel("Flat Area")
plt.ylabel('Sale Price')
plt.legend()
plt.show()

## Using the y=mx+c formula to plot the linear regression model
# Values got after repeated adjustments(started with m=0 and c=0)
c=39428
m=219
line=[]
for i in range(len(sample_data)):
    line.append(sample_data.Flat_Area[i]*m+c)
plt.figure(dpi=100)
plt.ticklabel_format(style='plain')
plt.scatter(sample_data.Flat_Area,sample_data.Sale_Price,color='red')
plt.plot(sample_data.Flat_Area,line,color='black',label='mx+c')
plt.xlabel('Flat Area')
plt.ylabel('Sale Price')
plt.legend()

from sklearn.metrics import mean_squared_error as mse
MSE = mse(sample_data.Sale_Price,line)
plt.title('Slope '+str(m)+' MSE '+str(MSE))
plt.show()

# We can see that the line fits the graph properly when we increase the values of m.
# We can write a function that helps the line to fit the data properly.
def slope_func(slope,intercept,sample_data):
    sale = []
    for i in range(len(sample_data.Flat_Area)):
        tmp=sample_data.Flat_Area[i]*slope+intercept
        sale.append(tmp)
    MSE = mse(sample_data.Sale_Price,sale)
    return MSE

slope=[i/10 for i in range(0,5000)]
Cost=[]
for i in slope:
    cost=slope_func(slope=i,intercept=0,sample_data=sample_data)
    Cost.append(cost)

'''Arranging in Dataframe'''
cost_frame=pd.DataFrame({
    'Slope':slope,
    'Cost':Cost
})
# Plotting the cost data according to every beta

plt.plot(cost_frame.Slope,cost_frame.Cost,color='blue',label='Cost function curve')
plt.xlabel('Slope')
plt.ylabel('Cost')
plt.legend()
plt.show()
# from here we get the best value of slope
# To get the best value of intercept we do the same
def intercept_error(slope,intercept,sample_data):
    sale = []
    for i in range(len(sample_data.Flat_Area)):
        tmp=sample_data.Flat_Area[i]*slope+intercept
        sale.append(tmp)
    MSE = mse(sample_data.Sale_Price,sale)
    return MSE

intercept = [i for i in range(5000,50000)]
Cost = []
for i in intercept:
    # slope=234 we get from the graph
    cost=intercept_error(slope=234,intercept=i,sample_data=sample_data)
    Cost.append(cost)

cost_frame=pd.DataFrame({
    'intercept':intercept,
    'cost':Cost
})

plt.plot(cost_frame.intercept,cost_frame.cost,color='blue',label='Cost function curve')
plt.xlabel('Intercept')
plt.ylabel('Cost')
plt.legend()
plt.show()

# We get the optimum value of c as 10834

################################################## Refer to google drive linear regression############################