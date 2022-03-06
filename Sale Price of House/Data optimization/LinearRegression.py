'''Linear Regression Model'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Transformed2.csv')

sale_price = data['Sale_Price'].head(30)
flat_area = data['Flat Area (in Sqft)'].head(30)
sample_data = pd.DataFrame({'Sale_Price':sale_price,'Flat_Area':flat_area})

## Creating a mean regression model
sample_data['mean_sale_price'] = sample_data.Sale_Price.mean()

'''Implementing Gradient Descent algorithm to find m and c for more than two input variables'''


def param_init(Y):
    '''This function only takes target variable and returns the parameters'''
    m = 0.1
    c = Y.mean()
    return m,c

def generate_pred(m,c,X):
    '''X = independent variables
        returns the predicted values generated with parameters m and c'''
    prediction=[]
    for x in X:
        predict = (m * x) + c
        prediction.append(predict)
    return prediction

def compute_cost(prediction,Y):
    '''This function returns the cost(mean squared error) between the predictions and Y'''
    cost=np.sum(((prediction-Y)**2)/len(Y))
    return cost

def gradients(predictions,Y,X):
    '''Returns the Gradient cost functions with respect to m and c '''
    n=len(Y)
    Gm=(2/n)*np.sum((predictions-Y)*X)
    Gc = (2 / n) * np.sum((predictions - Y))
    return Gm,Gc

def param_update(m_old , c_old , Gm_old , Gc_old , alpha):
    '''Update and return the new values of m and c'''
    m_new = m_old - (alpha*Gm_old)
    c_new = c_old - (alpha*Gc_old)
    return m_new,c_new

def result(m,c,X,Y,cost,predictions,i):
    '''Print and plot the final result from gradient descent'''

    ## If the Gradient descent converged to the maximum value before the maximum iteration
    if i < max_iter -1:
        print('****Gradient descent has converged at iteration {}****'.format(i))
    else:
        print('Result after ',max_iter,' iteration is ****')

    ## plotting the final result
    plt.figure(dpi=100)
    plt.scatter(X,Y,color='red',label='data points')
    label = 'final regression line: m = {}; c = {}'.format(str(m), str(c))
    plt.plot(X,predictions,color='green',label=label)
    plt.xlabel('flat area')
    plt.ylabel('Sale price')
    plt.title('Final regression')
    plt.legend()
    plt.show()

''' This scaling to done to prevent the dataset from exploding'''
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
## Defining and Reshaping the dataset
sale_price = scalar.fit_transform(sample_data['Sale_Price'].values.reshape(-1,1))
flat_area = scalar.fit_transform(sample_data['Flat_Area'].values.reshape(-1,1))

##declaring the parameters
max_iter = 1000
cost_old = 0
alpha = 0.01

## Step 1 = Initializing the values of m and c
m,c = param_init(sale_price)

## Gradient Descent in action
for i in range(0,max_iter):
    ## Step 2 = Genarating Predictions
    predictions = generate_pred(m,c,flat_area)

    ## Step 3 = Calculating cost
    cost_new = compute_cost(predictions , sale_price)

    ## checking if GD converged
    if abs(cost_new - cost_old) < 10 ** (-7):
        break

    ## Checking if GD converged
    Gm , Gc = gradients(predictions,sale_price,flat_area)

    ## Updating the parameters
    m,c = param_update(m,c,Gm,Gc,alpha)

    ## Display results after 20 iterations
    if i%20 == 0:
        print('After iterations',i,':m=', m,'; c=', c,'; Cost=', cost_new)

    ## Updating the cost function
    cost_old = cost_new

## Final results
result(m,c,flat_area,sale_price,cost_new,predictions,i)





