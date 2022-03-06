import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

trans_data=pd.read_csv('Transformed2.csv')

# iloc() is used to define the variables.

Y=trans_data.iloc[:,3]
X=trans_data.iloc[:,4:34]

# splitting the X and Y variable in a ratio of 70%-30%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

print(X_train.shape)
print(X_test.shape)
# If a fixed value is assigned like random_state = 42, then no matter how many times you execute your code the result
# would be the same i.e. same values in train and test datasets.

'''Feature scaling'''

scale = preprocessing.StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

