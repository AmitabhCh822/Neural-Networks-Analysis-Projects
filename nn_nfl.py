####Amitabh Chakravorty
####Dr. Treu
####Neural Net NFL
####11/17/2021

# mlp for binary classification
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# pd.read_fwf() reads the .dat file by dividing the data into columns
df1 = pd.read_fwf('nfl_combine_2014.dat.txt', index_col=0)

#Converting the dat file into csv file
df1.to_csv('nfl.csv')
df = pd.read_csv('nfl.csv')

#Deleting the first three columns which are not be used in the analysis
df.drop(df.iloc[:,[0,1,2]], axis=1, inplace=True)

#Labeling the columns (Code given by Dr. Treu)
df.columns = ['Grade', 'Height', 'Length', 'Weight', '40Yard', 'BenchPress',
              'VerticalJump', 'BroadJump', '3Cone', '20Yard', 'Extra']

#Eliminating rows with missing values (Code retrieved from Stack Overflow)
df = df.dropna()  

#Converting the Grade attribute to nominal from numeric
df['Grade'] = np.where(df['Grade'] > df['Grade'].median(),'good','bad') #Code retrieved from Stack Overflo)

X = df.iloc[:,[1,2,3,4,5,6,7,8,9]]  #All the attributes other than 'Grade' are the predictors
y = df.iloc[:,0]  #'Grade' is the output attribute

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid')) # sigmoid for binary classification
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)