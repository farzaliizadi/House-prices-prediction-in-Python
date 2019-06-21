"""Created on Wed Apr  3 20:36:22 2019@author: Izadi"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
df = pd.read_csv('HousePrices.csv')
df=df.head(50000)
df.head()
df.info()
df.describe()
df.columns
X = df.drop(['Prices'], axis=1)
y = df.Prices
#Cheking Null values
s1 = set(df.isnull().sum())
s1 #null=0
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =pd.DataFrame(sc.fit_transform(X), columns=X.columns)
y_DF = pd.DataFrame(y)
y_scaled = sc.fit_transform(y_DF.values.reshape(len(y_DF),1)).ravel()
#Type of y is pd.Series and to apply  sc to it gives me error, but as a pd.DataFrame it works.
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Create the regressor: reg
model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0)
# Fit the model to the data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)
from sklearn.metrics import mean_squared_error,r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse
model.intercept_ , model.coef_
rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
rmse_train
r2_score(y_train, model.predict(X_train))
r2_score(y_test, y_pred)
# Recursive feature elimination
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
selector = RFE(model, n_features_to_select=8, step=1)
selcetor = selector.fit(X,y)
features = X.columns.values
features 
selector.support_                              
selector.ranking_ 
#9.I got  8 True and 7 False, even area is False, i will keep that, so feature_cols are:
elected_features = ['Area', 'Baths', 'Garage',  'White Marble', 'Black Marble','Indian Marble','Floors', 'City', 'Fiber','Glass Doors']
# Droping the False columns aning all over again.
X = df.drop(['FirePlace', 'Solar', 'Electric', 'Swiming Pool', 'Garden', 'Prices'], axis=1)
y = df.Prices
#********************************************************
#droping Area ndd bath 
elected_features = ['Garage', 'White Marble', 'Black Marble','Indian Marble','Floors', 'City', 'Fiber','Glass Doors']
# Droping the False columns aning all over again.
X = df.drop(['Area','Baths', 'FirePlace', 'Solar', 'Electric', 'Swiming Pool', 'Garden', 'Prices'], axis=1)
y = df.Prices
#******************************************************************************
#Again taking all features
X = df.drop(['Prices'], axis=1)
y = df.Prices

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =pd.DataFrame(sc.fit_transform(X), columns=X.columns)
y_DF = pd.DataFrame(y)
y_scaled = sc.fit_transform(y_DF.values.reshape(len(y_DF),1)).ravel()
#Type of y is pd.Series and to apply  sc to it gives me error, but as a pd.DataFrame it works.
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Create the regressor: reg
model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0)
# Fit the model to the data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)
from sklearn.metrics import mean_squared_error,r2_score
model.intercept_, model.coef_
mean_squared_error(y_train, model.predict(X_train))
mean_squared_error(y_test, y_pred)
r2_score(y_train, model.predict(X_train))
r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse
rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
rmse_train
r2_score(y_train, model.predict(X_train))
r2_score(y_test, y_pred)
# Recursive feature elimination
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
selector = RFE(model, n_features_to_select=8, step=1)
selcetor = selector.fit(X_scaled, y_scaled)
features = X.columns.values
features 
selector.support_                              
selector.ranking_ 
# Since runing r2 and adj_r2 together takes very long time because of 5000000 obsevations, I did these loops separately.

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
results = []
for i in range(1,len(X_scaled)-1):
    selector = RFE(model, n_features_to_select=i, step=1)
    selector.fit(X_scaled,y)
    r2 = selector.score(X_scaled,y)
    selected_features = features[selector.support_]
    msr = mean_squared_error(y, selector.predict(X_scaled))
    results.append([i, r2, msr, ','.join(selected_features)])

msr
r2
results
########################################################################
or i in range(1,len(X_scaled[0])-1):
#adj_r2
max_r2= 0
list_r2 =[]
for i in range (1, len(X.iloc[0])+1):
    selector=RFE(regr, i, step=1)
    selector=selector.fit(X_scaled,y_scaled)
    adj_r2= 1-((len(X)-1)/(len(X)-i-1))*(1-selector.score(X_scaled,y_scaled))
    list_r2.append(adj_r2) #mse =
    if max_r2 < adj_r2:
        sel_features = selector.support_
        max_r2 = adj_r2
max_r2         
X_sub = X_scaled.iloc[:,sel_features]


#*************************************************
#Cross validation with ShuffleSplit
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.model_selection import cross_val_score
crossValscore = cross_val_score(lr,X, y, cv=20)
crossValscore
from sklearn.model_selection import ShuffleSplit
n_samples =df.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(lr, df, df.Prices, cv=cv)
cross_val_score(lr, X, y,cv =5)
#***************************************************************
## simulate splitting a dataset of 25 observations into 5 folds
from sklearn.model_selection import KFold
kf =KFold(n_splits=5, shuffle=False).split(range(25))

for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))

#**************************************************************
#gridSearch
import os
import numpy as np
import pandas as pd
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
df = pd.read_csv('HousePrices.csv')
df=df.head(50000) 
X = df.drop(['Prices'], axis=1)
y = df.Prices
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =pd.DataFrame(sc.fit_transform(X), columns=X.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0)   
from sklearn import svm
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV  
gamma= 'auto' ; gamma = 'scale'
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVR()
gr = GridSearchCV(svr, parameters, cv=5)
gr.fit(X_train, y_train)
gr.score(X_test,y_test)

#***********************************************************
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

from sklearn.ensemble import  
forest_reg = RandomForestRegressor()
grd = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grd.fit(X_test, y_test)
grd.score(X_test, y_test)
#*************************************************************************
#Random Forest
#RandomForestClassifier
import os 
import numpy as np
import pandas as pd
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
df = pd.read_csv('HousePrices.csv')
X = df.drop(['Prices'], axis=1)
y = df.Prices
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
rf.score(X_test,y_test)
 
#*********************************************** 
#Polynomia of degree 1, 2,3, 4.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
dk = pd.read_csv('SEALEVEL.csv',index_col=0)
dk.head()
X = dk.iloc[:,1:2].values
y = dk.iloc[:,2].values
from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X,y)
y_pred = linear_reg1.predict(X)
from sklearn.preprocessing import PolynomialFeatures
poly2_reg =  PolynomialFeatures(degree=2)
X_poly2 = poly2_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly2, y)
y_pred2 = linear_reg2.predict(X_poly2)

poly3_reg=  PolynomialFeatures(degree=3)
X_poly3= poly3_reg.fit_transform(X)
linear_reg3 = LinearRegression()
linear_reg3.fit(X_poly3,y)
y_pred3 = linear_reg3.predict(X_poly3)

poly4_reg=  PolynomialFeatures(degree=4)
X_poly4= poly4_reg.fit_transform(X)
linear_reg4 = LinearRegression()
linear_reg4.fit(X_poly4,y)
y_pred4 = linear_reg4.predict(X_poly4)

labels=['deg1','deg2','deg3','deg4']
plt.scatter(X,y, color='red')
plt.plot(X,y_pred, color='blue')
plt.plot(X,y_pred2, color='purple')
plt.plot(X,y_pred3, c='green')
plt.plot(X,y_pred4, color='black')
plt.xlabel('Level position', size=30)
plt.ylabel('Ditance from Sea Level', size=30)
plt.title("Polynomial Plot", size=50)
plt.legend(labels, loc=2)
plt.show()
#********************************************************************
#SVM SVR
# the impact on the results
import os
import pandas as pd
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
df = pd.read_csv('House_prices.csv',index_col=[0,1])
X = df.iloc[:, 0:15]
y = df.iloc[:, 15]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0)
from sklearn import svm
from sklearn.svm import SVR
clr = svm.SVR(kernel='linear', C=0.01)
y_pred = clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)
clr.score(X_test,y_test)
svr_rbf = svm.SVR(kernel ='rbf', C=10**3, gamma=0.1)
y_pred = svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)
svr_rbf.score(X_test,y_test)
svr_lin = SVR(kernel ='linear', C=10)
y_pred = svr_lin.fit(X_train, y_train)
y_pred = svr_lin.predict(X_test)
svr_lin.score(X_test,y_test)
svr_poly2 = SVR(kernel ='poly', C=10, degree=2, gamma=1)
y_pred = svr_poly2.fit(X_train, y_train)
y_pred = svr_poly2.predict(X_test)
svr_poly2.score(X_test,y_test)
svr_poly3 = SVR(kernel ='poly', C=10**3, degree=3, gamma='auto')
y_pred = svr_poly3.fit(X_train, y_train)
y_pred = svr_poly3.predict(X_test)
svr_poly3.score(X_test,y_test)
svr_sig = SVR(kernel='sigmoid', C=0.02)
y_pred = svr_sig .fit(X_train, y_train)
y_pred = svr_sig.predict(X_test)
svr_sig.score(X_test,y_test)
#***********************************************
#plots and corr
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pylab import rcParams
os.chdir(r'D:\desktop\data mining\ML\LinearReg_ML')
df = pd.read_csv('HousePrices.csv')
df=df.head(50000)
df.columns
rcParams['figure.figsize'] = 15, 10
sns.countplot(df['Prices'])
plt.scatter(df.Area,df.Prices)
plt.scatter(df.Baths,df.Prices)

df.corr(method='pearson') 
df.corr(method='spearman')
df.corr(method='kendall')

# from def to plt.show excute alltogether then correlation_matrix(ddef correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(df.corr('kendall'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.title('Kendall heat map', size=30)
    plt.show()
    

