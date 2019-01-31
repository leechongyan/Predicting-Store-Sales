import pandas as pd
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import math
import matplotlib.pyplot as plt

#   Data Import
trainData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/train.csv',dtype={"StateHoliday":object})
testData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/test.csv',dtype={"StateHoliday":object})
storeData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/store.csv')
testData.drop("Id", axis=1, inplace=True)

#   Feature Engineer the trainData and testData together
combineData = pd.concat([trainData,testData],keys=['x','y'],sort=False)
print(combineData.dtypes)
print(combineData.StateHoliday.unique())
print(storeData.dtypes)

def cleanData(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = pd.get_dummies(data, columns=["StateHoliday"], prefix=["stateholiday"])
    return data

#   Feature Engineering
def createFeatures(data):

    ### Date Counter for Promotion ###
    data = data.sort_values(['Store', 'Date'])
    data['PromoSince'] = 0
    data['NoPromoSince'] = 0

    #   Creating a date counter to evaluate how many days the promo is up
    for i in range(0,len(data)):
        storeName = data.iloc[i,8]
        if data.iloc[i,4]==0:
            data.iloc[i, 9]=0
        else:
            count=0
            for k in range(i,-1,-1):
                if data.iloc[k, 4]==1 and data.iloc[k,8]==storeName:
                    count = count+1
                else:
                    break
            data.iloc[i, 9]=count

    #   Creating a date counter to evaluate how many days the promo is down
    for i in range(0,len(data)):
        storeName=data.iloc[i,8]
        if data.iloc[i,4]==1:
            data.iloc[i, 10]=0
        else:
            count=0
            for k in range(i,-1,-1):
                if  data.iloc[k,4]==0 and data.iloc[k,8]==storeName:
                    count=count+1
                else:
                    break
            data.iloc[i, 10] = count

    #   Creating isWeekend, tmrWeekend, yestWeekend
    data['isWeekend']=0
    data['tmrWeekend'] = 0
    data['yestWeekend'] = 0
    data.loc[(data['DayOfWeek']==6) | (data['DayOfWeek']==7), ['isWeekend']]=1
    data.loc[data['DayOfWeek']==5,['tmrWeekend']]=1
    data.loc[data['DayOfWeek']==1,['yestWeekend']]=1

    #   Creating yestClosed, tmrClosed
    data['yestClosed']=0
    data['tmrClosed']=0
    for i in range(1, len(data)):
        storeName = data.iloc[i, 8]
        if data.iloc[i-1,3]==0 and data.iloc[i-1,8]==storeName:
            data.iloc[i,14]=1
        if i!=(len(data)-1): #Prevent out of index error
            if data.iloc[i+1,3]==0 and data.iloc[i+1,8]==storeName:
                data.iloc[i,15]=1

    return data


#   Initial Data Type transformation
combineData['Date'] = pd.to_datetime(combineData['Date'])
combineData = pd.get_dummies(combineData, columns=["StateHoliday"], prefix=["stateholiday"])


#   Filter to get Store of storetype a b c d
combineData=combineData.loc[(combineData['Store']==2)|(combineData['Store']==3)|(combineData['Store']==5)|(combineData['Store']==6)|(combineData['Store']==85)|(combineData['Store']==259)|(combineData['Store']==262)|(combineData['Store']==274)|(combineData['Store']==1)|(combineData['Store']==4)|(combineData['Store']==21)|(combineData['Store']==25)|(combineData['Store']==13)|(combineData['Store']==15)|(combineData['Store']==18)|(combineData['Store']==20)]

#   Feature Engineering
combineData=createFeatures(combineData)

#   Changing Date Field
combineData['year']=combineData['Date'].dt.year
combineData['month']=combineData['Date'].dt.month
combineData['day']=combineData['Date'].dt.day

#   Splitting Data into test and train
combineData = combineData.sort_values(['Date'])
trainData=combineData.loc['x']
testData=combineData.loc['y']
print(len(trainData))
trainSet=trainData.iloc[0:11759]
testSet=trainData.iloc[11760:14703]

trainSet.drop(columns="Date",inplace=True)
testSet2=testSet.drop(columns="Date")
testData.drop(columns="Date",inplace=True)

#   storeData label encoding
storeData=pd.get_dummies(storeData,columns=["StoreType","Assortment","PromoInterval"],prefix=["storetype","assortment","promointerval"])


trainSet=trainSet.loc[trainSet["Open"]==1]
testSet2=testSet2.loc[testSet2["Open"]==1]
testSet=testSet.loc[testSet["Open"]==1]
testData=testData.loc[testData["Open"]==1] #    Will just leave the testData set here for the time being

trainSet=pd.merge(trainSet, storeData, on='Store', how='inner')
testSet2=pd.merge(testSet2, storeData, on='Store', how='inner')

#   Working on trainData
train_x=trainSet.drop(['Customers','Sales'],axis=1)
train_y=trainSet[['Sales']].copy()
test_x=testSet2.drop(['Customers','Sales'],axis=1)
test_y=testSet2[['Sales']].copy()

#   Initialise Parameters to be tested
params_dist_grid = {
    'max_depth': [1, 2, 3, 4],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001), # uniform discrete random distribution
    'learning_rate': uniform(), # gaussian distribution
    'subsample': uniform(), # gaussian distribution
    'colsample_bytree': uniform() # gaussian distribution
}

#   Setting the fixed parameter
params_fixed = {
    'objective': 'reg:linear',
    'silent': 1
}

#   Setting Seed
seed=342

#   Setting the cross validation folds
cv = StratifiedKFold(shuffle=True, random_state=seed)
cv.split(train_x,train_y)

#   Instantiating the RandomisedSearchCV
rs_grid = RandomizedSearchCV(
    estimator=XGBRegressor(**params_fixed, seed=seed),
    param_distributions=params_dist_grid,
    n_iter=20,
    cv=cv,
    scoring='neg_mean_squared_error',
    random_state=seed
)

#   HyperParameter Tuning
rs_grid.fit(train_x, train_y)

#   Extracting the best parameter
best_param = rs_grid.best_params_

#   Training the final model
clf = XGBRegressor(**best_param)
clf.fit(train_x,train_y)

#   Identifying the most important predictors
plot_importance(clf)
plt.savefig('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/image/feature_importance.png')
plt.close()

#   Predicting on the testSet
predictions = clf.predict(test_x)

#   Calculating the RMSE
rms = math.sqrt(mean_squared_error(test_y.values, predictions))
print(rms)

#   Print the discrepancies
combinedResult=pd.DataFrame(data={'Store':testSet['Store'].values.ravel(), 'Date':testSet['Date'].values.ravel(),'predictions': predictions, 'actual': test_y.values.ravel()})
print(combinedResult)

#   Show the predictions for store 1
combinedResult1=combinedResult.loc[combinedResult['Store']==1]

# gca stands for 'get current axis'
ax = plt.gca()
#   Saving the predicted graph for store 1
combinedResult1.plot(kind='line',x='Date',y='actual',ax=ax)
combinedResult1.plot(kind='line',x='Date',y='predictions', color='red', ax=ax)
plt.savefig('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/image/store1.png')
plt.show()
plt.close()

#   Saving the model
clf.save_model('C:/Users/leech/OneDrive/Desktop/Analytics RossMann/rossmann-store-sales/model/finalmodel.model')


