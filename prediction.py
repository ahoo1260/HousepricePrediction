import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from source.featureEngineering import getImportantFeatures
from source.models import getENet, getKRR, getGBoost, getXGB, getLGB, getLasso
from source.stackingModels import AveragingModels

color = sns.color_palette()
sns.set_style('darkgrid')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.model_selection import KFold, cross_val_score

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


#------------------------------DEFINING CONSTANTS-----------------------------------------------------------------------
N_FOLDS= 5

#------------------------------DEFINING METHODS-------------------------------------------------------------------------
def rmsle_cv(model, X_train, y_train):
    kf = KFold(N_FOLDS, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error",cv = kf))
    return(rmse)

#-------------------------------LOADING DATA----------------------------------------------------------------------------
train = pd.read_csv('../data/train_house.csv')
test = pd.read_csv('../data/test_house.csv')

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#------------------Finding Outliers-----------------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



# -------------------------analysis on price variable ------------------------------------------------------------------
sns.distplot(train['SalePrice'] , fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#---------------make price variable more normally distributed-----------------------------------------------------------
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution
sns.distplot(train['SalePrice'] , fit=norm)
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()





#---------------------Cleaning Features and getting important features---------------------------------------------
X_train,X_test,y_train=getImportantFeatures(train,test)

#---------------------Testing Different Models--------------------------------------------------------------------------

lasso=getLasso()
score = rmsle_cv(lasso,X_train,y_train)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ENet=getENet()
score =rmsle_cv(ENet,X_train,y_train)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

krr=getKRR()
score = rmsle_cv(krr,X_train,y_train)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

gboost=getGBoost()
score = rmsle_cv(gboost,X_train,y_train)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

Xgb=getXGB()
score = rmsle_cv(Xgb,X_train,y_train)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

lgb=getLGB()
score = rmsle_cv(lgb,X_train,y_train)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


#---------------------Stacking Models-----------------------------------------------------------------------------------
averaged_models = AveragingModels(models = (ENet, gboost, krr))

score = rmsle_cv(averaged_models,X_train,y_train)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))