# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:49:12 2017

@author: franc
"""
import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as pl

#start cronometro
t0 = time.time()

df_crude= pd.read_csv('Crude Oil.csv', index_col=[0], parse_dates=True)
df_sp=pd.read_csv('SP500.csv', index_col=[0],parse_dates=True)
df_eeri=pd.read_csv('EERI EUR.csv',index_col=[0],parse_dates=True)
df_eur=pd.read_csv('price_eur.csv',index_col=[0],parse_dates=True,dayfirst=True)
df_eurusd=pd.read_csv('price_eurusd.csv',index_col=[0],parse_dates=True,dayfirst=True)
df_comodity=pd.read_csv('price_comodity.csv',index_col=[0],parse_dates=True,dayfirst=True)
df_index=pd.read_csv('price_index.csv',index_col=[0],parse_dates=True,dayfirst=True)
df_sptr=pd.read_csv('price_sptr.csv',index_col=[0],parse_dates=True,dayfirst=True)

df_crude.columns = df_crude.columns.map(lambda x: str(x) + '_crude')
df_sp.columns = df_sp.columns.map(lambda x: str(x) + '_sp')
df_eeri.columns = df_eeri.columns.map(lambda x: str(x) + '_eeri')


#merge 12 coeff
df_global=pd.concat([df_crude,df_sp,df_eeri], axis=1).drop(['Crude Oil_crude','SP500_sp','EERI EUR_eeri'],axis=1)
df_sptr=df_sptr.join(df_global,how='inner').iloc[::-1]
df_eur=df_eur.join(df_global,how='inner').iloc[::-1]
df_eurusd=df_eurusd.join(df_global,how='inner').iloc[::-1]
df_comodity=df_comodity.join(df_global,how='inner').iloc[::-1]
df_index=df_index.join(df_global,how='inner').iloc[::-1]

X=df_sptr
y=df_sptr['PX_LAST']

up_threshold=0.02
down_threshold=-0.02
portfolio1=np.array([0])
portfolio2=np.array([]) #To be define before the cycle
portfolio3=np.array([0]) #this is just ret on start and end
invs1=0
invs2=0
invs3=0
positive_count1=0
number_of_trades1=0
positive_count2=0
number_of_trades2=0
positive_count3=0
number_of_trades3=0
y_total=pd.DataFrame()
y_total2=pd.DataFrame()
y_total3=pd.DataFrame()
gain=np.array([1]) 
loss=np.array([1]) 
lag=int(10) #lag
k=int(500) #initial set measure
     
for index in range(int((len(X)-k)/lag)-2):  #scorriamo ogni lag

#    X_train=X.head(k+index*lag) #[len(X)-k-2*i:len(X)-2*i]
#    y_train=y[lag:k+lag*(1+index)] #[len(y)-k-i:len(y)-i]
#    X_test=X[len(X_train):len(X_train)+lag]
#    y_test=y[len(y_train)+lag:len(y_train)+2*lag]
    print(index)
    X_train=X[index*lag:k+index*lag]
    y_train=y[lag*(1+index):k+lag*(1+index)]
    X_test=X[k+index*lag+1:k+index*lag+1+lag]
    y_test=y[k+lag*(1+index)+1:k+lag*(1+index)+1+lag]
    
    # online feature selection
    df_corrMatrix=X[index*lag:k+index*lag].corr()
    np.fill_diagonal(df_corrMatrix.values, np.NaN)
    df_corrMatrix[df_corrMatrix < 0] = np.NaN
    df_corrMatrix=df_corrMatrix.drop(['PX_LAST'])
#    df_corrMatrix['avg'] = df_corrMatrix.mean(axis=1)
#    df_corrMatrix.sort_values('avg',ascending=True,inplace=False)
    df_corrMatrix.sort_values('PX_LAST',ascending=False,inplace=True)
    feature=list(df_corrMatrix.head(n=6).index.values)
    X_train=X_train[feature]
    X_test=X_test[feature]

# ONLY FOR: MLPREG
    scaler = StandardScaler()  
## Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train3 = scaler.transform(X_train)  
## apply same transformation to test data
    X_test3 = scaler.transform(X_test) 
# ONLY FOR: MLPREG

#    linreg=linear_model.Ridge()
    linreg3 = MLPRegressor(solver='lbfgs', random_state=1)

#    #GRID SEARCH
    parameters={'alpha':[10**-20,1000]}
    clf=GridSearchCV(linreg3, parameters)
    clf.fit(X_train3,y_train)

#   PREDICT
    linreg= linear_model.LinearRegression()
    linreg2=linear_model.ElasticNetCV()

#    linreg=linear_model.LarsCV()
    linreg.fit(X_train,y_train)
    linreg2.fit(X_train,y_train)
    y_pred = linreg.predict(X_test)
    y_pred2 = linreg2.predict(X_test)
    y_pred3=clf.predict(X_test3)
    y_pred=pd.Series(y_pred, index=y_test.index)
    y_pred2=pd.Series(y_pred2, index=y_test.index)
    y_pred3=pd.Series(y_pred3, index=y_test.index)
    y_total=pd.concat([y_total,y_pred])
    y_total2=pd.concat([y_total2,y_pred2])
    y_total3=pd.concat([y_total3,y_pred3])

 #STRATEGIA 1
    start=y_test[len(y_test)-1] #punto partenza calcolo portfolio
    avg=y_pred.mean()
    ret=(avg-start)/start #mean return
    if (ret>up_threshold):
        portfolio1=np.append(portfolio1,y_test[(lag-1)]-y_train[len(y_train)-1])
        invs1=invs1+y_train[len(y_train)-1]
        number_of_trades1=number_of_trades1+1
        if (portfolio1[len(portfolio1)-1]>0):
            positive_count1=positive_count1+1;
    elif (ret<down_threshold):
        portfolio1=np.append(portfolio1,y_train[len(y_train)-1]-y_test[(lag-1)])
        invs1=invs1+y_train[len(y_train)-1] 
        number_of_trades1=number_of_trades1+1
        if (portfolio1[len(portfolio1)-1]>0):
            positive_count1=positive_count1+1;
 #STRATEGIA 2
    start=y_test[len(y_test)-1] #punto partenza calcolo portfolio
    avg=y_pred.mean()
    ret=(avg-start)/start #mean return
    if (ret>up_threshold):
        strat=y_test[(lag-1)]-y_train[len(y_train)-1]
        if ((sum(gain)/len(gain))>0.5):
            portfolio2=np.append(portfolio2,strat)
            invs2=invs2+y_train[len(y_train)-1]
        else :
            portfolio2=np.append(portfolio2,-strat)
            invs2=invs2+y_train[len(y_train)-1]          
        if strat>0:
            gain=np.append(gain,1)
        else:
            gain=np.append(gain,0)
        number_of_trades2=number_of_trades2+1
        if (portfolio2[len(portfolio2)-1]>0):
            positive_count2=positive_count2+1;   
    elif (ret<down_threshold):
        strat=y_train[len(y_train)-1]-y_test[(lag-1)]
        if ((sum(loss)/len(loss))>0.5):
            portfolio2=np.append(portfolio2,strat)
            invs2=invs2+y_train[len(y_train)-1] 
        else:
            portfolio2=np.append(portfolio2,-strat)
            invs2=invs2+y_train[len(y_train)-1]
        if (strat>0):
            loss=np.append(loss,1)
        else:
            loss=np.append(loss,0)
        number_of_trades2=number_of_trades2+1
        if (portfolio2[len(portfolio2)-1]>0):
            positive_count2=positive_count2+1;   
#STRATEGIA 3
    ret3=(float(y_pred.tail(1))-start)/start
    if (ret3>up_threshold):
       portfolio3=np.append(portfolio3,y_test[(lag-1)]-y_train[len(y_train)-1])
       invs3=invs3+y_train[len(y_train)-1]
       number_of_trades3=number_of_trades3+1
       if (portfolio3[len(portfolio3)-1]>0):
           positive_count3=positive_count3+1;  
    elif (ret3<down_threshold):
       portfolio3=np.append(portfolio3,y_train[len(y_train)-1]-y_test[(lag-1)])
       invs3=invs3+y_train[len(y_train)-1] 
       number_of_trades3=number_of_trades3+1
       if (portfolio3[len(portfolio3)-1]>0):
           positive_count3=positive_count3+1;       

y_total4=y_total.join(y,how='inner')

pl.plot(y_total4, color='green',label="Real Data")
pl.plot(y_total, color='yellow',label="Linear Regression")
pl.plot(y_total2, color='red',label="ElasticNetCV")
pl.plot(y_total3, color='blue',label="MLP")
pl.show()

print('Parameters')
print('Window')
print(k)
print('Lag')
print(lag)
print('Portfolio results')
print('Portfolio 1')
#print(sum(portfolio1))
#print('Money invested')
#print(invs1)
print('Number of Good Trades/overall')
print(positive_count1/number_of_trades1)
print('Portfolio 2')
#print(sum(portfolio2))
#print('Money invested')
#print(invs2)
print('Number of Good Trades/overall')
print(positive_count2/number_of_trades2)
print('Portfolio 3')
#print(sum(portfolio3))
#print('Money invested')
#print(invs3)
print('Number of Good Trades/overall')
print(positive_count3/number_of_trades3)
#end cronometro
t1 = time.time()
print ("-" * 100)
print ("Completed in %.2f minutes" % ((t1-t0)/60))
print ("\a")