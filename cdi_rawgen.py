#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:03:57 2018

@author: divyansh
"""
import numpy as np
import pandas as pd
import random
from matplotlib import pylab as plt
import sklearn.preprocessing as pps
import seaborn as sns

## This is a random approach to generate data

def ratings_generator(nd):
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    cus = []

   # brand_no = []
    #for b in range(0,10):
    for c in nd:
            cus.append(c)
            r1.append(random.randint(-5,5))
            r2.append(random.randint(-5,5))
            r3.append(random.randint(-5,5))
            r4.append(random.randint(-5,5))
            r5.append(random.randint(-5,5))
        #brand_no.append(np.repeat(b+1,repeats=len(nd)))       
    data = pd.DataFrame({'customer_no':cus, 'rating1':r1, 'rating2':r2, 'rating3':r3, 'rating4':r4, 'rating5':r5})
    # Scaling Ratings from 0 to 1
    rat = ['rating1','rating2','rating3','rating4','rating5']
    for col in rat:
        data[col] = (data[col]-min(data[col]))/(max(data[col])-min(data[col]))
    return(data)
    
""" rescaling ratings 
for i in range(1,5001):
    for col in ['rating1','rating2','rating3','rating4','rating5']:    
        data[data['customer_no']==i][col] = (data[data['customer_no']==i][col]-np.percentile(data[data['customer_no']==i][col],10))/(np.percentile(data[data['customer_no']==i][col],90) - np.percentile(data[data['customer_no']==i][col],10))
"""
           

def avg_rating(data):   # Avg rating per customer per brand for feedback (5 questions)
    rat = ['rating1','rating2','rating3','rating4','rating5']
    avg_rating1 = data[rat].apply(np.mean,axis=1)
    data['avg_rating'] = avg_rating1 
    return(data)

def brand_ratings(): # Generates ratings for a brand
            d={}
            e={}
            f={}
            g={}
            h={}
            fd = pd.DataFrame()
            fe = pd.DataFrame()
            ff = pd.DataFrame()
            fg = pd.DataFrame()
            fh =pd.DataFrame()
            for j in range(1,6): # Feedbacks from 1 to 5
                print("\nFeedback %i:\n"%j)
                nd = np.array(range(1,5001))
                for i in range(0,4): # Iterations for each feedback
                        r = ratings_generator(nd)
                        if(j==1):
                            d[i] = avg_rating(r)
                            if(i==0):
                                fd = d[i]
                            else:
                                fd = pd.merge(fd, d[i],how='left',on='customer_no')
                        elif(j==2):
                            e[i] = avg_rating(r)
                            if(i==0):
                                fe = e[i]
                            else:
                                fe = pd.merge(fe, e[i],how='left',on='customer_no')  
                        elif(j==3):
                            f[i] = avg_rating(r)
                            if(i==0):
                                ff = f[i]
                            else:
                                ff = pd.merge(ff, f[i], how='left',on='customer_no')
                        elif(j==4):
                            g[i] = avg_rating(r)
                            if(i==0):
                                fg = g[i]
                            else:
                                fg = pd.merge(fg, g[i],how='left',on='customer_no')
                        else:
                            h[i] = avg_rating(r)
                            if(i==0):
                                fh = h[i]
                            else:
                                fh = pd.merge(fh, h[i],how='left',on='customer_no')
                        print("Iteration %i:\n"%(i+1),avg_rating(r).head())
                        nd = np.array(avg_rating(r)[(avg_rating(r)['avg_rating']<0.5) | (pd.isna(avg_rating(r)['avg_rating']))]['customer_no'])
                        #nd = np.array(avg_rating(r)[(avg_rating(r)['avg_rating']<0.5)]['customer_no'])
                        if(len(nd)==0):
                            break
            return(fd,fe,ff,fg,fh)

from functools import reduce

def brand_ratings_select(i):  # Outputs all feedback ratings for brand number(1-10) selected
    f = brand_ratings()
    f[0]['Brand_no'] = np.repeat(i,repeats=5000)
    f[1]['Brand_no'] = np.repeat(i,repeats=5000)
    f[2]['Brand_no'] = np.repeat(i,repeats=5000)
    f[3]['Brand_no'] = np.repeat(i,repeats=5000)
    f[4]['Brand_no'] = np.repeat(i,repeats=5000)
    print("\nRatings, Feedback 1:\n",f[0])
    print("\nRatings, Feedback 2:\n",f[1])
    print("\nRatings, Feedback 3:\n",f[2])
    print("\nRatings, Feedback 4:\n",f[3])
    print("\nRatings, Feedback 5:\n",f[4])
    dfs = [f[0],f[1],f[2],f[3],f[4]]
    return(reduce(lambda left,right: pd.merge(left,right,on=['customer_no','Brand_no']), dfs))  
    
# Combining 10 brand ratings..
      
final = pd.DataFrame()
for i in range(1,11):
    final = pd.concat([final,brand_ratings_select(i)],axis=0,ignore_index=True)    
    
final.to_csv('ratings.csv', index=False)

# Adding demographic data and NPS
final['NPS'] = [random.randint(0,10) for i in range(0,50000)]
final['Gender'] = [random.randint(0,1) for i in range(0,50000)]
recode = {0:'Male',1:'Female'}
final['Gender'] = final['Gender'].map(recode)
final['Response_rate'] = [random.randint(20,100) for i in range(0,50000)]
final['Education'] = [random.randint(1,3) for i in range(0,50000)]
recode = {1:'High School', 2:'College', 3:'Doctoral'}
final['Education'] = final['Education'].map(recode)
final['Age'] = [random.randint(1,7) for i in range(0,50000)]
recode = {1:'Under 13', 2:'13-17', 3:'18-25',4:'26-34',5:'35-54',6:'55-64',7:'Above 64'}
final['Age'] = final['Age'].map(recode)
final['Marital'] = [random.randint(1,3) for i in range(0,50000)]
recode = {1:'Single', 2:'Married', 3:'Divorced'}
final['Marital'] = final['Marital'].map(recode)

final.to_csv('ratings2.csv', index=False)
                                      

#######################################################################
##NPS modelling
#######################################################################

## This is a linear regression focused approach to generate data

import numpy as np
import pandas as pd
import random
from matplotlib import pylab as plt
import sklearn.preprocessing as pps
import seaborn as sns
from sklearn.datasets import make_regression
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from functools import reduce


def regression():
# Generate a random regression problem
    X, y = make_regression(n_samples=5000, n_features=10, n_informative=6, n_targets=1, random_state=100, noise=0.05)
        
    # NPS
    scaler = pps.MinMaxScaler(feature_range=(0, 10), copy=True)
    scaler.fit(y.reshape(-1,1))
    y = scaler.transform(y.reshape(-1,1))
    y = np.round(y,0)
    
    odata = X
    
    X = pd.DataFrame(X, columns=['Gender','Response_Rate(%)','Education','Age','Marital_Status','Feedback_OR1','Feedback_OR2','Feedback_OR3','Feedback_OR4','Feedback_OR5'], index=range(0,5000))
    
    # Gender
    scaler = pps.MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(X['Gender'].reshape(-1,1))
    X['Gender'] = scaler.transform(X['Gender'].reshape(-1,1))  
    X['Gender'] = np.round(X['Gender'],0) 
    recode = {0:'Male',1:'Female'}
    X['Gender'] = X['Gender'].map(recode) 
    
    # Response Rate
    scaler = pps.MinMaxScaler(feature_range=(20, 100), copy=True)
    scaler.fit(X['Response_Rate(%)'].reshape(-1,1))
    X['Response_Rate(%)'] = scaler.transform(X['Response_Rate(%)'].reshape(-1,1))
    X['Response_Rate(%)'] = np.round(X['Response_Rate(%)'],2)
    
    
    # Education
    scaler = pps.MinMaxScaler(feature_range=(1, 3), copy=True)
    scaler.fit(X['Education'].reshape(-1,1))
    X['Education'] = scaler.transform(X['Education'].reshape(-1,1))
    X['Education'] = np.round(X['Education'],0)
    recode = {1:'High School', 2:'College', 3:'Doctoral'}
    X['Education'] = X['Education'].map(recode)
    
    # Age
    scaler = pps.MinMaxScaler(feature_range=(1,7), copy=True)
    scaler.fit(X['Age'].reshape(-1,1))
    X['Age'] = scaler.transform(X['Age'].reshape(-1,1))
    X['Age'] = np.round(X['Age'],0)
    recode = {1:'Under 13', 2:'13-17', 3:'18-25',4:'26-34',5:'35-54',6:'55-64',7:'Above 64'}
    X['Age'] = X['Age'].map(recode)
    
    # Marital
    scaler = pps.MinMaxScaler(feature_range=(1, 3), copy=True)
    scaler.fit(X['Marital_Status'].reshape(-1,1))
    X['Marital_Status'] = scaler.transform(X['Marital_Status'].reshape(-1,1))
    X['Marital_Status'] = np.round(X['Marital_Status'],0)
    recode = {1:'Single', 2:'Married', 3:'Divorced'}
    X['Marital_Status'] = X['Marital_Status'].map(recode)
    
    # Average Ratings (per feedback)
    for i in ['Feedback_OR1','Feedback_OR2','Feedback_OR3','Feedback_OR4','Feedback_OR5']:    
        scaler = pps.MinMaxScaler(feature_range=(0, 1), copy=True)
        scaler.fit(X[i].reshape(-1,1))
        X[i] = scaler.transform(X[i].reshape(-1,1))
        X[i] = np.round(X[i],5)
    for col in ['Gender','Age','Education','Marital_Status']:
        X[col] = X[col].astype('category')       
    return(X,y)    
    

d = regression()
NPS = d[1].copy()
# Adding randomness to NPS

"""idx = random.sample(range(0,len(d[1])),round(len(d[1])*.2))
a = np.array([random.randint(0,10) for i in idx])
d[1][idx] = a.reshape(round(len(d[1])*.2),1)
idx2 = random.sample(range(0,len(d[1])),round(len(d[1])*.3))
a2 = np.array([random.randint(8,10) for i in idx2])
d[1][idx2] = a2.reshape(round(len(d[1])*.3),1)
"""

## Data Visualization and Analysis-----------------------------------------------

# Frequency distributions
print("\nCounts for gender:")
print(d[0]['Gender'].value_counts(sort=False,normalize=False))

print("\nPercentages for gender:")
print(d[0]['Gender'].value_counts(sort=False,normalize=True)*100)

print("\nCounts for Age Category:")
print(d[0]['Age'].value_counts(sort=False,normalize=False))

print("\nPercentages for Age Category:")
print(d[0]['Age'].value_counts(sort=False,normalize=True)*100)

print("\nCounts for Education:")
print(d[0]['Education'].value_counts(sort=False,normalize=False))

print("\nPercentages for Education:")
print(d[0]['Education'].value_counts(sort=False,normalize=True)*100)

print("\nCounts for Marital Status:")
print(d[0]['Marital_Status'].value_counts(sort=False,normalize=False))

print("\nPercentages for Marital Status:")
print(d[0]['Marital_Status'].value_counts(sort=False,normalize=True)*100)

# Countplots, barplots and boxplots
for cat in ['Gender','Education','Marital_Status','Age']:
    sns.countplot(cat, data=d[0]) # Frequency distribution
    plt.xlabel('%s'%cat)
    plt.ylabel('Count')
    plt.title('%s distribution among feedback givers'%cat)
    plt.savefig('counts_%s.png'%cat)
    plt.show()
    sns.barplot(x=d[0][cat].reshape(5000,),y=d[1].reshape(5000,), estimator=np.mean, capsize=.2) # mean NPS for each category
    plt.title("%s - barplot with NPS Score"%cat) 
    plt.savefig('bars_%s.png'%cat)
    plt.show()
    sns.boxplot(x=d[0][cat].reshape(5000,),y=d[1].reshape(5000,)) # gives outliers and general skewness 
    plt.title("%s - boxplot for NPS Score"%cat)
    plt.savefig('boxes_%s.png'%cat)
    plt.show()
    
import warnings
warnings.filterwarnings("ignore")

# Distribution of dependent variable - NPS (Should be normally distributed)
import scipy.stats as s
sns.distplot(d[1],kde=True)
plt.xlabel('NPS Distribution')
plt.savefig('NPS_Dist.png')
plt.show() # log, square root, cube root Transformations can be performed if non-normal

fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = s.probplot(d[1].reshape(5000,), dist=s.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2 = fig.add_subplot(212) #boxcox to transform the data so it’s closest to normal
xt, _ = s.boxcox(d[1][d[1]>0].reshape(len(d[1][d[1]>0]),))
prob = s.probplot(xt, dist=s.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation') # QQ plot to assess the transformation - should be almost aligned with 45 degree straight line
plt.savefig('qqplot_NPS.png')
plt.show()
print("Skewness before transformation: ",s.skew(d[1])) # Skewness of distribution - should be close to 0
print("Kurtosis before transformation: ",s.kurtosis(d[1]))
d[1][d[1]>0] = xt
print("Skewness after transformation: ",s.skew(d[1])) # Skewness of distribution - should be close to 0
print("Kurtosis after transformation: ",s.kurtosis(d[1]))

# Plotting distributions and scatterplots of quantitative variables
for var in ['Response_Rate(%)','Feedback_OR1','Feedback_OR2','Feedback_OR3','Feedback_OR4','Feedback_OR5']:
    sns.distplot(d[0][var])
    plt.title("%s - Distribution"%var)
    plt.savefig("Dist_%s"%var)
    plt.show()
    sns.regplot(x=d[0][var],y=d[1].reshape(5000,), scatter=True)
    plt.xlabel("%s"%var)
    plt.title('Relationship between %s and NPS score'%var)
    plt.savefig('Reg_%s'%var)
    plt.show()

## Linear Regression------------------------------------------------
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
lm = linear_model.LinearRegression()

#Data Management

# integer encode
label_encoder = LabelEncoder()
d[0]['Gender'] = label_encoder.fit_transform(d[0]['Gender'])
print(d[0]['Gender'])

d[0]['Age'] = label_encoder.fit_transform(d[0]['Age'])
print(d[0]['Age'])

d[0]['Education'] = label_encoder.fit_transform(d[0]['Education'])
print(d[0]['Education'])

# binary encode
"""
onehot_encoder = OneHotEncoder(sparse=False)
d[0]['Marital_Status'] = label_encoder.fit_transform(d[0]['Marital_Status'])
encoded = onehot_encoder.fit_transform(d[0]['Marital_Status'].reshape(len(d[0]['Marital_Status']),1))
print(encoded)
"""

df = pd.get_dummies(d[0],columns=['Marital_Status'], drop_first=True) # Creating dummy variables for non ordinal categorical variable
dff = df.join(pd.DataFrame(d[1]))

# Data Management ends

# Train test split    
X_train, X_test, y_train, y_test = train_test_split(df, d[1], test_size=0.2, random_state=1111)

# Fitting model    
model = lm.fit(X_train,y_train)
preds = lm.predict(X_test)

plt.scatter(y_test, preds)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

print( 'Score: ', model.score(X_test, y_test))
print('\nMean Squared Error: ', mean_squared_error(y_test,preds))

print("\n Coefficients: ",dict(zip(df.columns, model.coef_.reshape(11,)))) # Coefficient of each feature

# Cross Validation
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

#k fold CV
for k in range(2,11):   
        scores = cross_val_score(model, df, d[1], cv=k)
        print('\nCross-validated scores:', scores)
        print('\nMean score:', np.mean(scores))
        
# Make cross validated predictions
predictions = cross_val_predict(model, df, d[1], cv=6) # k=6  
plt.scatter(d[1], predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted')
plt.title('Cross-Predicted values vs true values')
plt.savefig('cv_pred_true(lm).png')
plt.show()

accuracy = metrics.r2_score(d[1], predictions)
print ('Cross-Predicted Accuracy:', accuracy) # R squared

## Lasso Regression (for feature selection and regularization)-----------------------------------------
from sklearn.linear_model import LassoLarsCV
model=LassoLarsCV(cv=10, precompute=False).fit(X_train,y_train)

# print variable names and regression coefficients
print("Lasso regression coefficients: ",dict(zip(df.columns, model.coef_)))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.savefig('coef_lasso.png')
plt.show()

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.savefig('mse_lasso.png')
plt.show()
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(X_train,y_train)
rsquared_test=model.score(X_test,y_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)



## Random Forest--------------------------------------------
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sklearn

y_train1 = (y_train.astype('int'))
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(X_train,y_train1)

predictions=classifier.predict(X_test)
y_test1 = y_test.astype('int')
print("confusion matrix: ",sklearn.metrics.confusion_matrix(y_test1, predictions))
print("Accuracy = ",sklearn.metrics.accuracy_score(y_test1, predictions))

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train,y_train1)

# display the relative importance of each attribute
print("Feature importances: ",dict(zip(df.columns,model.feature_importances_)))

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(X_train,y_train1)
   predictions=classifier.predict(X_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(y_test1, predictions)
   
plt.cla()
plt.plot(trees, accuracy)
plt.title("Number of trees vs accuracy - classification")
plt.savefig('rf_acc.png')
plt.show()

## Random forest - regression
y_train = (y_train.astype('long'))
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)
y_test = y_test.astype('long')
print("Accuracy = ",sklearn.metrics.accuracy_score(y_test, predictions))

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train,y_train)
# display the relative importance of each attribute
print("Feature importances: ",dict(zip(df.columns,model.feature_importances_)))

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(X_train,y_train)
   predictions=classifier.predict(X_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(y_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)
plt.title("Number of trees vs accuracy - regression")
plt.savefig('rf_acc2.png')
plt.show()
    
## Calculating Brand-wise NPS score

# 0 to 6 = Detracters
# 7 to 8 = Neutrals
# 9 to 10 = Promoters

proms = len(NPS[NPS>=9])/len(NPS)
dets = len(NPS[NPS<=6])/len(NPS)

NPS_Score = (proms-dets)*100
print("NPS Score for Brand: ",NPS_Score)

######################################################################
## CDI, BDI and NPS Calculation
######################################################################

## Calculating Brand-wise NPS score

# 0 to 6 = Detracters
# 7 to 8 = Neutrals
# 9 to 10 = Promoters

NPS_Score = [None for j in range(1,11)] 
for i in range(1,11):
    proms = sum(final[final['Brand_no']==i]['NPS']>=9)/len(final[final['Brand_no']==i])
    dets = sum(final[final['Brand_no']==i]['NPS']<=6)/len(final[final['Brand_no']==i])
    NPS_Score[i-1] = (proms-dets)*100
    
print("NPS Score for Brands: \n",pd.DataFrame({'Brand_no':range(1,11), 'NPS_Score':NPS_Score}))   
    
## Calculating customer level CDI
avgs = final.loc[:, final.columns.str.startswith('avg')]
cust_cdi = avgs.apply(np.mean, axis=1)
cdi = pd.DataFrame({'Cust_no':final['customer_no'], 'Brand_no':final['Brand_no'], 'cdi':cust_cdi})
print("Customer level CDI: \n", cdi)

# Outputting to a txt file
import subprocess
with open("output_cdi.txt", "w+") as output:
    subprocess.call(["python", "./script.py"], stdout=output)    
    
