#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv("https://raw.githubusercontent.com/training-ml/Files/main/diabetes.csv")


# In[3]:


data.tail()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.isna().sum()


# In[7]:


data.info()


# In[8]:


data.describe()


# Need to correction in pregnacnies
# Problem - think practically
# Glucose , BloodPressure, Skinthickness , Insulin is 0 that won't we possible

# In[9]:


plt.figure(figsize=(20,20),facecolor='w')
plotno=1
for col in data:
    if plotno<=8:
        plt.subplot(2,4,plotno)
        sns.distplot(data[col])
        plt.xlabel(col)  
    plotno+=1
plt.show()
    


# In[10]:


#we can see there is some skewness in the data


# In[11]:


data['BMI']=data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())


# In[12]:


plt.figure(figsize=(20,20),facecolor='w')
plotno=1
for col in data:
    if plotno<=8:
        plt.subplot(2,4,plotno)
        sns.distplot(data[col])
        plt.xlabel(col)  
    plotno+=1
plt.show()
    


# In[13]:


df_features=data.drop('Outcome',axis=1)


# In[14]:


# Visulize the outliers using boxplot
plt.figure(figsize=(15,20))
plotno=1
for col in df_features:
    if plotno<=30:
        plt.subplot(10,3,plotno)
        sns.boxplot(df_features[col])
        plt.xlabel(col,fontsize=10)
    plotno+=1
plt.show()


# In[15]:


#find the IQR to identify the outliers

#1st quartile
q1=data.quantile(0.25)
q3=data.quantile(0.75)

#IQR
iqr=q3-q1


# Outlier detection formula
# higher side ==>q3+(1.5*IQR)
# lower side == q1-(1.5*IQR)

# In[16]:


#Validating the outlier
preg_high = (q3.Pregnancies+(1.5*iqr.Pregnancies))
preg_high


# In[17]:


index=np.where(data['Pregnancies']>preg_high)
index


# In[18]:


data=data.drop(data.index[index])
data.shape


# In[19]:


data.reset_index()


# In[20]:


bp_high=(q3.BloodPressure+(1.5*iqr.BloodPressure))
bp_high


# In[21]:


index=np.where(data['BloodPressure']>bp_high)
index


# In[22]:


data=data.drop(data.index[index])
data.shape


# In[23]:


data.reset_index()


# In[24]:


st_high=(q3.SkinThickness+(1.5*iqr.SkinThickness))
print(st_high)

index=np.where(data['SkinThickness']>st_high)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[25]:


insu_high=(q3.Insulin+(1.5*iqr.Insulin))
print(insu_high)

index=np.where(data['Insulin']>insu_high)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[26]:


BMI_high=(q3.BMI+(1.5*iqr.BMI))
print(BMI_high)

index=np.where(data['BMI']>BMI_high)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[27]:


dpf_high=(q3.DiabetesPedigreeFunction+(1.5*iqr.DiabetesPedigreeFunction))
print(dpf_high)

index=np.where(data['DiabetesPedigreeFunction']>dpf_high)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[28]:


age_high=(q3.Age+(1.5*iqr.Age))
print(age_high)

index=np.where(data['Age']>age_high)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[29]:


bp_low=(q1.BloodPressure-(1.5*iqr.BloodPressure))
print(bp_low)

index=np.where(data['BloodPressure']<bp_low)
index

data=data.drop(data.index[index])
data.shape

data.reset_index()


# In[ ]:





# In[30]:


# Visulize the outliers using boxplot
plt.figure(figsize=(15,20))
plotno=1
for col in data:
    if plotno<=9:
        plt.subplot(3,3,plotno)
        sns.distplot(data[col])
        plt.xlabel(col,fontsize=10)
    plotno+=1
plt.show()


# In[31]:


x=data.drop(columns='Outcome')
y=data['Outcome']


# #stripplot - we use to find the relaton between dependent& independent variables

# In[32]:


plt.figure(figsize=(20,20),facecolor='w')
plotno=1
for col in x:
    if plotno<=8:
        ax=plt.subplot(3,3,plotno)
        sns.stripplot(y,x[col])
        plt.xlabel(col)  
    plotno+=1
plt.show()
    


# Great !! let's check the multicolinearity in the dependent variables

# In[33]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#for the normalization


# In[34]:


x_scaled.shape[1]


# In[36]:


#Finding the multicollinerity
vif=pd.DataFrame()
vif["vif"]=[variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif["Features"]=x.columns

#let's check the values
vif


# vif values less than 5 that's means no multicollinearly
# 

# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=.25,random_state=355)


# In[39]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[40]:


import pickle
#writng diffrent model file to file
with open('modeForPrediction.sav','wb') as f:
    pickle.dump(log_reg,f)
with open('sandardScalar.sav', 'wb') as f:
    pickle.dump(scaler,f)


# In[41]:


y_pred=log_reg.predict(x_test)


# In[45]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[46]:


conf_mat=confusion_matrix(y_test,y_pred)
conf_mat


# In[49]:


true_positive=conf_mat[0][0]
false_positive=conf_mat[0][1]
false_negative=conf_mat[1][0]
true_negative=conf_mat[1][1]


# In[52]:


Accuracy=(true_positive+true_negative)/(true_positive+false_positive+false_negative+true_negative)
Accuracy


# In[54]:


recall=true_positive/(true_positive+false_negative)
recall


# In[56]:


precision=true_positive/(true_positive+false_positive)
precision


# In[59]:


#F1_score
f1_Score=2*(recall*precision)/(recall+precision)
f1_Score


# In[ ]:




