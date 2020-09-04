#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import date,timedelta,datetime
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import pyodbc
from datetime import datetime
today = datetime.today()
import numpy as np
import time
from dateutil.relativedelta import relativedelta


# In[2]:


start = time.time()
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ABP_PRD;'     
                      'Database=CHURN_BASELINE_TRAIN;'
                      'UID=user;'
                      'PWD=password')
script_demo = 'select * from CHURN_BASELINE_TRAIN.dbo.DEMOGRAPHIC'
#script_acc = 'select * from CHURN_BASELINE_TRAIN.dbo.ACCOUNTS'#select top 1000 * from CHURN_BASELINE_TRAIN.dbo.DEMOGRAPHICSLAST_SEEN_DATE


dfDemo = pd.read_sql_query(script_demo, conn, index_col=None,parse_dates=True)

conn.close()
end = time.time()
print(end - start)


# ## Read in Account Table from The Database

# In[3]:


start = time.time()
start = time.time()
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ABP_PRD;'     
                      'Database=CHURN_BASELINE_TRAIN;'
                      'UID=user;'
                      'PWD=password')
script_acc = 'select * from CHURN_BASELINE_TRAIN.dbo.ACCOUNT'
dfAcc = pd.read_sql_query(script_acc, conn, index_col=None,parse_dates=True)
conn.close()
end = time.time()
print(end - start)


# ## Read in Transaction Table from The Database

# In[4]:


start = time.time()
start = time.time()
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ABP_PRD;'     
                      'Database=CHURN_BASELINE_TRAIN;'
                      'UID=user;'
                      'PWD=password')
script_TRANSACTION = 'select * from CHURN_BASELINE_TRAIN.dbo.[TRANSACTIONST]'
dfTRANSACTION = pd.read_sql_query(script_TRANSACTION, conn, index_col=None,parse_dates=None)
conn.close()
end = time.time()
print(end - start)


# In[5]:


dfTRANSACTION.shape


# ## Read in LastSeen Table from The Database

# In[6]:


start = time.time()
start = time.time()
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ABP_PRD;'     
                      'Database=CHURN_BASELINE_TRAIN;'
                      'UID=user;'
                      'PWD=password')
script_last_seen = 'select * from CHURN_BASELINE_TRAIN.dbo.LAST_SEEN_DATET'
dfast_seen = pd.read_sql_query(script_last_seen, conn, index_col=None,parse_dates=True)
conn.close()
end = time.time()
print(end - start)


# In[7]:


dfast_seen.shape


# ## Churn Date: today's date minus six months

# In[8]:


today_date = datetime.now()
churn_date = pd.to_datetime(today_date).normalize() + relativedelta(months=-6)
print(churn_date)


# ## Check the size of the Data

# In[9]:


dfDemo.shape


# ## Drop Duplicates

# In[10]:


Demo_duplicatedrop= dfDemo.drop_duplicates() 


# In[11]:


Demo_duplicatedrop.shape


# In[12]:


Demo_duplicatedrop.head(1)


# In[13]:


Demo_duplicatedrop.CIF_CREATION_DATE.isna().sum()


# ## Convert Creation date to age 

# In[14]:


Demo_duplicatedrop['CIF_AGE'] = Demo_duplicatedrop['CIF_CREATION_DATE'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))


# In[15]:


Demo_duplicatedrop.head(1)


# ## Convert Age to int 

# In[16]:


Demo_duplicatedrop['AGE'].isnull().sum()


# In[17]:


Demo_duplicatedrop.AGE.fillna(Demo_duplicatedrop.AGE.median(),inplace =True)


# In[18]:


Demo_duplicatedrop['CIF_AGE'].isnull().sum()


# In[19]:


Demo_duplicatedrop['AGE'] = pd.to_numeric(Demo_duplicatedrop['AGE'], errors='coerce')
Demo_duplicatedrop['CIF_AGE'].astype(str).astype(int)


# In[20]:


Demo_duplicatedrop['AGE'].isnull().sum()


# In[21]:


Demo_duplicatedrop.SEX.unique()


# In[22]:


Demo_duplicatedrop["GENDER"] = Demo_duplicatedrop["SEX"].apply(lambda x: "Male" if x =='M'  else 'Female' if x=='F' else 'Other_gender')


# In[23]:


Demo_duplicatedrop.HAS_DEBIT_CARD.unique()


# In[24]:


Demo_duplicatedrop["HAS_DEBITCARD"] = Demo_duplicatedrop["HAS_DEBIT_CARD"].apply(lambda x: 'Y' if x =='YES'  else 'N' if x=='NO' else 'N')


# In[25]:


Demo_duplicatedrop["MARITALSTATUS"] = Demo_duplicatedrop["MARITAL_STATUS"].apply(lambda x: 'M' if x =='Married'  else 'S' if x=='Single' else 'Other_Marital_status')


# In[26]:


Demo_duplicatedrop.head(1)


# ##  Converting Categorical Columns to Numeric Columns

# In[27]:


SEX = pd.get_dummies(Demo_duplicatedrop.GENDER).iloc[:,1:]#pick the first twocolumns
HAS_DEBIT_CARD = pd.get_dummies(Demo_duplicatedrop.HAS_DEBIT_CARD).iloc[:,1:]
MARITAL_STATUS = pd.get_dummies(Demo_duplicatedrop.MARITALSTATUS).iloc[:,1:]


# In[28]:


dataset = pd.concat([Demo_duplicatedrop,SEX,HAS_DEBIT_CARD,MARITAL_STATUS], axis=1)


# In[29]:


dataset.head(1)


# In[30]:


dataset.columns


# In[31]:


dataset.rename(columns={'F':'GENDER'}, inplace=True)
dataset.rename(columns={'M':'MALE'}, inplace=True)
dataset.rename(columns={'YES':'DEBT_CARD'}, inplace=True)# 1 for has debitcard


# In[32]:


dataset.GENDER.unique()


# In[33]:


dataset.head(1)


# In[34]:


dataset.shape


# ## Reading other tables

# In[35]:


dfAcc.head(1)


# In[36]:


dfAcc.shape


# ## Drop Duplicate Accounts

# In[37]:


dfAcc= dfAcc.drop_duplicates() 


# In[38]:


dfAcc.shape


# In[39]:


#dfAcc['nb_months'] = ((datetime.now() - dfAcc.AC_OPEN_DATE)/np.timedelta64(1, 'M')).astype(int)


# In[40]:


#dfAcc.head(1)


# ## Number of account less than 6months 

# In[41]:


(dfAcc['AC_OPEN_DATE'] > churn_date).value_counts()


# ## Drop Accounts less than 6months

# In[42]:


dfAcc= dfAcc[dfAcc['AC_OPEN_DATE'] < churn_date]


# In[43]:


#dfAcc.nb_months.isna().sum()


# In[44]:


#(dfAcc['nb_months'] < 6).value_counts()


# In[45]:


#dfAcc= dfAcc[dfAcc['nb_months'] > 6]


# In[46]:


dfAcc.shape


# In[47]:


dfAcc.info()


# In[48]:


dfAcc[dfAcc['CUST_NO'] == '014106211'] #Sanity check


# In[49]:


dfAcc.info()


# In[50]:


dfAcc['BAL_MONTHEND'] = pd.to_numeric(dfAcc['BAL_MONTHEND'], errors='coerce')


# In[51]:


dfAcc.shape


# ## Get the number of account

# In[52]:


#dfAcc[['CUST_NO','BAL_MONTHEND']].groupby('CUST_NO').sum()


# In[53]:


acc_count =dfAcc.groupby(['CUST_NO'],as_index=False)["BAL_MONTHEND"].agg(['sum','count']).reset_index()


# In[54]:


(acc_count['count'] > 1).value_counts()


# In[55]:


acc_count.head()


# In[56]:


acc_count.shape


# In[57]:


#dfAcc_cust.head(1)


# In[58]:


#dfAcc_cust[dfAcc_cust['CUST_NO'] == '014106211']#Sanity check


# In[59]:


#dfAcc_cust.shape


# In[60]:


acc_final= pd.merge(dfAcc,acc_count,how ='left', left_on=['CUST_NO'], right_on=['CUST_NO'])


# In[61]:


acc_final.head(1)


# In[62]:


acc_final.shape


# ## Join Account Balance to Demographics

# In[63]:


dataset= pd.merge(dataset,acc_final,how ='inner', left_on=['CUSTOMER_NO'], right_on=['CUST_NO'])


# In[64]:


dataset.head(1)


# In[65]:


dataset.shape


# In[66]:


dataset=dataset.drop(['CIF_CREATION_DATE', 'CIF_CREATION_DATE','MARITAL_STATUS','GENDER','SEX','HAS_DEBIT_CARD','CUST_NO','ACCOUNT_CLASS'], axis=1)


# In[67]:


dataset[dataset['CUSTOMER_NO'] == '014106211']


# In[68]:


dfast_seen.head(1) 


# In[69]:


dfast_seen.shape


# ## Join Account to Last Seen

# In[70]:


acc_lastseen= pd.merge(dfAcc,dfast_seen,how ='left', left_on=['CUST_AC_NO'], right_on=['AC_NO'])


# In[71]:


acc_lastseen.shape 


# In[72]:


acc_lastseen.head(1)


# In[73]:


#acc_lastseen_cust = acc_lastseen.groupby(['CUST_NO'],as_index=False)[["LAST_TRN_DATE"]].max()


# In[74]:


#acc_lastseen_cust.head(1)


# ## Transactions

# In[75]:


dfTRANSACTION.head(1)


# In[76]:


dfTRANSACTION.shape


# In[77]:


dfTRANSACTION.columns


# ## Transforming the transaction table

# In[78]:


tran_pivot_AMT = dfTRANSACTION.pivot(index='CUST_AC_NO', columns='DRCR_IND', values='LCY_AMOUNT')
tran_pivot_AMT.reset_index(inplace = True)
tran_pivot_AMT.rename(columns=lambda x: x.replace('CUST_AC_NO', 'CUST_AC_NO'),inplace=True)
# tran_pivot.head()


# In[79]:


tran_pivot_AMT.rename(columns={'C':'CREDIT_AMOUNT'}, inplace=True)
tran_pivot_AMT.rename(columns={'D':'DEBIT_AMOUNT'}, inplace=True)


# In[80]:


tran_pivot_COUNT = dfTRANSACTION.pivot(index='CUST_AC_NO', columns='DRCR_IND', values='TRN_COUNT')
tran_pivot_COUNT.reset_index(inplace = True)
tran_pivot_COUNT.rename(columns=lambda x: x.replace('CUST_AC_NO', 'CUST_AC_NO'),inplace=True)


# In[81]:


tran_pivot_COUNT.rename(columns={'C':'CREDIT_COUNT'}, inplace=True)
tran_pivot_COUNT.rename(columns={'D':'DEBIT_COUNT'}, inplace=True)


# In[82]:


tran_pivot_COUNT.head(1)


# In[83]:


tran_pivot_COUNT.shape


# In[84]:


tran_pivot_AMT.head(1)


# In[85]:


tran_pivot_AMT.shape


# In[86]:


tran_pivot= pd.merge(tran_pivot_AMT,tran_pivot_COUNT,how ='left', on =['CUST_AC_NO'])


# In[87]:


tran_pivot.head(1)


# In[88]:


tran_pivot.shape


# ## Joining all the tables together

# In[89]:


acc_tran = pd.merge(dataset, tran_pivot, how='left', left_on=['CUST_AC_NO'], right_on=['CUST_AC_NO'])


# In[90]:


acc_tran.head()


# In[91]:


#acc_tran_cust=acc_tran.groupby(['CUST_NO'],as_index=False)[["CREDIT_AMOUNT","DEBIT_AMOUNT","CREDIT_COUNT","DEBIT_COUNT"]].sum()


# In[92]:


acc_tran.shape


# In[93]:


acc_lastseen.shape


# In[94]:


#acc_tran_cust.shape


# In[95]:


#acc_tran_cust.head(1)


# In[96]:


#acc_tran = pd.DataFrame(acc_tran[['CUST_NO','CREDIT_AMOUNT','DEBIT_AMOUNT','CREDIT_COUNT', 'DEBIT_COUNT', 'CUST_AC_NO' ]].groupby("CUST_NO").agg('sum'))


# In[97]:


data =pd.merge(acc_tran, acc_lastseen, how='left', left_on=['CUST_AC_NO'], right_on=['AC_NO'])


# In[98]:


data.head(1)


# In[99]:


data.shape


# In[100]:


data= data.drop_duplicates() 


# In[101]:


data.shape


# In[102]:


#data1=pd.merge(data, acc_tran_cust, how='left', left_on=['CUSTOMER_NO'], right_on=['CUST_NO'])


# In[103]:


#data1.head()


# In[104]:


#data1.shape


# In[105]:


#datafinalmerge = pd.merge(data1, acc_tran, how='left', left_on=['CUST_AC_NO'], right_index=True)
#datafinalmerge.head(1)


# In[106]:


#datafinalmerge.shape


# In[107]:


#data1.columns


# In[108]:


#data_drop= data1.drop(['CIF_CREATION_DATE', 'MARITAL_STATUS','SEX','HAS_DEBIT_CARD','MALE','O','P','CUST_NO','CUST_NO_x','CUST_NO_y'], axis = 1) 


# In[109]:


#data_drop.columns


# In[110]:


#data_drop['ACC_AGE'] = data_drop['AC_OPEN_DATE'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))


# ## Get the numbers of  last transaction days 

# In[111]:


data['Target_in_days'] =(datetime.now() - data['LAST_TRN_DATE']).dt.days


# In[112]:


data['Target_in_days'].isna().sum()


# In[113]:


data.tail(5)


# In[114]:


data.shape


# In[115]:


data.Target_in_days.fillna(1,inplace =True)


# In[116]:


data.Target_in_days.isna().sum()


# ## Creation of Target Variable

# In[117]:


#data['Target'] = np.where(data['LAST_TRN_DATE'] > 180, 1, 0)


# In[118]:


data['Target'] = np.where(data['LAST_TRN_DATE'] < churn_date, 1, 0)


# In[119]:


data.head(1)


# In[230]:


data.groupby(['CUST_NO'],as_index=False)["Target"].agg(['sum','count']).reset_index()


# In[157]:


Result = data[['CUSTOMER_NO','AC_NO','Target']]


# In[184]:


from datetime import date, timedelta
import calendar
prev_month = pd.to_datetime(date.today()).normalize() + relativedelta(months=-1)
Result['CHURN_RUN_DATE'] = prev_month.replace(day=calendar.monthrange(date.today().year, prev_month.month)[1])


# In[187]:


Result.head(1)


# In[188]:


Result.to_csv(r'C:\Users\abdul\Desktop\Pred.csv')


# In[120]:


data.shape


# In[ ]:





# In[121]:


data.Target.isna().sum()


# In[122]:


data['Target'].value_counts()


# In[123]:


sizes = data['Target'].value_counts(sort = True)
colors = ["red","green"] 
rcParams['figure.figsize'] = 5,5
labels=['non-active', 'active']
# Plot
plt.pie(sizes,colors=colors,labels =labels,
        autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Churn in Dataset')
plt.show()


# In[229]:


data.head(1)


# In[124]:


data.columns


# In[125]:


finaldataset =  data.drop(['CUSTOMER_NO','NO','HAS_DEBITCARD' ,'sum','MARITALSTATUS','LAST_TRN_DATE','CUST_AC_NO_x','CUST_AC_NO_y','AC_OPEN_DATE_x','CUST_NO','BAL_MONTHEND_x','ACCOUNT_CLASS','AC_NO','Target_in_days','AC_OPEN_DATE_y',], axis=1)


# In[126]:


finaldataset.isna().sum()


# In[127]:


finaldataset.BAL_MONTHEND_y .fillna(0,inplace =True)
finaldataset.CREDIT_AMOUNT.fillna(0,inplace =True)
finaldataset.DEBIT_AMOUNT.fillna(0,inplace =True)
finaldataset.DEBIT_COUNT.fillna(0,inplace =True)
finaldataset.CREDIT_COUNT.fillna(0,inplace =True)


# In[128]:


#finaldataset['BAL_MONTHEND'] = pd.to_numeric(finaldataset['BAL_MONTHEND_y'], errors='coerce')


# In[129]:


finaldataset.head()


# In[130]:


pd.DataFrame(finaldataset).fillna(0, inplace =True)


# ##  Isolate the variable (target) that we’re predicting from the dataset.
# 

# In[131]:


X =  finaldataset.drop(['Target'], axis=1)
y = finaldataset['Target']


# ## We’ll use 20% of the data for the test set and the remaining 80% for the training set

# In[132]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#  ## Random forest algorithm, since it’s simple and one of the most powerful algorithms for classification problems.

# In[133]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)


# In[134]:


predictions[:20]


# ## Machine Learning Algorithm Evaluation

# In[135]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))


# In[204]:


pwd


# In[205]:


import pickle
import os
pickle.dump(classifier,open(r'C:\\Users\\abdul\\Desktop\\finalized_model.sav','wb'))


# In[206]:


load_model  = pickle.load(open(r'C:\\Users\\abdul\\Desktop\\finalized_model.sav','rb'))


# In[208]:


Predict = load_model.predict(X_train)


# In[212]:


X_train


# In[224]:


Predict[:20]


# In[225]:


classifier.predict_proba(X_test)[:, 0][:20]


# In[226]:


classifier.predict_proba(X_test)[:20]


# In[228]:


a.max(axis=1)[:20]


# In[210]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_train,Predict ))  
print(accuracy_score(y_train, Predict ))


# In[211]:


y_train


# In[136]:


feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[137]:


from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)


# In[138]:


sel.get_support()


# In[139]:


selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)


# In[140]:


print(selected_feat)


# In[141]:


finaldataset[finaldataset.columns[:]].corr()['Target']


# In[142]:


finaldataset.info()


# In[147]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=10)
print('Scores =', scores)


# In[154]:


scores.mean()


# ##  XGBoost Model

# In[144]:


from sklearn import datasets
from sklearn import metrics
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[145]:


from xgboost import XGBClassifier


# In[146]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[148]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[153]:


predictions[:20] #array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])


# In[150]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:




