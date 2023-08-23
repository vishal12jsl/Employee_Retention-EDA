#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


# In[2]:


df=pd.read_csv("emp_retention.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df['Attrition'].value_counts()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df['Attrition'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%')
plt.show()


# ## from the pie chart out of 1470 employees 16% of the employee left their job due to some reasons and 84% of the employees preferred to continue their job 

# *removing useless features*

# In[9]:


df.columns


# In[10]:


df.drop(columns=['StandardHours','StockOptionLevel','Over18','EmployeeNumber','EmployeeCount'],inplace=True)
df.head()


# In[11]:


df.columns


# In[12]:


df.shape


# *analysis of rating feature*\
# *job satisfaction*\
# *environment satisfaction*\
# *relationship satisfaction*\
# *job environment*\
# *worklife balance*\
# *peeformance rating*
# 

# In[13]:


df['JobSatisfaction'].value_counts()


# In[14]:


df['JobSatisfaction'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%')


# In[15]:


fig=plt.figure()
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
labels='low','medium','high','veryhigh'

df['JobSatisfaction'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax1)
df['EnvironmentSatisfaction'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax2)
df['RelationshipSatisfaction'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax3)
df['JobInvolvement'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax4)
fig.legend(labels=labels,loc='center')
plt.show()


# ## from this plot we can infer that 
# **60% of the employees are not satisfied with the job**\
# **almost 61% of the employees are not satisfied with the environment**\
# **60% of the employees are not satisfied in their relationship**\
# **84% of the employees are not involved in their job**

# In[16]:


fig2=plt.figure()
ax5=fig2.add_subplot(121)
ax6=fig2.add_subplot(122)

labels_list1='bad','good','better','best'
labels_list2='low','good','excellent','outstanding'

df['WorkLifeBalance'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax5)
ax5.legend(labels=labels_list1,loc='upper right')

df['PerformanceRating'].astype(str).value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None,ax=ax6)
ax6.legend(labels=labels_list2,loc='upper right')
plt.show()


# ## From the above piecharts we can say that
# **Almost 61% of the employee have rated their worklife balance as bad**\
# **Almost 85% of the employee have a low performance rating**

# ## Analysis of business travel feature

# In[17]:


df.columns


# In[18]:


data=df.groupby("BusinessTravel")['Attrition'].value_counts(normalize=False).unstack()
data.plot(kind='bar',alpha=1,stacked='False')
plt.title('Business travel vs Attrition')
plt.ylabel("Number of employee")
plt.show()


# **From the above data it is clear that employee who travels rarely have more have more attrition rate forllowed by employee 
# who travels frequently**\
# *best way to reduce the attrition is to conduct monthly survey and to assign travel according to the employees business travel interest*

# ## Analysis of work experience
# **Years at company**\
# **years in current role**\
# **years since last promotion**\
# **years with current manager**\
# **total working years**

# In[19]:


df.columns


# In[20]:


we=df[['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TotalWorkingYears']]


# In[21]:


we.head()


# In[22]:


yac=df.groupby("YearsAtCompany")['Attrition'].value_counts(normalize=False).unstack()
yac.plot(kind='bar',stacked='False',figsize=(10,6))
plt.title("years at company of an employee")
plt.ylabel("No of employees")
plt.show()


# **It is observed that newly joined employee quit their jobs most so more concern should be given to the freshers and their cause of leaving the company**

# In[23]:


ywcm=df.groupby("YearsWithCurrManager")['Attrition'].value_counts(normalize=False).unstack()
twy=df.groupby("TotalWorkingYears")['Attrition'].value_counts(normalize=False).unstack()
fig=plt.figure()
ax0=fig.add_subplot(121)
ax1=fig.add_subplot(122)
ywcm.plot(kind='bar',stacked='False',figsize=(20,6),ax=ax0)
ax0.set_title("same role")
ax0.set_xlabel("Years with manager")
ax0.set_ylabel("no of emloyees")


twy.plot(kind='bar',stacked='False',figsize=(25,8),ax=ax1)
ax1.set_title("work exerience")
ax1.set_xlabel("total years")
ax1.set_ylabel("no of emloyees")
plt.show()


# **from first grapgh it is clear that relationship between employee and manager was not so good as we saw attrition in the starting years , so it is important that manager communicate with the employee from the starting in order to reduce the attrition**\
# \
# **second graph shows that fresher leave the company so new policy should be implemented in so that freshers dont leave the company**

# In[24]:


ycr=df.groupby("YearsInCurrentRole")['Attrition'].value_counts(normalize=False).unstack()
ylp=df.groupby("YearsSinceLastPromotion")['Attrition'].value_counts(normalize=False).unstack()
fig=plt.figure()
ax0=fig.add_subplot(121)
ax1=fig.add_subplot(122)
ywcm.plot(kind='bar',stacked='False',figsize=(20,6),ax=ax0)
ax0.set_title("same role")
ax0.set_xlabel("Years in current role")
ax0.set_ylabel("no of emloyees")


twy.plot(kind='bar',stacked='False',figsize=(25,8),ax=ax1)
ax1.set_title("last experience")
ax1.set_xlabel("years since last promotion")
ax1.set_ylabel("no of emloyees")
plt.show()


# **from te above two graphs it is clear that employees who are in same post and not getting promoted tend to leave the company,experienced employees leaving their job would affect the company most**

# ## Analysis on Department

# In[25]:


df


# 

# In[27]:


df.columns


# In[28]:


dpt=df[['Department','Attrition']]
dpt.head()


# In[29]:


dpt['Department'].value_counts()


# In[32]:


dpt['Department'].value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None)
plt.axis('equal')
plt.legend(labels=dpt['Department'].unique())
plt.show()


# In[33]:


gda=df[['Gender','DistanceFromHome','Attrition']]
gda.head()


# In[35]:


gda['Gender'].value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None)
plt.axis('equal')
plt.legend(labels=gda['Gender'].unique())
plt.show()


# In[37]:


ms=df[['MaritalStatus','Attrition']]
ms.head()


# In[38]:


ms['MaritalStatus'].value_counts()


# In[40]:


ms['MaritalStatus'].value_counts().plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',labels=None)
plt.axis('equal')
plt.legend(labels=ms['MaritalStatus'].unique())
plt.show()


# In[ ]:




