#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

import sys
data = pd.read_csv(sys.argv[1],index_col=0)
data = data.T
# data.head()


# In[3]:


plt.figure(0)
fig, axes = plt.subplots(3,3,figsize=(15,15))
axes = axes.flatten()

i = 0
ctr = 0
while i-ctr < 9:
    try:
        col = data.iloc[:,i]
        col.plot.kde(ax=axes[i-ctr],label='Data')
        axes[i-ctr].legend()
        axes[i-ctr].set_title(data.columns[i])
    except:
        ctr += 1
    i += 1


# In[4]:


plt.figure(0)
fig, axes = plt.subplots(3,3,figsize=(15,15))
axes = axes.flatten()

i = 0
ctr = 0
while i-ctr < 9:
    
    try:
        col = data.iloc[:,i]
        col.plot.kde(ax=axes[i-ctr],label='Data')
        pd.Series(np.random.poisson(np.mean(col),5000)).plot.kde(ax=axes[i-ctr],label='Poisson')
        
        axes[i-ctr].legend()
        axes[i-ctr].set_title(data.columns[i])
    except:
        ctr += 1
    i += 1
    
# fig.savefig('distribution.png')


# In[5]:


#MLE of lambda of poisson distribution is the sample mean
MLE = np.mean(data)
print('MLE of lambda of first 5 genes')
print(MLE[:5])


# In[6]:


#Diagnostic Plots
var = MLE #For Poisson, variance = mean = lambda
plt.title('Diagnostic Plot')
plt.scatter(np.mean(data),np.log(np.var(data)),label='Data')
plt.scatter(MLE,np.log(MLE),label='Poisson')
plt.xlabel('Mean')
plt.ylabel('Variance (in log scale)')
plt.legend()
plt.savefig('Diagnostic.png')
plt.show()

