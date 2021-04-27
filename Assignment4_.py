#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Starter Packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis


# In[2]:


#Random number generation using Numpy.Random.Normal 
f1 = np.random.normal(0,1,(10000))
f2 = np.random.normal(0,1,(10000))

#
#Display of F1 and F2, specified in terms of distribution parameters 
print("First randomm generation values : \n", f1)
print("Second random generation values : \n", f2) 


# In[3]:


#Generation of 10,000 samples of 5 OBSERVED linear features
E = np.random.normal(0,1)
x1 = 2*f1 + 3*f2 + E
x2 = f1 - 10*f2 + E 
x3 = (-5*f1) + 5*f2 + E 
x4 = 7*f1 + 11*f2 + E
x5 = (-6*f1) - 7*f2 + E
#
print("X1 : \n", x1)
print("X2 : \n", x2)
print("X3 : \n", x3)
print("X4 : \n", x4)
print("X5 : \n", x5)


# In[4]:


#Now we can generate a np.array with the output from x1...x5 
array = np.array((x1, x2, x3, x4, x5))
Tarray = np.transpose(array)
print(len(Tarray))


# In[5]:


#Use StandardScaler to make each of the 5 features zero mean and unit variance 
scaler = StandardScaler()

#Now scale each feature individually and print out the results 
scaledArray = scaler.fit_transform(Tarray)
#
print("Scaled Array : \n", scaledArray)


# #### This is the PCA reduction code

# In[6]:


#First we will test the dimensionality reduction as a trial with all 5 components 
pca = PCA(n_components = 5).fit(scaledArray)
print(pca.components_)


# In[7]:


#Now that we have tested the dimensionality reduction we wil use 2 instead of 5 
pCa2 = PCA(n_components = 2).fit(scaledArray)
print(pCa2.components_)

print(pCa2.transform(scaledArray))
#
scaledPCA2 = PCA(n_components = 2).fit_transform(scaledArray)


# #### This is the FactorAnalysis Reduction Code 

# In[8]:


#This is just to get a feel of the analysis that we will be doing
fa = FactorAnalysis().fit(scaledArray)
print(fa.components_)


# In[9]:


#Now we can use factor analysis to determin how many factors to drop. We already know we are going from 5 to 2. 
#But this creates a good visualization of the values. 

#If we print all 5 factors this is what we will see.3 of the factors prove to be useless 
print(FactorAnalysis(n_components = 5).fit_transform(scaledArray))

#When we print out 2, we see that the reduction works. 
print(FactorAnalysis(n_components = 2).fit_transform(scaledArray))
faArray = FactorAnalysis(n_components = 2).fit_transform(scaledArray)


# In[14]:


#Now that we have the dimensionality reduction we can begin plotting the results. 
#The plots here are representing the 4 combinations of PCA reduction 
X1 = f1
X2 = f2 
Y1 = scaledPCA2[:,0]
Y2 = scaledPCA2[:,1]
Z1 = faArray[:,0]
Z2 = faArray[:,1]

fig, axes= plt.subplots(nrows=2, ncols=2)

axes[0][0].scatter(X1,Y1, color = 'r')
axes[0][1].scatter(X1,Y2, color = 'g')
axes[1][0].scatter(X2,Y1, color = 'b')
axes[1][1].scatter(X2,Y2, color = 'y')

plt.tight_layout()
plt.show()


# In[15]:


#The plots here are representing the 4 combination of Factor Analysis reduction 

f, ax= plt.subplots(nrows=2, ncols=2)

ax[0][0].scatter(X1,Z1, color = 'r')
ax[0][1].scatter(X1,Z2, color = 'b')
ax[1][0].scatter(X2,Z1, color = 'g')
ax[1][1].scatter(X2,Z2, color = 'y')

plt.tight_layout()
plt.show()

