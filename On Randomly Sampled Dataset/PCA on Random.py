#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Step1: Generate a 20 * 2 matrix from a "Normal" (Gaussian) Distribution
arr1 = np.random.normal(size = (20,2))
arr1


# In[3]:


# Step2: Plot the data from step1 
x = arr1[:,0]
y = arr1[:,1]
plt.scatter(x,y)
plt.xlabel("X-Data Points")
plt.ylabel("Y-Data Points")
plt.title("Scatter Plot - Normal Distribution")
plt.show()


# In[4]:


#Step3: Generate a (2 * 2) matrix from a "uniform" distribution such that values are between [0,1]
arr2 = np.random.uniform(0,1,size = (2,2))
arr2


# In[5]:


#Step4: Multiply matrices from Step1 and Step3
datamat = np.matmul(arr1,arr2) #np.dot(arr1,arr2) Same Thing
datamat


# In[6]:


#Step5: Plot the data from Step4
x = datamat[:,0]
y = datamat[:,1]
plt.scatter(x,y)
plt.xlabel("X-Data Points")
plt.ylabel("Y-Data Points")
plt.title("Scatter Plot")
plt.show()


# In[7]:


#Step6: Find the variance along the data axis and the direction perpendicular to it.
var_dataaxs = np.var(y)
var_perp_dataaxs = np.var(x)
print(var_dataaxs)
print(var_perp_dataaxs)


# In[8]:


#Step7: Find Co-Variance Matrix
covar_mat = np.cov(datamat.T)
covar_mat


# In[9]:


#Step8: Calculate Eigen Vectors And EigenValues of Covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covar_mat)
print(eigenvalues)
print(eigenvectors)


# In[10]:


#Step9: Find Proportion of variance
tot = eigenvalues.sum()
p1 = eigenvalues[0] / tot
p2 = eigenvalues[1] / tot
print(p1*100)
print(p2*100)


# In[11]:


#Step10: Implement Y=PX where P is Row feature vector/Eigen vector with top row/column having main significant component (Here Column has) and X is the input data transposed
P = eigenvectors.copy()
P[:, [0, 1]] = P[:, [1, 0]]
print(P)
X = datamat.T
Y = np.matmul(P,X)


# In[12]:


#Step11: Calculate Co-variance matrix Sy
cov_y = np.cov(Y)
cov_y
# Note: As you can see the diagonal vales are dominant and the non diagonal values are almost 0


# In[13]:


#Step12: Plotting the values using both eigen vectors
# Note: Y is the matrix with both eigen vectors multiplied
print(Y.shape)
plt.scatter(Y[0,:], Y[1,:])
plt.axis('equal')
plt.grid()
plt.xlabel("Transformed X")
plt.ylabel("Transformed Y")
plt.title("Change of Basis")
plt.show()


# In[14]:


#Step13: Plotting the value using only one eigen vector i.e. Dimentionality reduction
eig_significant = P[:,0]
print(eig_significant.shape)
eig_significant = np.reshape(eig_significant, (1,2))
print(eig_significant.shape)

matrix_1d = np.matmul(eig_significant, X)
matrix_1d.shape
x_val = matrix_1d[0,:]
y_val = np.zeros_like(x_val)

plt.scatter(x_val,y_val)
plt.grid()
plt.xlabel("Transformed X")
plt.ylabel("Transformed Y")
plt.title("Dimentionality Reduction")
plt.show()

