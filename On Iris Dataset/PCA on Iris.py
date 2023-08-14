#!/usr/bin/env python
# coding: utf-8

# In[88]:


from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# In[89]:


iris = datasets.load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names )
targets = iris.target
df.head()


# In[90]:


df.shape


# In[91]:


scaler = StandardScaler()
normalized_iris = scaler.fit_transform(df)
df = pd.DataFrame(data = normalized_iris, columns = iris.feature_names)


# In[92]:


df.head()


# In[93]:


slength = normalized_iris[:,0]
swidth = normalized_iris[:,1]
plength = normalized_iris[:,2]
pwidth = normalized_iris[:,3]
feature_arr = [slength, swidth, plength, pwidth]
var_arr = []
for i in range(len(feature_arr)):
    calc_var = np.var(feature_arr[i])
    var_arr.append(calc_var)
var_arr


# In[94]:


covariance_mat = np.cov(df.T)


# In[95]:


covariance_mat


# In[96]:


#Calculating Eigenvector And EigenValues
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
print(eigenvalues)
print(eigenvectors)


# In[97]:


#Finding Proportion of variance
tot = eigenvalues.sum()
pov = (eigenvalues/tot) * 100
pov


# In[98]:


#Change of basis (Y = PX)
P = eigenvectors.copy()
X = df.T
print(P.shape)
print(X.shape)
Y = np.matmul(P.T,X)
Y


# In[99]:


Y.shape


# In[13]:


cov_y = np.cov(Y)
print(cov_y)
print("As you can see the diagonal elements are dominant and non-diagonal elements are almost 0")


# In[14]:


def PCAAnalysis(set):
    print("Subset: ", set)
    print("No of principle components: ", len(set))
    neweig = eigenvectors[:,set]
    # print(neweig.shape)
    # print(X.shape)
    mat = np.matmul(neweig.T,X)
    covariance_calc = np.cov(mat)
    print("Covariance for this subset is: ")
    print(covariance_calc)
    return mat



# In[80]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([0]).T
target_names = ['Setosa','Versicolor', 'Virginica']
# print(target_names)
# print(Plot_mat.shape)
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class
# print(Plot_mat)
# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[mask], [0] * len(Plot_mat[0][mask]), marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC0')
plt.grid()
plt.title('PCA Scatter Plot')


# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0].png')

plt.show()


# In[100]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([3]).T
target_names = ['Setosa','Versicolor', 'Virginica']
# print(target_names)
# print(Plot_mat.shape)
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class
# print(Plot_mat)
# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[mask], [0] * len(Plot_mat[0][mask]), marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC3')
plt.grid()
plt.title('PCA Scatter Plot')


# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[3].png')


plt.show()


# In[82]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,1]).T
Plot_mat = np.asarray(Plot_mat)


# # Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# # Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[:, 0][mask],Plot_mat[:, 1][mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC0')
plt.ylabel('PC1')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,1].png')


plt.show()


# In[83]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([1,2])
Plot_mat = np.asarray(Plot_mat)


# print(Plot_mat[0].shape)
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[0][mask],Plot_mat[1][mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[1,2].png')


plt.show()


# In[ ]:





# In[84]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([2,3])
Plot_mat = np.asarray(Plot_mat)

print(Plot_mat.shape)
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[0][mask],Plot_mat[1][mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC2')
plt.ylabel('PC3')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[2,3].png')


plt.show()


# In[85]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([0,1,2]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# Plot the scatter plot with different markers
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection='3d')
a = Plot_mat[:, 0]
b = Plot_mat[:, 1]
c = Plot_mat[:, 2]

scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = ax.scatter(a[mask],b[mask], c[mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,1,2].png')


plt.show()


# In[86]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([1,2,3]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# Plot the scatter plot with different markers
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection='3d')
a = Plot_mat[:, 0]
b = Plot_mat[:, 1]
c = Plot_mat[:, 2]

scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = ax.scatter(a[mask],b[mask], c[mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[1,2,3].png')


plt.show()


# In[87]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([0,2,3]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 
# Define marker styles for each class
marker_styles = ['o', 's', '^']  # Use different markers for each class

# Plot the scatter plot with different markers
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection='3d')
a = Plot_mat[:, 0]
b = Plot_mat[:, 1]
c = Plot_mat[:, 2]

scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = ax.scatter(a[mask],b[mask], c[mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

ax.set_xlabel('PC0')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,2,3].png')


# plt.tight_layout()

plt.show()


# In[ ]:




