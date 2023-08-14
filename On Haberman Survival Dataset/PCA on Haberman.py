#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


# In[2]:


current_dir = os.getcwd() + '\haberman_survival\haberman.data'
print(current_dir)
#Features in this dataset are: 
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (Class attribute)
#       -- 1 = the patient survived 5 years or longer
#       -- 2 = the patient died within 5 year

features_avail = ['PatientAge','OperationDate','DetectedNodes', 'Status'] 
df = pd.read_csv(current_dir, names = features_avail)


# In[3]:


print(df.shape)
df.head(10)


# In[4]:


df_arr = np.asarray(df)
X = np.delete(df_arr, 3, axis = 1)
targets = df_arr[:,3]
targets[targets == 1] = 0
targets[targets == 2] = 1
print(targets)
df = df.drop('Status', axis = 1)
print(df.shape)
df.head()


# In[5]:


scaler = StandardScaler()
normalized_haberman = scaler.fit_transform(df)
df = pd.DataFrame(data = normalized_haberman, columns = features_avail[0:-1])
df


# In[6]:


#Calculating Variance
var_arr = []
p_age = normalized_haberman[:,0]
p_od = normalized_haberman[:,1]
p_dn = normalized_haberman[:,2]
list_avail = [p_age,p_od,p_dn]
for i in range(len(list_avail)):
    calc_var = np.var(list_avail[i])
    var_arr.append(calc_var)
var_arr


# In[7]:


covariance_mat = np.cov(df.T)


# In[8]:


covariance_mat


# In[9]:


#Calculating Eigenvector And EigenValues
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
print(eigenvalues)
print(eigenvectors)


# In[10]:


#Finding Proportion of variance
tot = eigenvalues.sum()
pov = (eigenvalues/tot) * 100
pov


# In[43]:


#Change of basis (Y = PX)
P = eigenvectors.copy()
X = df.T
Y = np.matmul(P.T,X)
Y


# In[12]:


cov_y = np.cov(Y)
print(cov_y)
print("As you can see the diagonal elements are dominant and non-diagonal elements are almost 0")


# In[45]:


#Taking Only 3 Principle Component
Plot_mat = Y.T
Plot_mat = np.asarray(Plot_mat)

print(Plot_mat.shape) 
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


# In[13]:


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



# In[14]:


print(targets)


# In[30]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([0]).T
target_names = ['Survived >= 5', 'Survived <5']
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


# In[31]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([1]).T
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

plt.xlabel('PC1')
plt.grid()
plt.title('PCA Scatter Plot')


# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[1].png')


plt.show()


# In[32]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([2]).T

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

plt.xlabel('PC2')
plt.grid()
plt.title('PCA Scatter Plot')


# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[2].png')


plt.show()


# In[33]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,1])
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

plt.xlabel('PC0')
plt.ylabel('PC1')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,1].png')


plt.show()


# In[34]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([1,2])
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

plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[1,2].png')


plt.show()


# In[35]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,2])
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

plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,2].png')


plt.show()


# In[36]:


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


# In[ ]:




