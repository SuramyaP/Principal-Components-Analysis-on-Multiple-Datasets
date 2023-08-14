#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# In[21]:


digits = datasets.load_digits()
Y = digits.target
# digits.DESCR


# In[3]:


df = pd.DataFrame(data = digits.data, columns = digits.feature_names)
df


# In[4]:


scaler = StandardScaler()
normalized_digits = scaler.fit_transform(df)
df = pd.DataFrame(data = normalized_digits, columns = digits.feature_names)


# In[5]:


df


# In[6]:


#Calculating Co-variance Matrix
covariance_mat = np.cov(df.T)
covariance_mat.shape


# In[7]:


#Calculating Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
print(eigenvalues)
print(eigenvectors.shape)


# In[8]:


#Calulating Proportion Of Variance
# print(type(eigenvalues))
tot = eigenvalues.sum()
pov = (eigenvalues / tot) * 100
print(pov)
print('Total Proportion of Variance = ', pov.sum())
print('Looks like the most important PCs are PC0, PC1, PC2, PC3 and so on.)')


# In[9]:


#Change of Basis (Y= PX)
P = eigenvectors.copy()
X = df.T
print(P.shape)
print(X.shape)
Y = np.matmul(P.T, X)
Y


# In[10]:


#Checking if the diagonal elements are dominant and non diagonal elements are nearly or not
cov_y = np.cov(Y)
cov_y = pd.DataFrame(data = cov_y)
print("As you can see the diagonal elements are dominant and non-diagonal elements are almost 0")
cov_y


# In[11]:


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


# In[12]:


targets =digits.target


# In[13]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([0]).T
target_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
# print(target_names)
# print(Plot_mat.shape)
# Define marker styles for each class
marker_styles = ['+' ,'X' ,'*','o', 's', '^', 'D', 'p', 'P', 'x']  # Use different markers for each class
# print(Plot_mat)
# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    # print(label)
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


# In[14]:


#Taking Only 1 Principle Component
Plot_mat = PCAAnalysis([63]).T
# print(target_names)
# print(Plot_mat.shape)
# Define marker styles for each class
# print(Plot_mat)
# Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[mask], [0] * len(Plot_mat[0][mask]), marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC63')
plt.grid()
plt.title('PCA Scatter Plot')


# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[63].png')

plt.show()


# In[15]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,1]).T
Plot_mat = np.asarray(Plot_mat)

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


# In[16]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,9]).T
Plot_mat = np.asarray(Plot_mat)

# # Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[:, 0][mask],Plot_mat[:, 1][mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC0')
plt.ylabel('PC9')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,9].png')


plt.show()


# In[17]:


#Taking Only 2 Principle Component
Plot_mat = PCAAnalysis([0,60]).T
Plot_mat = np.asarray(Plot_mat)

# # Plot the scatter plot with different markers
scatter_handles = []
for label in set(targets):
    mask = [t == label for t in targets]
    scatter = plt.scatter(Plot_mat[:, 0][mask],Plot_mat[:, 1][mask], marker=marker_styles[label], label=target_names[label])
    scatter_handles.append(scatter)

plt.xlabel('PC0')
plt.ylabel('PC60')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[0,60].png')


plt.show()


# In[18]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([0,1,2]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 


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


# In[19]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([1,9,40]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 


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
ax.set_ylabel('PC9')
ax.set_zlabel('PC40')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[1,9,40].png')


plt.show()


# In[20]:


#Taking Only 3 Principle Component
Plot_mat = PCAAnalysis([60,61,62]).T
Plot_mat = np.asarray(Plot_mat)

# print(Plot_mat.shape) 


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

ax.set_xlabel('PC60')
ax.set_ylabel('PC61')
ax.set_zlabel('PC62')
# plt.axis('equal')
plt.grid()
plt.title('PCA Scatter Plot')

# Add legend
plt.legend(handles=scatter_handles, title='Targets')
plt.savefig('[60,61,62].png')


plt.show()


# In[ ]:




