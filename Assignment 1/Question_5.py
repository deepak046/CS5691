#!/usr/bin/env python
# coding: utf-8

# # General Instructions to students:
# 
# 1. There are 5 types of cells in this notebook. The cell type will be indicated within the cell.
#     1. Markdown cells with problem written in it. (DO NOT TOUCH THESE CELLS) (**Cell type: TextRead**)
#     2. Python cells with setup code for further evaluations. (DO NOT TOUCH THESE CELLS) (**Cell type: CodeRead**)
#     3. Python code cells with some template code or empty cell. (FILL CODE IN THESE CELLS BASED ON INSTRUCTIONS IN CURRENT AND PREVIOUS CELLS) (**Cell type: CodeWrite**)
#     4. Markdown cells where a written reasoning or conclusion is expected. (WRITE SENTENCES IN THESE CELLS) (**Cell type: TextWrite**)
#     5. Temporary code cells for convenience and TAs. (YOU MAY DO WHAT YOU WILL WITH THESE CELLS, TAs WILL REPLACE WHATEVER YOU WRITE HERE WITH OFFICIAL EVALUATION CODE) (**Cell type: Convenience**)
#     
# 2. You are not allowed to insert new cells in the submitted notebook.
# 
# 3. You are not allowed to **import** any extra packages.
# 
# 4. The code is to be written in Python 3.6 syntax. Latest versions of other packages maybe assumed.
# 
# 5. In CodeWrite Cells, the only outputs to be given are plots asked in the question. Nothing else to be output/print. 
# 
# 6. If TextWrite cells ask you to give accuracy/error/other numbers you can print them on the code cells, but remove the print statements before submitting.
# 
# 7. The convenience code can be used to check the expected syntax of the functions. At a minimum, your entire notebook must run with "run all" with the convenience cells as it is. Any runtime failures on the submitted notebook as it is will get zero marks.
# 
# 8. All code must be written by yourself. Copying from other students/material on the web is strictly prohibited. Any violations will result in zero marks.
# 
# 9. All datasets will be given as .npz files, and will contain data in 4 numpy arrays :"X_train, Y_train, X_test, Y_test". In that order. The meaning of the 4 arrays can be easily inferred from their names.
# 
# 10. All plots must be labelled properly, all tables must have rows and columns named properly.
# 
# 11. Change the name of file with your roll no.
# 
# 

# In[1]:


# Cell type : CodeRead

import numpy as np
import matplotlib.pyplot as plt


# **Cell type : TextRead**
# 
# # Problem 4: Learning Binary Bayes Classifiers from data with Max. Likelihood 
# 
# Derive Bayes classifiers under assumptions below, and use ML estimators to compute and return the results on a test set. 
# 
# BayesA) Assume $X|Y=-1 \sim \mathcal{N}(\mu_-, I)$ and  $X|Y=1 \sim \mathcal{N}(\mu_+, I)$. *(Same known covariance)*
# 
# BayesB) Assume $X|Y=-1 \sim \mathcal{N}(\mu_-, \Sigma)$ and $X|Y=1 \sim \mathcal{N}(\mu_+, \Sigma)$ *(Same unknown covariance)*
# 
# BayesC) Assume $X|Y=-1 \sim \mathcal{N}(\mu_-, \Sigma_-)$ and $X|Y=1 \sim \mathcal{N}(\mu_+, \Sigma_+)$ *(different unknown covariance)*
# 
# 

# In[2]:


# Cell type : CodeWrite

def compute_mean(X,Y,target): 
    return (X[Y==target].sum(axis=0)/X[Y==target].shape[0]).reshape((X.shape[1],1))

def compute_prior(X,Y):
    prior0 = (X[Y==-1].shape[0]/X.shape[0])
    prior1 = 1 - prior0
    return np.array([prior0,prior1]).reshape((2,1))

def gaussian_loglikelihood(data,mu,sigma,D=2):
    def helper(x): 
        x = x.reshape(D,1)
        return (np.matmul(np.matmul((x-mu).T,np.linalg.inv(sigma)),(x-mu)))
    return (-((D/2)*np.log(2*np.pi)) -((1/2)*np.log(np.linalg.det(sigma))) -((1/2)*np.apply_along_axis(helper,axis=1,arr=data))).reshape(data.shape[0],1)

def classify_using_prob(prob,thresh=0.5):
    Y_test_pred = np.zeros((prob.shape[0],))  
    np.putmask(Y_test_pred,prob >= thresh,+1)
    np.putmask(Y_test_pred,prob < thresh,-1)
    return Y_test_pred

def function_for_A(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption BayesA.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)
    
    """

    mu0 = compute_mean(X_train,Y_train,-1)
    mu1 = compute_mean(X_train,Y_train,1) 
    prior = compute_prior(X_train,Y_train)
    
    ll0 = np.exp(gaussian_loglikelihood(X_test,mu0,np.eye(2)))
    ll1 = np.exp(gaussian_loglikelihood(X_test,mu1,np.eye(2)))
    
    prob = prior[1]*ll1/(prior[0]*ll0+prior[1]*ll1)   
    return (classify_using_prob(prob),prob)
    
def function_for_B(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption BayesB.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)
    
    """

    mu0 = compute_mean(X_train,Y_train,-1)
    mu1 = compute_mean(X_train,Y_train,1) 
    prior = compute_prior(X_train,Y_train)
    
    cov_matrix = (X_train[Y_train==-1].shape[0]*np.cov(X_train[Y_train==-1].T) + 
                  X_train[Y_train==1].shape[0]*np.cov(X_train[Y_train==1].T))/X_train.shape[0]
    
    ll0 = np.exp(gaussian_loglikelihood(X_test,mu0,cov_matrix))
    ll1 = np.exp(gaussian_loglikelihood(X_test,mu1,cov_matrix))
    
    prob = prior[1]*ll1/(prior[0]*ll0+prior[1]*ll1) 
    return (classify_using_prob(prob),prob)

def function_for_C(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption BayesC.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)
    
    """
    mu0 = compute_mean(X_train,Y_train,-1)
    mu1 = compute_mean(X_train,Y_train,1)
    
    prior = compute_prior(X_train,Y_train)
    
    ll0 = np.exp(gaussian_loglikelihood(X_test,mu0,np.cov(X_train[Y_train==-1].T)))
    ll1 = np.exp(gaussian_loglikelihood(X_test,mu1,np.cov(X_train[Y_train==1].T)))
    
    prob = prior[1]*ll1/(prior[0]*ll0+prior[1]*ll1)
    return (classify_using_prob(prob),prob)


# In[3]:


# Cell type : Convenience

# Testing the functions above

# To students: You may use the example here for testing syntax issues 
# with your functions, and also as a sanity check. But the final evaluation
# will be done for different inputs to the functions. (So you can't just 
# solve the problem for this one example given below.) 
# try to remove everything or comment out your lines before submitting.


X_train_pos = np.random.randn(1000,2)+np.array([[1.,2.]])
X_train_neg = np.random.randn(1000,2)+np.array([[2.,4.]])
X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
Y_train = np.concatenate(( np.ones(1000), -1*np.ones(1000) ))
X_test_pos = np.random.randn(1000,2)+np.array([[1.,2.]])
X_test_neg = np.random.randn(1000,2)+np.array([[2.,4.]])
X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
Y_test = np.concatenate(( np.ones(1000), -1*np.ones(1000) ))

Y_pred_test_1a, _ = function_for_A(X_train, Y_train, X_test)
Y_pred_test_1b, _ = function_for_B(X_train, Y_train, X_test)
Y_pred_test_1c, _ = function_for_C(X_train, Y_train, X_test)


# **Cell type : TextRead**
# 
# # Problem 4
# 
# #### 4a) Run the above three algorithms (BayesA,B and C), for the two datasets given (datasetA.npz, datasetB.npz) in the cell below.
# #### In the next CodeWrite cell, Plot all the classifiers (3 classification algos on 2 datasets = 6 plots) on a 2d plot (color the positively classified area light green, and negatively classified area light red). Add the training data points also on the plot. Plots to be organised into 2 as follows: One plot for each dataset, with 3 subplots in each for the three classifiers. Label the 6 plots appropriately.
# 
# 
# 
# 

# In[4]:


# Cell type : CodeWrite
# write the code for loading the data, running the three algos, and plotting here. 
# (Use the functions written previously.)

import sys
for dataset in [sys.argv[1],sys.argv[2]]:
    with np.load(dataset) as data:
        X_train,Y_train,X_test,Y_test = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
            
        X_mesh,Y_mesh = np.meshgrid(np.arange(X_train[:,0].min() - 0.5, X_train[:,0].max() + 0.5, 0.05), 
                                    np.arange(X_train[:,1].min() - 0.5, X_train[:,1].max() + 0.5, 0.05))
        grids = np.concatenate([X_mesh.reshape(-1,1), Y_mesh.reshape(-1,1)], axis=1)
        
        classifiers = [function_for_A,function_for_B,function_for_C]
        
        plt.figure(0)
        fig, axes = plt.subplots(1,3,sharex=False,sharey=True,figsize=(15,5))
        for i in range(3):
            region,_ = classifiers[i](X_train, Y_train, grids)
            axes[i].contourf(X_mesh,Y_mesh,region.reshape(X_mesh.shape),colors=['lightcoral','palegreen'])
            axes[i].scatter(X_train[Y_train==-1][:,0], X_train[Y_train==-1][:,1],s=2,label='Class -1')
            axes[i].scatter(X_train[Y_train== 1][:,0], X_train[Y_train== 1][:,1],s=2,label='Class 1')
            axes[i].set_title(f'Bayes Classifier {chr(ord("A") + i)} on {dataset}')
            axes[i].set_xlabel('x1')
            axes[i].set_ylabel('x2')
            axes[i].legend()  
        plt.show()
        fig.savefig(f'{dataset}_scatter.png')


# ####4b) Give the ROC Curves for all the classifiers.
# 
# 

# In[5]:


# Cell type : CodeWrite
# write the code for loading the data, running the three algos, and plotting here. 
# (Use the functions written previously.)

for dataset in ['datasetA.npz','datasetB.npz']:
    with np.load(dataset) as data:
        X_train,Y_train,X_test,Y_test = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

        errors = []
        
        plt.figure(0)
        fig, axes = plt.subplots(1,3,sharex=False,sharey=True,figsize=(15,5))
        for i in range(3):
            
            pred, prob = classifiers[i](X_train, Y_train, X_test)
            diff = pred - Y_test
            errors.append(np.count_nonzero(diff!=0)/X_test.shape[0])

            threshold = np.arange(-0.1,1.1,0.0001)
            fpr = []
            tpr = []
            for thresh in threshold:
                
                pred = classify_using_prob(prob,thresh)
                
                diff = pred - Y_test
                fp = np.count_nonzero(diff == 2)
                fpr.append(fp/np.count_nonzero(Y_test == -1))
                tp = np.count_nonzero(pred == 1) - fp
                tpr.append(tp/np.count_nonzero(Y_test == 1))
                
            axes[i].plot(fpr,tpr)
            axes[i].set_title(f'ROC curve for Classifier {chr(ord("A") + i)} on {dataset}')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
        plt.show()
        fig.savefig(f'{dataset}_roc.png')
        
#     print(errors)


# ####4c) In the next Textwrite cell, give the error rate of the three classifiers on the three datasets as 3x2 table, with appropriately named rows and columns.
# 
# 

# **Cell type : TextWrite**
# (Write your observations and table of errors here)
# 
# |         | Dataset A | Dataset B |
# |---------|-----------|-----------|
# | Bayes A | 0.066     | 0.5085    |
# | Bayes B | 0.0675    | 0.504     |
# | Bayes C | 0.0335    | 0.0745    |

# ####4d) In the next Textwrite cell, summarise your observations regarding the 6 learnt classifiers.

# **Cell type : TextWrite**
# (Write your observations and table of errors here)
# 
