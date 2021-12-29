import numpy as np


def strong_rule(X,y,theta_before, lamb, lamb_before):
    """
    Strong rule algorithm.

    An heuristic rule which try to eliminate features that are guaranteed to have
    zero coefficients in Lasso problem. It might mistakenly discard active features.

    Parameters
    ----------
    X : np.ndarray of shape (n,p)
        Input design matrix. n is the number of elements 
        and p the number of features
    y : np.ndarray of shape (n,1)
        The response vector
    theta_before : np.ndarray of shape (p,1)
        The vector containing the model coefficients of the LASSO model 
        defined on X for the design matrix and lamb_before for the regularization 
        parameter
    lamb : float
        The regularization parameter for the new model
    lamb_before : float
        The regularization parameter of the precedent model

    Return
    ------
    An upper bound of |<x_j, theta_before>| for all the features j in [1,...,j]

    """

    return (lamb_before/lamb)*np.abs(np.dot(X.T,theta_before)) + ((lamb_before/lamb) -1)

def safe_screening(X,y,theta_before, lamb, lamb_before):
    s_star = np.max(np.min(np.dot(theta_before.T,y)/(lamb*np.linalg.norm(theta_before,ord=2)),1),-1)
    return np.abs(np.dot(X.T,y))/lamb + np.linalg.norm(X,ord=2)*np.linalg.norm(s_star*theta_before-y/lamb)

def DPP_screening(X,y,theta_before, lamb, lamb_before):
    pass

n=10
p=100
thetab = np.random.randn(n,1)
X = np.random.randn(n,p)
y = np.random.randn(n,1)
lambd=1
lambd_before=10
# print(strong_rule(X,y,thetab,lambd,lambd_before))
# print(safe_screening(X,y,thetab,lambd,lambd_before))
