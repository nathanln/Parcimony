import numpy as np

#Implementation of existing approaches

def strong_rule(X,θ1star, λ1, λ2, tol=10^(-9)):
    """
    Strong rule algorithm.

    Implements the strong rule from the article R. Tibshirani, J. Bien, J. H. Friedman, 
    T. Hastie, N. Simon, J. Taylor, and R. J.Tibshirani. Strong rules for discarding predictors 
    in lasso-type problems. Journal of the Royal Statistical Society: Series B, 74:245–266, 2012.
    It is an heuristic rule which try to eliminate features that are guaranteed to have zero coefficients 
    in Lasso problem. It might mistakenly discard active features. The method estimates an upper bound of 
    |<x_j, theta_before>| for all the features j in [1,...,j].

    Parameters
    ----------
    X : np.ndarray of shape (n,p)
        Input design matrix. n is the number of elements 
        and p the number of features
    θ1star : np.ndarray of shape (p)
        The vector containing the model coefficients of the LASSO model 
        defined on X for the design matrix and lamb_before for the regularization 
        parameter
    λ1 : float
        The regularization parameter of the precedent model
    λ2 : float
        The regularization parameter for the new model
    tol : float (default is 10e-9)
        Tolerance for the inequality

    Return
    ------
    np.ndarray of shape (p):
        The number of the features j if the scalar product |<x_j, theta_before>| < 1 for all the features j in [1,...,j]
    """
    return np.where(((λ1/λ2)*np.abs(np.dot(X.T,θ1star)) + ((λ1/λ2) -1))<1-tol)

def safe_screening(X,y,θ1star, λ1, tol=10^(-9)):
    """
    Safe screening method.

    Implements the screening method from the article L. Ghaoui, V. Viallon, and T. Rabbani. 
    Safe feature elimination in sparse supervised learning. Pacific Journal of Optimization, 
    8:667–698, 2012. It is an exact rule which try to eliminate features that are guaranteed 
    to have zero coefficients in Lasso problem. The method estimates an upper bound 
    of |<x_j, theta_before>| for all the features j in [1,...,j].

    Parameters
    ----------
    X : np.ndarray of shape (n,p)
        Input design matrix. n is the number of elements 
        and p the number of features
    y : np.ndarray of shape (n)
        The response vector
    θ1star : np.ndarray of shape (p)
        The vector containing the model coefficients of the LASSO model 
        defined on X for the design matrix and lamb_before for the regularization 
        parameter
    λ1 : float
        The regularization parameter of the precedent model
    tol : float (default is 10e-9)
        Tolerance for the inequality

    Return
    ------
    np.ndarray of shape (p):
        The number of the features j if the scalar product |<x_j, theta_before>| < 1 for all the features j in [1,...,j]
    """

    s_star = np.max([np.min([np.dot(θ1star.T,y)/(λ1*np.linalg.norm(θ1star)),1]),-1])
    return np.where((np.abs(np.dot(X.T,y))/λ1 + np.linalg.norm(X,axis=0)*np.linalg.norm(s_star*θ1star-y/λ1))<1-tol)

# Implementation of the Sasvi method

def get_parameter(y , λ, θ1star):
    """
    Computation of a or b according to λ.

    Returns the parameter a or b needeed for the Sasvi Method

    Parameters
    ----------
    y : np.ndarray of shape (n)
        The response vector
    λ : float
        A regularization parameter.
    θ1star : np.ndarray of shape (p)
        The vector containing the model coefficients of the LASSO model 
        defined on X for the design matrix and lamb_before for the regularization 
        parameter

    Return
    ------
    float :
        The parameter a or b.
    """
    return(-θ1star+y/λ)

def x_ort(X,a):
    """
    Orthogonal projections of X onto the null space of a.

    Parameters
    ----------
    X : np.ndarray of shape (n,p)
        Input design matrix. n is the number of elements 
        and p the number of features    
    a : float
        Parameter

    Return
    ------
    np.ndarray of shape (n,p) :
        the orthogonal projections of X onto the null space of a.
    """
    return(X-np.outer(a,np.dot(X.T,a))/(np.linalg.norm(a)**2))

def y_ort(y,a):
    """
    Orthogonal projections of y onto the null space of a.

    Parameters
    ----------
    y : np.ndarray of shape (n)
        The response vector  
    a : float
        Parameter

    Return
    ------
    np.ndarray of shape (n) :
        the orthogonal projections of y onto the null space of a.
    """
    return(y-a*np.dot(y,a)/(np.linalg.norm(a)**2))

def sasvi(X,a,b,λ1,λ2, θ1star, Xort, yort, tol=10^(-9)):
    """
    Sasvi screening method

    Implements the screening method from the article Jun Liu et al. “Safe Screening with Variational Inequalities 
    and Its Application to Lasso”. In : (2014). url : https://arxiv.org/pdf/1307.7577v3.pdf. It is an exact rule which 
    try to eliminate features that are guaranteed to have zero coefficients in Lasso problem. The method estimates an 
    upper bound of |<x_j, theta_before>| for all the features j in [1,...,j].

    Parameters
    ----------
    X : np.ndarray of shape (n,p)
        Input design matrix. n is the number of elements 
        and p the number of features
    a : float
        Parameter from the article
    b : float 
        Parameter from the article
    λ1 : float
        The regularization parameter of the precedent model
    λ2 : float
        The regularization parameter for the new model
    θ1star : np.ndarray of shape (p)
        The vector containing the model coefficients of the LASSO model 
        defined on X for the design matrix and lamb_before for the regularization 
        parameter
    Xort : np.ndarray of shape (n,p)
        Orthogonal matrix of X
    yort : np.ndarray of shape (n)
        Vector orthogonal to the response vector y
    tol : float (default is 10e-9)
        Tolerance for the inequality

    Return
    ------
    np.ndarray of shape (p):
        The number of the features j if the scalar product |<x_j, theta_before>| < 1 for all the features j in [1,...,j]
    """

    # Computation of algebra operations
    normX = np.linalg.norm(X,axis=0)
    normb = np.linalg.norm(b)
    Xtb = np.dot(X.T,b)
    Xttheta = np.dot(X.T,θ1star)

    #Case 1
    if((np.linalg.norm(a)/float(a.size)) < tol):
        
        # Computation of algebra operations
        Xta = np.dot(X.T,a)
        ba = np.dot(b,a)
        norma = np.linalg.norm(a)
        normXort = np.linalg.norm(Xort,axis=0)
        normyort = np.linalg.norm(yort)
        Xorttyort = np.dot(Xort.T,yort)

        bool_vec=(np.any(a)!=0 and ba/(normb*norma)>np.abs(Xta)/normX*norma)
        uplus=bool_vec*Xttheta+(1/λ2-1/λ1)/2*(normXort*normyort+Xorttyort)
        uminus=bool_vec*(-Xttheta)+(1/λ2-1/λ1)/2*(normXort*normyort-Xorttyort)

        bool_vec=(((Xta >0) & (ba/(normb*norma)<=Xta/normX*norma)))
        uminus+=bool_vec*(-Xttheta+1/2*(normX*normb-Xtb))

        bool_vec=(((Xta<0) & (ba/(normb*norma)<=-Xta/normX*norma)))
        uplus+=bool_vec*(Xttheta+1/2*(normX*normb+Xtb))
    #Case 2
    else:
        uminus=(-Xttheta+1/2*(normX*normb-Xtb))
        uplus=(Xttheta+1/2*(normX*normb+Xtb))
        
    return(np.where((uplus<1-tol) & (uminus<1-tol)))

