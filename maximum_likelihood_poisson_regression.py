#!/usr/bin/env python
# coding: utf-8

# # Maximum Likelihood and Poisson Regression
# 
# * This project includes some tests to help you make sure that your implementation is correct.  When you see a cell with `assert` commands, these are tests.
# 
# * Once you have completed the project, delete the cells which have these `assert` commands.  You will not need them.
# 
# * When you are done and have answered all the questions, convert this notebook to a .py file using `File > Download as > Python (.py)`.  Name your submission `project2.py` and submit it to OWL.
# 

# In[2067]:


#It's dangerous to go alone.  Take these!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import linregress
from scipy.special import gammaln
from scipy.stats import poisson
import pytest
import statsmodels.api as sm


# ## Maximum Likelihood
# 
# The poisson distribution https://en.wikipedia.org/wiki/Poisson_distribution is a discrete probability distribution often used to describe count-based data, like how many snowflakes fall in a day.
# 
# If we have count data $y$ that are influenced by a covariate or feature $x$, we can used the maximum likelihood principle to develop a regression model relating $x$ to $y$.
# 
# ### Question 1
# Write a function called `poissonNegLogLikelihood` that takes a count `y` and a parameter `lam` and produces the negative log likelihood of `y` assuming that it was generated by a Poisson distribution with parameter `lam`. You may also want to use `scipy.misc.gammaln` to compute the log of a factorial.  The Gamma Function, $\Gamma(x)$, is a sort of generalized factorial, and `gammaln` efficiently computes the natural log of the Gamma Function.
# 
# It is worth noting that $\Gamma(x) \neq x!$ so it might be worth your while to do a little reading on the gamma function before implementing this function.

# In[2176]:


def poissonNegLogLikelihood(lam, y):
    """
    Computes the negative log-likelihood for a Poisson random variable.

    Inputs:
    lam - float or array.  Parameter for the poisson distribution.
    y - float or array.  Observed data.

    Outputs:
    log_lik - float.  The negative log-likelihood for the data (y) with parameter (lam).

    """
    ### BEGIN SOLUTION
    
    log_lik = - (np.sum(y * np.log(lam) - lam - gammaln((y+1))))
    
    return log_lik

    ### END SOLUTION


# ### Tests
# 
# One of the comments we recieved from the first project is that there were few obvious ways to check if the function you've written is working correctly.  This is an approach to address those concerns!
# 
# Now that you've written the `poissonNegLogLikelihood`, let's test to see if it works.  Below are 3 `assert` statements.  They return nothing if your implementation does what it is supposed to do and raise an error if there is an error.  If you can run this cell and don't see an `AssertionError`, your implementation produces numbers when it's supposed to and does not produce numbers when it isn't supposed to (the tests say nothing about whether the numbers are correct.)

# Here are some more tests that check the values your function produces.  These are graded tests, but you can see what the answer should be ahead of time.  There are also some tests that are not shown to you at the moment.  If your `poissonNegLogLikelihood` is working, then this cell should run without any errors.
# 
# Once you are satisfied with your implementation of `poissonLogLikelihood`, please delete the cell below.

# ### Question 2
# 
# Write a function called `poissonMLE` which accepts as it's first argument an array of data `data` and returns the maximum likelihood estimate for a poisson distribution $\lambda$.  You should use `scipy.optimize.minimize`. You don't have to calculate gradients for this example.

# In[2179]:


def poisson_mle(data):
    """
    Compute the maximum likelihood estimate (mle) for a poisson distribution given data.

    Inputs:
    data - float or array.  Observed data.

    Outputs:
    lambda_mle - float.  The mle for poisson distribution.
    """
    ### BEGIN SOLUTION
    
    lambda_mle = minimize(poissonNegLogLikelihood, 1, args=(data))
    return float(lambda_mle.x)
    
    ### END SOLUTION


# Again, here are some tests for `poisson_mle`.  If you pass these tests, then that should mean your function is working.  You can also write your own tests to make sure the function is working.  Once you are happy with your implmentation, please delete the cell below.

# ### Question 3
# 
# Write a function called `poissonRegressionLogLikelihood` that takes a vector $\mathbf{y}$ of counts, a design matrix $\mathbf{X}$ of features for each count (including a column of 1s for the intercept), and a vector $\mathbf{b}$ of parameters. The function should compute the likelihood of this dataset, assuming that each $y$ is independently distributed with a poisson distribution with parameter $\lambda = exp(X\beta)$.  That is to say, your function should work in the general case for $n$ obervations and $p$ parameters.
# 
# Hint: You can use `poissonNegLogLikelihood` in this answer!

# In[2181]:


def poissonRegressionNegLogLikelihood(b, X, y):
    """
    Computes the negative log-likelihood for a poisson regression.

    Inputs:
    b - array.  Coefficients for the poisson regression
    X - array.  Design matrix.
    y - array.  Observed outcomes.

    Outputs:
    log_lik - float.  Negative log likelihood for the poisson regression with coefficients b.

    """
    ### BEGIN SOLUTION
         
    lam = np.array([np.exp(X@b)]).reshape(-1,1)
    log_lik = poissonNegLogLikelihood(lam, y)   
    
    return log_lik
    
    ### END SOLUTION


# ### Question 4
# 
# In `poissonRegressionNegLogLikelihood`, why did we apply the exponential function to the linear predictor?  What might have happened had we just passed $\lambda =X\beta$?  Enter your answer below in markdown.

# We applied the exponential function to the linear predictor to keep the parameter non-negative even when the regressors in the design matrix X or the regressor coefficiens in Beta have negative values. Its essential that we have a non-negative parameter, because to calculate the negative log-likelihood for a Poisson random variable we take the natural logarithm of the parameter. A natural logarithm of a negative number is not mathematically defined. Therefore, if we have not used the exponential function our implementation will not work as the function used to calculate the negative log-likelihood for a Poisson random variable will return an error or no result. 

# ### Question 5
# 
# Write a function called `fitPoissonRegression` which takes as its first argument data `x` and as its second argument outcomes `y` and returns the coefficients for a poisson regression.

# In[2183]:


def fitPoissonRegression(X, y):
    """
    Fits a poisson regression given data and outcomes.

    Inputs:
    X - array.  Design matrix
    y - array.  Observed outcomes

    Outputs:
    betas_est - array.  Coefficients which maximize the negative log-liklihood.
    """
    ### BEGIN SOLUTION
    
    nrows,ncols = X.shape
    betas=np.zeros((ncols,1))  
    RES = minimize(poissonRegressionNegLogLikelihood, betas, args=(X,y))
    betas_est = RES.x
        
    return betas_est
    
    ### END SOLUTION


# ### Question 6
# 
# Write a function called `makePoissonRegressionPlot` which loads in the data from `poisson_regression_data.csv`, plots a scatterplot of the data, fits a poisson regression to this data, plots the model predictions over $x \in [-2,2]$, and then saves the plot under the file name `poisson_regression.png`.  

# In[2185]:


def makePoissonRegressionPlot():
    ### BEGIN SOLUTION
    
    data = pd.read_csv('poisson_regression_data.csv')
    fig, ax = plt.subplots(dpi = 120)
    data.plot.scatter(x = 'x', 
                         y = 'y', 
                        alpha = 0.5,
                         ax = ax)

    
    #Defining X and y. 
    X = np.array(data['x']).reshape(-1,1)
    X = np.insert(X,0,1,axis=1)
    y = np.array(data['y']).reshape(-1,1)
    
    #Defining the x interval of -2 to 2. 
    x_subset = np.arange(-2,2,0.1).reshape(-1,1)
    x_subset = np.insert(x_subset,0,1,axis=1)
    
    #Getting the values of beta
    actual_betas = fitPoissonRegression(X, y).reshape(-1,1)

    #Predicting values. 
    predicted_y = np.exp(x_subset@actual_betas.reshape(-1,1))
    
    #Defining x_subset coordinates
    x_subset_coordinates = np.delete(x_subset,0,axis=1)

    #Plotting the poisson regression to the data.
    ax.plot(x_subset_coordinates, predicted_y, color='red')

    #Saving the plot in a png file. 
    plt.savefig('poisson_regression.png')
    
    ### END SOLUTION
    # This function does not have to return anything
    return None


# ### Question 7
# 
# Write a function called `makeLinearRegressionPlot`  which loads in the data from `poisson_regression_data.csv`, plots a scatterplot of the data, fits a linear regression to this data, plots the model predictions over $x \in [-2,2]$, and then saves the plot under the file name `linear_regression.png`. 

# In[2187]:


def makeLinearRegressionPlot():
    
    ### BEGIN SOLUTION
    data = pd.read_csv('poisson_regression_data.csv')
    fig2, ax = plt.subplots(dpi = 120)
    data.plot.scatter(x = 'x', 
                         y = 'y', 
                        alpha = 0.5,
                         ax = ax)

    #Defining X and y. 
    X = np.array(data['x']).reshape(-1,1)
    X = np.insert(X,0,1,axis=1)
    y = np.array(data['y']).reshape(-1,1)
    
    #Defining the x interval of -2 to 2. 
    x_subset = np.arange(-2,2,0.1).reshape(-1,1)
    x_subset = np.insert(x_subset,0,1,axis=1)
    

    #Fitting a linear regression to the data. 
    ols_fit = sm.OLS(y,X).fit()
    
    
    #Finding the predicted values. 
    predicted_values_of_OLS_model = ols_fit.predict(x_subset)

        
    #Defining x_subset coordinates
    x_subset_coordinates = np.delete(x_subset,0,axis=1)

    
    #Plotting the linear regression to the data.
    ax.plot(x_subset_coordinates, predicted_values_of_OLS_model, color = 'black')
    
    #Saving the plot in a png file. 
    plt.savefig('linear_regression.png')
        
    ### END SOLUTION
    # This function does not have to return anything
    return None


# ### Question 8
# 
# 
# 1) Explain in 2 or 3 sentences why the coefficients from OLS are different from those from Poisson regression.
# 
# 2) Explain in 2 or 3 sentences why the predicted mean counts are different. Do you see any major problems with the predictions from OLS?
# 
# Provide your answer below in markdown
# 
# 
# 

# Your answer here: 
# 
# The coefficients for the OLS model are chosen so that the sum of squared erros are minimized, while the coefficients for the Poisson regression are chosen to maximize the maximum likelihood estimate. Additionally, the estimated coefficients of the Poisson regression are not dependent on the assumption that the mean is equal to the variance. 
# 
# 
# The predicted mean counts are different because the OLS model predict the outcome in a way that makes it fit on a line, while the Poisson model predicted mean counts follow a curve. A major problem with the OLS model is that for certain values of x_i it can yield negative results. Its nonsense to have a predicted negative count. Also, the assumption of equal variance in linear regression of OLS is violated because the when the mean rate for a Poisson variable increase its variance increase. 
