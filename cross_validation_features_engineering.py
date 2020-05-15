#!/usr/bin/env python
# coding: utf-8

# In[198]:


# It's dangerous to go alone.  Take these!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import t
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
pd.set_option('display.max_columns', 500)

get_ipython().run_line_magic('matplotlib', 'inline')


# ### You're A Data Scientist!
# You are working as a Junior Data Scientist for a professional football (er, Soccer) club.  The owner of the team is very interested in seeing how the use of data can help improve the team's peformance, and perhaps win them a championship!
# 
# The draft is coming up soon (thats when you get to pick new players for your team), and the owner has asked you to create a model to help score potential draftees.  The model should look at attributes about the player and predict what their "rating" will be once they start playing professionally.
# 
# The football club's data team has provided you with data for 17,993 footballers from the league.  Your job: work with the Senior Data Scientist to build a model or models, perform model selection, and make predictions on players you have not yet seen.
# 
# ### The Dataset
# 
# The data is stored in a csv file called `footballer_data.csv`.  The data contain 52 columns, including some information about the player, their skills, and their overall measure as an effective footballer.
# 
# Most features relate to the player's abilities in football related skills, such as passing, shooting, dribbling, etc.  Some features are rated on a 1-5 scale (5 being the best), others are rated on 0-100 (100 being the best), and others still are categorical (e.g. work rate is coded as low, medium, or high).
# 
# The target variable (or $y$ variable, if you will) is `overall`.  This is an overall measure of the footballer's skill and is rated from 0 to 100.  The most amazingly skilled footballer would be rated 100, where as I would struggle to score more than a 20. The model(s) you build should use the other features to predict `overall`.
# 
# 
# 
# ### Part A
# 
# Read in the data and take a look at the dataframe.  There should be 52 columns. The outcome of interest is called `overall` which gives an overall measure of player performance. Not all of the other columns are particularly useful for modelling though (for instance, `ID` is just a unique identifier for the player.  This is essentially an arbitrary number and has no bearing on the player's rating).
# 
# The Senior Data Scientist thinks the following columns should be removed:
# 
# * ID
# * club
# * club_logo
# * birth_date
# * flag
# * nationality
# * photo
# * potential
# 
# The Senior Data Scientist would also like the following columns converted into dummy variables:
# 
# * work_rate_att
# * work_rate_def
# * preferred_foot
# 
# Clean the data according to the Senior Data Scientist's instructions.

# In[199]:


model_data = pd.read_csv('footballer_data.csv')

#Dropping unneeded columns 
model_data = model_data.drop(['ID','club','club_logo','birth_date','flag','nationality','photo','potential'], 
                             axis = 'columns')


#Converting categorical columns into dummy variables. 
model_data = pd.get_dummies(model_data, drop_first=True)


# ### Part B
# 
# The data should all be numerical now. Before we begin modelling, it is important to obtain a baseline for the accuracy of our predictive models. Compute the absolute errors resulting if we use the median of the `overall` variable to make predictions. This will serve as our baseline performance. Plot the distribution of the errors and print their mean and standard deviation.

# In[200]:


#Median of the Overall variable 
overall_median = np.median(model_data.overall)

#Calculating absolute errors 
absolute_errors = abs(overall_median - model_data.overall)

#Plot distribution of errors 
sns.distplot(absolute_errors)

#Calculating the absolute errors mean and standard deviation
abs_error_mean = np.mean(absolute_errors)
abs_std = np.std(absolute_errors)

#Print error mean and standard deviation 
print(f"Absolute errors mean: {abs_error_mean}")
print(f"Absolute errors standard deviation: {abs_std}")


# ### Part C
# To prepare the data for modelling, the Senior Data Scientist recomends you use `sklearn.model_selection.train_test_split` to seperate the data into a training set and a test set.
# 
# The Senior Data Scientist would like to estimate the performance of the final selected model to within +/- 0.25 units using mean absolute error as the loss function of choice.  Decide on an appropriate size for the test set, then use `train_test_split` to split the features and target variables into appropriate sets.

# In[202]:


#Determine the size of the test data 
n = round(((2*abs_std)/0.25)**2)

#Defining the X and y
X = model_data.drop('overall', axis = 'columns')
y = model_data.overall

#Splitting the data. 
Xtrainval, Xtest, ytrainval, ytest = train_test_split(X,y, test_size = n, random_state = 0)


# ### Part D
# 
# The Senior Data Scientist wants you to fit a linear regression to the data as a first model.  Use sklearn to build a model pipeline which fits a linear regression to the data. (This will be a very simple, one-step pipeline but we will expand it later.) You can read up on sklearn pipelines [here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Note that the sklearn linear regression adds its own intercept so you don't need to create a column of 1s.

# In[203]:


linear_model = Pipeline([
    ('linear_regression', LinearRegression())
])


# ## Part E
# 
# The senior data scientist wants a report of this model's cross validation score.  Use 5 fold cross validation to estimate the out of sample performance for this model.  You may find sklearn's `cross_val_score` useful.

# In[204]:


cv_scores = cross_val_score(linear_model, 
                           Xtrainval, 
                           ytrainval, 
                           cv = 5, 
                           scoring = make_scorer(mean_absolute_error))

print(cv_scores)
print(f"CV loss: {cv_scores.mean()}")


# ### Part F
# 
# That's impressive!  Your model seems to be very accurate, but now the Senior Data Scientist wants to try and make it more accurate.  Scouts have shared with the Senior Data Scientist that players hit their prime in their late 20s, and as they age they become worse overall.
# 
# The Senior Data Scientist wants to add a quadratic term for age to the model.  Repeat the steps above (creating a pipeline, validating the model, etc) for a model which includes a quadratic term for age.

# In[205]:


#Adding a quadratic term for age
Xtrainval2 = Xtrainval
Xtrainval2 = Xtrainval2.assign(age2 = Xtrainval.age**2)

#Calculating the cv scores
cv_scores = cross_val_score(linear_model, 
                           Xtrainval2, 
                           ytrainval, 
                           cv = 5, 
                           scoring = make_scorer(mean_absolute_error))

print(cv_scores)
print(f"CV loss: {cv_scores.mean()}")


# ### Part G
# 
# 
# The Senior Data Scientist isn't too happy that the quadratic term has not improved the fit of the model much and now wants to include quadratic and interaction term for every feature (That's a total of 1080 features!!!!)
# 
# Add sklearn's `PolynomialFeatures` to your pipeline from part C.  Report the cross validation score.

# In[206]:


linear_model_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear_regression', LinearRegression())
])

#Calculating the cv scores
cv_scores = cross_val_score(linear_model_poly, 
                           Xtrainval, 
                           ytrainval, 
                           cv = 5, 
                           scoring = make_scorer(mean_absolute_error))

print(cv_scores)
print(f"CV loss: {cv_scores.mean()}")


# ### Part H
# 
# The Senior Data Scientist is really happy with the results of adding every interaction into the model and wants to explore third order interactions (that is adding cubic terms to the model).
# 
# This is not a good idea!  Talk them down from the ledge.  Write them an email in the cell below explaining what could happen if you add too may interactions.
# 
# ---

# Hey Boss,
# 
# I got your email about adding cubic terms to the model.  I think we should not add cubic terms to the model, please let me explain my reasoning. 
# 
# If we add cubic terms to the model we will have 17,295 features not including the bias factor. Our original data set holds information for 17,993 footballers. Since we want to estimate the performance of the final selected model to within +/- 0.25 units using mean absolute error as the loss function of choice, we decided to hold out 1162 data entries for our test data set. This leaves us with 16,831 data points for the training and validation data sets. If we add a cubic term, we will have more features than training data points for our model. This means there will be no longer a unique least squares coefficient estimate for our ordinary least squares regression model and the variance will be infinite. 
# 
# Even if we compromise and decrease the size of our test data set to have a bigger training data set, which has data entries at least as big as the features we will have after adding the cubic term, it will still a bad idea to add cubic terms to the model. This is because the number of training data points will be equal to or approaching the number of features of the model. Meaning that there will be high variability in the least square fit leading to overfitting that results in minimal training loss but high generalization error. Consequently, the model will make poor predictions on future data points that were not used in the model training. 
# 
# Sincerely,
# 
# Junior Data Scientist 

# ### Part I
# 
# You've successfully talked the Senior Data Scientist out of adding cubic terms to the model. Good job!
# 
# Based on the cross validation scores, which model would you choose?  Estimate the performance of your chosen model on the test data you held out, and do the following:
# 
# - Compute a point estimate for the generalization error.
# - Compute a confidence interval for the generalization error.  
# - Plot the distribution of the absolute errors.
# 
# Is the test error close to the cross validation error of the model you chose? Why do you think this is the case?
# 

# Based on the cross validation scores, which model would you choose?
# 
# The model with quadratic and interaction terms for every feature has the lowest cross-validation scores, and therefore the smallest validation error. The model with quadratic and interaction terms made the most accurate predictions in comparison with the other models. Consequently, this will be our model of choice. 

# In[208]:


interact_model = linear_model_poly.fit(Xtrainval, ytrainval)
predictions = interact_model.predict(Xtest)

#Calculating the generalization error and its point estimate
absolute_gen_error = abs(predictions - ytest)
point_estimate = np.mean(absolute_gen_error)


#Confidence interval calculations
stderr = absolute_gen_error.std()/np.sqrt(len(absolute_gen_error)) 
critval = t.ppf (0.975,len(absolute_gen_error)-1) 
ci = np.array([point_estimate - critval*stderr, 
           point_estimate + critval*stderr])


print(f"Confidence interval for the the generalization error: {ci}")
print(f"A point estimate for the generalization error: {point_estimate}")


# In[209]:


#Plotting a distribution of the absolute errors
sns.distplot(absolute_gen_error)


# Is the test error close to the cross validation error of the model you chose? Why do you think this is the case?
# 
# Yes, the test error is close to the cross validation error of the chosen model. This is because we used the k-cross vlaidation method, which reduced the downward bias usually associated with using a validation set to choose the model space and evaluate its performance. When the bias is reduced, the cross validation error and the test error become close. 
