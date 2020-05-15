#!/usr/bin/env python
# coding: utf-8

# In[694]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
pd.set_option('display.max_columns', 500)
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# ### You're a Data Scientist...Which is Just a Statistician on a Mac, Right?
# 
# Your models from the last project really impressed some of the management in your football club.  In the spirit of Moneyball (it was a book before it was a movie, I recomend you read it), managers want to test some hypotheses relating a player's overall rating and some of their characteristics in order to make better decisions on what players to trade/sign.
# 
# Management heard somewhere on the internet that statistics and data science are more or less the same thing (the truth of this is the subject of many debates) and would now like you to create some *statistical models* for inference instead of prediction.
# 
# In this project, you're going to take off your "data" hat and put on your "science" hat.
# 
# ### The Dataset
# 
# To test some of the management's hypotheses, the football club has spent some money to go out and collect new data in `footballer_sample.csv`.  The variables are more or less the same from the previous dataset.
# 
# The data contain 52 columns, including some information about the player, their skills, and their overall measure as an effective footballer.
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
# Remember that the Senior Data Scientist from the last project thinks the following columns should be removed:
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
# 
# That still sounds like a pretty good idea.  Remove those columns.  Keep the categorical variables as they are encoded.  Statsmodels will automatically dummy encode them for us

# In[695]:


model_data = pd.read_csv('sampled_footballers.csv')

#Dropping unneeded columns 
model_data = model_data.drop(['ID','club','club_logo','birth_date','flag','nationality','photo','potential'], 
                             axis = 'columns')


# ### Part B
# 
# In statistics, it is useful to *standardize* our data to have mean 0 and standard deviation 1.  This has the effect of putting all the variables on the same scale.  It also has the added benefit of easing the interpretation of the coefficients to the following:
# 
# >Every 1 standard deviation change in the predictor $x$ results in a change of $\beta$ in the outcome.
# 
# Here, $\beta$ is the coefficient from the linear model we fit to the data. Standardize all the numeric variables.  A good way to check that you've done this correctly is to compute the means (which should be close to 0) and the standard deviations (which should be close to 1).

# In[696]:


#The code below will standardize columns in a given dataset if those columns have numeric variables
for column in model_data:
    #Testing whether a column hold numeric variables or not
    numeric_variables = np.issubdtype(model_data[column].dtype, np.number)
    #Check to see if a column contains numeric variables
    if numeric_variables == True:
        #Standardizing the data
        model_data[column] = (model_data[column] - np.mean(model_data[column])) / np.std(model_data[column])


# ### Part C
# 
# One of the things scouts like to disagree upon is how a player changes as they age.  Some insist that players hit their prime in their late 20s and as they reach middle age, they become worse because they can't keep up with younger players.
# 
# Other scouts are certain that the experience a player gains over their tenure makes them more valuable; they can anticipate what will happen on the field much better than a new player.
# 
# You decide that a quadratic term for age in a statistical model might be worth investigating. Write down a statistical model for these competing hypotheses.  What is the null hypothesis? What is the alternative hypothesis?
# 
# 

# Enter your answer in markdown here!
# 
# Null hypothesis($H_{0}$): Football players' overall performance increase in a linear manner as age increase. Therefore, $\beta_{2}$ for the quadratic term of age is equal to zero.
#       
# Alternative hypothesis($H_{1}$): Football players' overall performance increase until their late 20s, then start declining. Therefore, $\beta_{2}$ for the quadratic term of age is less than zero. 

# ### Part D
# 
# Fit your model from Part C.  What can you conclude from the model about the quadratic effect of age?  Answer in terms of the null hypothesis.

# In[697]:


#Saving the quadratic effect of age in a variable
age_quadratic = model_data['age']**2

#Fitting the model with the quadratic effect of age 
ols('overall ~ age + age_quadratic', data = model_data).fit().summary()


#  ### What can you conclude from the model about the quadratic effect of age? Answer in terms of the null hypothesis.
#  
# The quadratic effect of age has a p-value of of less than 0.001. Therefore, using a significance level of 0.005 we reject the null hypothesis. The difference between the $\beta_{2}$ for the quadratic term of age and zero is statistically significant.

# ### Part E
# 
# Management would also like to know how marking (the player's ability to prevent the other team from getting the ball) and interceptions (taking the ball when the opposing team is passing the ball between players) impact a player's overall ranking, controlling for age.
# 
# Those sound awfully similar, don't they?  Fit two models: one model only controls for age (including the quadratic term) and marking, the other controls for age (including the quadratic term), marking, AND interceptions.
# 
# Answer the following:
# 
# * In 1-2 sentecnes, what are the differences between the coefficient for marking in the first and second model?  The size of the coefficient really isn't the issue.  Look at the sign of the coeffieicnt instead.
# 
# * In 1-2 sentences, why is this difference troubling? How does the interpretation of a one standard deviation change in marking change between models?
# 
# * In 1-2 sentences, what might explain this difference? You might want to look at `model_data.corr()`.

# In[699]:


#Fitting and displaying a summary of a model with age, the quadratic effect of age and marking 
ols('overall ~ age + age_quadratic + marking', data = model_data).fit().summary()


# In[700]:


#Fitting and displaying a summary of a model with age, the quadratic effect of age, marking and interceptions
ols('overall ~ age + age_quadratic + marking + interceptions', data = model_data).fit().summary()


# ### what are the differences between the coefficient for marking in the first and second model?
# 
# The first model coefficient for marking is 0.1287, while in the second model the coefficient is -0.2276. In both models, the absolute distance of the coefficients from zero is about the same but the signs of the coefficients are switched. One model suggests that there is a positive correlation between marking and overall ranking, while the other model suggests a negative correlation.  
# 
# ### Why is this difference troubling? How does the interpretation of a one standard deviation change in marking change between models?
# 
# The difference is troubling because it interferes in determining the precise effect of the predictor marking, which we are trying to understand how it impacts the overall ranking of a football player.  
# 
# In the first model the interpretation of a one standard deviation change in marking lead to a 0.1287 increase in the overall ranking, while in the second model this interpretation change to a one standard deviation change in marking lead to a -0.2276 decrease in the overall ranking. 
# 
# 
# ### What might explain this difference?
# 
# Multicollinearity, which is when two or more explanatory variables in a multiple regression model are highly linearly related, can explain this difference. In the cell below we studied the correlation coefficients between the explanatory variables in our model, and it's clear that the variable marking and interceptions are highly correlated.  

# In[701]:


#Displaying correlation coefficients between explanatory variables in our model
model_data[['marking','interceptions','age']].corr()


# ### Part F
# 
# Fit the linear model `overall~ preferred_foot`.  Incredibly, the model says that **RIGHT FOOTED PLAYERS TEND TO BE WORSE AS COMPARED TO LEFT FOOTED PLAYERS**! Scounts don't believe you, this goes against everything they've believed about being left footed.  
# 
# Perform a randomization test on this data.  Perform 1000 randomizations of `preferred_foot`, fit the same model, and record the effects.  Plot a histogram of the effects from the randomized data and use `plt.axvline` to plot a vertical red line to indicate where the observed effect from our data lies.
# 
# Print out the p value (that is, the proportion of the resampled effects are larger than our observed effect in absolute value).

# In[702]:


#Fitting a linear model of overall ~ preferred_foot
ols('overall ~ preferred_foot', data = model_data).fit().summary()


# In[710]:


#The steps below is to perform a Randomization test

#Calculating test statistic from data
fitted_model = ols('overall ~ preferred_foot', data = model_data).fit()
test_statistic = fitted_model.params[1]

#Defining test statistic array
test_statistic_array = np.array([])

#Randomization test
for i in range(1000):
    preferred_foot_randomized = np.random.permutation(model_data['preferred_foot'])
    fitted_model_randomized = ols('overall ~ preferred_foot_randomized', data = model_data).fit()
    test_statistic_randomized = fitted_model_randomized.params[1] 
    test_statistic_array = np.append(test_statistic_array, test_statistic_randomized)

#Plotting a histogram of test statistic
sns.distplot(test_statistic_array)

#Plotting a vertical red line to indicate where the observed effect from our data lies 
plt.axvline(test_statistic, color='red')


# In[711]:


#Calculating the p-value
prop = 0
for i in test_statistic_array:
    if abs(i) > abs(test_statistic):
        prop += 1 
p_val = prop/len(test_statistic_array) 

#Printing the p value
print(f"The p value is: {p_val}")


# ### Part G
# 
# Your findings from the randomization test are incredible; left footed players are on average 2.5 points better than their right footed counterparts!  The management is prepared to spend a lot of money to replace the team full of lefties in order to gain a slight advantage.
# 
# However, you have a sneaking suspicion this isn't the whole story.  Before management replaces the entire team, you decide to take a look at the dataset from your predictive model, called `footballer_data.csv`.  Load that data, clean it up as you did in part A, and perform another regression of overall onto preferred_foot, this time controlling for age (including the quadratic term) and interceptions.  Answer the following in a markdown cell:
# 
# * What is the p-value for the effect of being right footed?  
# 
# * What does that mean in terms of the null hypothesis?
# 

# In[712]:


#Loading the dataset footballer_data.csv
full_data = pd.read_csv('footballer_data.csv')

#Dropping unneeded columns 
full_data = full_data.drop(['ID','club','club_logo','birth_date','flag','nationality','photo','potential'], 
                             axis = 'columns')

#Calculating the quadratic term of age from the footballer_data.csv
age_quad = full_data['age']**2

#Fitting the model of overall onto preferred_foot controlling for age, age quadratic term and interceptions
ols('overall ~ age + age_quad + preferred_foot + interceptions', data = full_data).fit().summary()


# ### What is the p-value for the effect of being right footed?
# 
# The P-value for the effect of being right footed is 0.243. 
# 
# ### What does that mean in terms of the null hypothesis?
# 
# This P-value indicates that if being right-footed has no effect on the overall ranking, we would obtain the observed difference 24.3% of the times due to random sampling error. Therefore, using a significance level of 0.05 we fail to reject the null hypothesis, which states that the effect of being right-footed is equal to zero.  

# ### Part H
# 
# The club owner, Owen Owner, saw the results of your randomization test and is convinced that he should replace the whole team with left-footed players. Using your results from Part G, write an email explaining to him why this isn't a worthwhile endeavour. 
# 
# 

# Dear Owen Owner,
# 
# Thank you for taking the time to look at my statistical analysis. Regarding the club's preference for left-footed players as a result of the randomization test you saw. I need to clarify a few things to explain why this preference isn't justified. 
# 
# To further investigate the effect of the preferred foot on overall performance, I performed regression statistical analysis of overall onto preferred_foot, controlling for age (including the quadratic term) and interceptions using the dataset footballer_data.csv. The results that I got suggest that being right-footed has no effect on the overall ranking. 
# 
# In order to understand why the results we got from the randomization test suggest otherwise, please let me explain a few concepts. The sampling done in the randomization test examine what would the sampling distribution look like if the null hypothesis was true, which assumes that there is no effect of being right-footed on overall ranking. A low p-value could mean that there is a relationship between being right-footed and overall ranking but it also could mean that we just got super lucky and saw an unusual test statistic. We have no strong theoretical justification for conducting a randomization test if randomization was not properly implemented by design in obtaining the data used to compute a p-value and do the randomization test. 
# 
# We can only infer that associations are causal if we randomize properly in the design. The results we got makes me believe that there was not proper randomization and therefore we can't justify our preference for the left-footed players based on the randomization test results. Thank you.  
# 
# Sincerely,
# 
# Junior Data Scientist  
