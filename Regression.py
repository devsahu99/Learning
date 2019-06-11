
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns


# In[6]:


# load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
#X.drop('CHAS', axis=1, inplace=True)
y = pd.Series(boston.target, name='MEDV')

# inspect data
X.head()


# In[7]:


# print(boston.DESCR)


# In[17]:


df=pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()


# In[20]:


sns.pairplot(pd.concat([df,y],axis=1))


# ### Matrix plot for all variables

# In[9]:


# sns.pairplot(df, size = 4)


# In[10]:


import statsmodels.api as sm

X_constant = sm.add_constant(X)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()


# # Checking Model Assumptions
# ## Linearity of Model

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(lin_reg, y)


# In[12]:


# Mean residuals should be zero
lin_reg.resid.mean()


# ## Check for multi-collinearity

# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=X.columns)


# ## Homoscedasticity (equal variance) of residuals
# 
# ### Potential solutions:
# 1. log transformation of dependent variable
# 2. using ARCH (auto-regressive conditional heteroscedasticity) models to model the error variance. 
# 
# 

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)


fitted_vals = lin_reg.predict()
resids = lin_reg.resid
resids_standardized = lin_reg.get_influence().resid_studentized_internal

fig, ax = plt.subplots(1,2)

sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Residuals vs Fitted', fontsize=16)
ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Scale-Location', fontsize=16)
ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')


# ## No autocorrelation of residuals
# 
# #### To investigate if autocorrelation is present use ACF plots and Durbin-Watson test.
# 
# #### Some notes on the Durbin-Watson test:
# 1. the test statistic always has value between 0 and 4
# 2. value of 2 means that there is no autocorrelation in the sample
# 3. values less than 2 indicate positive autocorrelation, values greater than 2 negative one.

# In[15]:


import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)
acf.show()


# In[16]:


from statsmodels.stats.stattools import durbin_watson
durbin_watson(lin_reg.resid, axis=0)


# # Normality of residuals

# In[30]:


sm.ProbPlot(lin_reg.resid).qqplot(line='s');
plt.title('Q-Q plot');


# In[31]:


from scipy import stats
jb = stats.jarque_bera(lin_reg.resid)
sw = stats.shapiro(lin_reg.resid)
ad = stats.anderson(lin_reg.resid, dist='norm')
    
print(f'Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}')
print(f'Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}')
print(f'Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')


# ## Predict on new dataset

# In[45]:


Xnew = sm.add_constant(X)
ypred = lin_reg.predict(Xnew)
print(ypred)
