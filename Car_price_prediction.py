#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression

# Linear Regression helps us to predict the future and the expected variables when we have continous data.Linear Regression predicts with the best fit line.In statistics, linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.
# 
# 
# Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering, and the number of independent variables getting used. There are many names for a regression’s dependent variable.  It may be called an outcome variable, criterion variable, endogenous variable, or regressand.  The independent variables can be called exogenous variables, predictor variables, or regressors.
# Linear regression is used in many different fields, including finance, economics, and psychology, to understand and predict the behavior of a particular variable. For example, in finance, linear regression might be used to understand the relationship between a company’s stock price and its earnings, or to predict the future value of a currency based on its past performance.
# 
# 
# This dataset contains some information about used cars listed on www.cardekho.com
# 
# The dataset was shared by following link:
# 
# https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv

# ### ▶️Step 1) Reading and Understanding the Data

# In[1]:


import warnings
warnings.simplefilter(action = 'ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go

sns.set()
palette = sns.color_palette("rainbow", 8)


# In[3]:


data=pd.read_csv("C:/Users/HP/Documents/backups/Car details v3.csv")
data


# ### step 2)Understanding the structure of the data

# In[4]:


data.head().style.background_gradient(cmap = "autumn")


# In[5]:


print("The number of rows in train data is {0}, and the number of columns in train data is {1}".
      format(data.shape[0], data.shape[1]))


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# ##### Basic info about dataset
# 
# ● There are null values in 'mileage' (221), 'engine' (221), 'max_power' (215), 'torque' (222), 'seats' (221) variables.
# 
# ● The dataset consists of 8128 rows and 13 columns.
# 
# ● Of the variables, 9 are object, 3 are integer and 1 are float

# In[8]:


#make dtypes of some variables 'category'

categoric_columns = ["fuel", "transmission", "owner", "seller_type"]
for column in categoric_columns:
    data[column] = data[column].astype("category")


# In[9]:


data["car_brand_name"] = data["name"].str.extract('([^\s]+)')
data["car_brand_name"] = data["car_brand_name"].astype("category")


# In[10]:


#extract values of 'engine' and 'mileage' variables

data["engine"] = data["engine"].str.extract('([^\s]+)').astype("float")
data["mileage"] = data["mileage"].str.extract('([^\s]+)').astype("float")


# In[11]:


#extract values of 'max_power' variable

data["max_power"] = data["max_power"].str.extract('([^\s]+)')
data["max_power"] = data["max_power"][~(data["max_power"] == "bhp")]
data["max_power"] = data["max_power"].astype("float")


# In[12]:


import datetime

data["car_age"] = (datetime.datetime.now().year) - (data["year"])


# In[13]:


data.drop(["name", "year", "torque"], axis = 1, inplace = True)


# In[14]:


data.describe().T.style.background_gradient(cmap = "viridis")


# #### Description of numeric variables
# ● The oldest car was produced in 1983 (age = 39), and the newest car was produced in 2020 (age = 2)
# 
# ● Minimum selling price is 29999 USD, maximum price is 10000000 USD, and average selling price is 638271 USD
# 
# ● The driving distance of the least driven car is 1 km, the most driven car's driving distance is 2360457 km, average driving distance is 69819 km
# 
# ● The number of seats of cars change from 2 seats to 14 seats
# 
# ● Minimum mileage is 0, maximum mileage is 42, average mileage is 19.4
# 
# ● Engine volume changes from 624 to 3604, average is 1458

# In[15]:


data.describe(include = "category").T


# #### About categoric variables
# ● Car brand name with highest frequency: Maruti (freq = 2448)
# 
# ● Fuel kind with highest frequency: Diesel (freq = 4402)
# 
# ● Seller type with highest frequency: Individual (freq = 6766)
# 
# ● Transmission type with highest frequency: Manual (freq = 7078)
# 
# ● Owner type with highest frequency: First owner (freq = 5289)

# In[16]:


#fill null values with median (numeric) and frequent values (categoric)

numeric_data = [column for column in data.select_dtypes(["int", "float"])]
categoric_data = [column for column in data.select_dtypes(exclude = ["int", "float"])]

for col in numeric_data:
    data[col].fillna(data[col].median(), inplace = True)
        
#replace missing values in each categorical column with the most frequent value
for col in categoric_data:
    data[col].fillna(data[col].value_counts().index[0], inplace = True)


# In[17]:


data.isna().sum()


# In[18]:


data.info()


# ### ▶️Step 3) Exploratory Data Analysis (EDA)

# In[19]:


Categorical = ['fuel','seller_type','transmission','owner']
i=0
while i<4:
    fig = plt.figure(figsize=[10,4])
    plt.subplot(1,2,1)
    sns.countplot(x=Categorical[i], data=data)
    i += 1
    plt.subplot(1,2,2)
    sns.countplot(x=Categorical[i], data=data)
    i += 1
    plt.show()


# In[20]:


sns.heatmap(data.corr(), annot=True, cmap='RdBu')
plt.show()


# In[21]:


plt.figure(figsize = (15, 8))
sns.barplot(x = "fuel", y = "selling_price", hue = "seller_type", data = data, saturation = 1);


# In[22]:


plt.figure(figsize = [5, 5], clear = True, facecolor = "#FFFFFF")
data["owner"].value_counts().plot.pie(explode = [0.1, 0.1, 0.1, 0.1, 0.1], autopct='%1.3f%%', shadow = True);


# In[23]:


owner=data.groupby('owner').sum().reset_index()
plt.pie(data=owner, x=owner['km_driven'], labels=owner['owner'], autopct='%.f%%',rotatelabels=True,colors=palette)
plt.xticks(fontsize=50), plt.yticks(fontsize=50)
plt.tight_layout()
plt.show()


# In[24]:


fig = px.histogram(data, x = "car_age",
                   y = "selling_price",
                   marginal = None, text_auto = True,
                   color = "owner", hover_data  = data.columns, width = 850, height = 500)
fig.show()


# In[25]:


fig = px.histogram(data, x = "seller_type",
                   y = "selling_price",
                   marginal = None, text_auto = True,
                   color = "owner", hover_data  = data.columns, width = 850, height = 500)
fig.show()


# In[26]:


fig = px.histogram(data, x = "owner",
                   y = "selling_price",
                   marginal = None, text_auto = True,
                   color = "owner", hover_data  = data.columns, width = 850, height = 500)
fig.show()


# In[27]:


fig = px.density_heatmap(data, x = "max_power", y = "selling_price", z = "mileage",
                        color_continuous_scale = "deep", text_auto = True,
                        title = "Density heatmap between variables")
fig.show()


# ### 📈Step 4) Training a Linear Regression Model

# ##### 1) Data Preparation
# 
# Choosing the features which we want to put in our model
# Categorical variables are converted to Numerical variables using One Hot (Dummy encoding)
# 🔸In Machine-learning activities, the data set may contain text or categorical values (basically non-numerical values), We need to convert them into Categorical variables for the smooth functioning of our Machine Learning Model. Algorithms usually work better with numerical inputs. So, the main challenge faced by Data scientists is to convert text/categorical data into numerical data and still make an algorithm/model to make sense of it.
# 
# As of now, Mainly Two methods are being used :
# 
# 1) Label Encoder
# 
# 2) One Hot Encoder
# 
# Both of these encoders are part of the SciKit-Learn library used in Python.But for this model we have used get dummies function for dummy encoding
# 
# 
# 

# In[28]:


# Dummy encoding

data_num = data.select_dtypes(exclude='category')
data_obj = data.select_dtypes(include='category')


# In[29]:


data_obj = pd.get_dummies(data_obj, drop_first=True)


# In[30]:


final_df = pd.concat([data_num, data_obj], axis=1)


# In[31]:


cars=final_df.copy()

cars.head(5)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

cars=sc.fit_transform(cars)

cars=pd.DataFrame(cars,columns=final_df.columns)


# ### 📈Step 4) Linear Regression Model Building
# 
# #### Finding the best fit line:
# When working with linear regression, our main goal is to find the best fit line that means the error between predicted values and actual values should be minimized. The best fit line will have the least error.
# The different values for weights or the coefficient of lines (m, c) gives a different line of regression, so we need to calculate the best values for a0 and a1 to find the best fit line, so to calculate this we use cost function.
# 
# The best fit line works with the equation y=mx+c
#                      
#                      where m=slope
#                            x=feature
#                            c=intercpet
# 
# #### Gradient Descent:
# 
#  Gradient descent is a powerful optimization algorithm used to minimize the loss function in a machine learning model. It’s a popular choice for a variety of algorithms, including linear regression, logistic regression, and neural networks. In this article, we’ll cover what gradient descent is, how it works, and several variants of the algorithm that are designed to address different challenges and provide optimizations for different use cases.
#  
# •	Gradient descent is used to minimize the MSE by calculating the gradient of the cost function.
# 
# •	A regression model uses gradient descent to update the coefficients of the line by reducing the cost function.
# 
# •	It is done by a random selection of values of coefficient and then iteratively update the values to reach the minimum cost function.
# 
# •	The iteration stops once we get the best fit line that is when it reaches the saturation.
#      
#                                 Gradient descent(m)=m-(diff(m)/dm)* learning rate(0.001)
# 
# 

# In[33]:


x = cars.drop('selling_price', axis=1)
y = cars['selling_price']


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[35]:


model=LinearRegression()


# In[36]:


model.fit(x_train,y_train)


# In[37]:


import pickle
pickle.dump(model, open('Car_model_new.pkl', 'wb'))


# ### ▶️Model Evaluation
# 
# #### Cross-validation
# ###### What is cross-validation?
# 
# In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.
# 
# ###### When should you use cross-validation?
# 
# Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions. However, it can take longer to run, because it estimates multiple models (one for each fold).
# 
# So, given these tradeoffs, when should you use each approach?
# 
# For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
# For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.
# There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.
# 
# Alternatively, you can run cross-validation and see if the scores for each experiment seem close. If each experiment yields the same results, a single validation set is probably sufficient.

# In[38]:


result = model.score(x_train, y_train)
print('Score:', result)


# In[39]:


y_pred=model.predict(x_test)


# In[40]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[41]:


model_new = LinearRegression()


# In[42]:


kfold_validation = KFold(7, shuffle=True, random_state=0)


# In[43]:


results = cross_val_score (model_new, x, y, cv=kfold_validation)


# In[44]:


print ('Results=', results)
print ('Mean=' ,np.mean(results))


# ### 🧪Regression Evaluation Metrics
# 
# #### Cost function-
# 
# o	Cost function optimizes the regression coefficients or weights. It measures how a linear regression model is performing.
# 
# o	We can use the cost function to find the accuracy of the mapping function, which maps the input variable to the output variable. This mapping function is also known as Hypothesis function.
# 
# 
# #### Here are three common evaluation metrics for regression problems:
# 
# Mean Absolute Error (MAE) is the mean of the absolute value of the errors.It represents the average of the absolute diff between actual and the predicted values in the dataset.
#     
# 
# Mean Squared Error (MSE) is the mean of the squared errors.It represents the average of the squared diff between actual and the predicted values in the dataset.
#     
# 
# Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.
# 
# R-squared represents the proportion of variance in the dependent variable.
# 
# Adjusted R-square is a modified version of R-square and is adjusted for the number of independent variable in the model.
#     
# 

# In[45]:


print ("MAE:", metrics.mean_absolute_error (y_test, y_pred))
print ("MSE:", metrics.mean_squared_error (y_test, y_pred))
print ("RMSE:", np.sqrt(metrics.mean_squared_error (y_test, y_pred)))


# ### ▶️Improve model

# #### ▶️Scaling

# In[46]:


cars2=final_df.copy()


# In[47]:


from sklearn import preprocessing
Scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
cars2 = Scaler.fit_transform (cars2)


# In[48]:


Scaler_2 = preprocessing.StandardScaler()
cars2= Scaler_2.fit_transform (cars2)


# In[49]:


cars2=pd.DataFrame(cars2,columns=final_df.columns)


# In[50]:


x2 = cars2.drop('selling_price', axis=1)
y2 = cars2['selling_price']


# In[51]:


from sklearn.feature_selection import f_regression


# In[52]:


p_value=f_regression(x2,y2)[1].round(3)


# In[53]:


reg_summary=pd.DataFrame(data=x.columns.values,columns=['Feature'])
reg_summary['P_values']=p_value


# In[54]:


reg_summary


# In[55]:


X2=x2.drop(['car_brand_name_Mahindra','car_brand_name_Mitsubishi','car_brand_name_Skoda','car_brand_name_Ashok'],axis=1)


# In[56]:


x2_train,x2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=100)


# In[57]:


model_2 = LinearRegression()


# In[58]:


model_2.fit(x2_train,y2_train)


# In[59]:


y2_pred=model_2.predict(x2_test)


# In[60]:


result = model_2.score(x2_train, y2_train)
print('Score:', result)


# In[61]:


print ("MAE:", metrics.mean_absolute_error (y2_test, y2_pred))
print ("MSE:", metrics.mean_squared_error (y2_test, y2_pred))
print ("RMSE:", np.sqrt(metrics.mean_squared_error (y2_test, y2_pred)))

