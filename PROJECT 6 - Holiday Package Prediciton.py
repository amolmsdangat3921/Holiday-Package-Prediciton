#!/usr/bin/env python
# coding: utf-8

# 1) Problem statement.
# 
# 
# "Trips & Travel.Com" company wants to enable and establish a viable business model to expand the customer base. 
# One of the ways to expand the customer base is to introduce a new offering of packages.
# Currently, there are 5 types of packages the company is offering * Basic, Standard, Deluxe, Super Deluxe, King.
# Looking at the data of the last year, we observed that 18% of the customers purchased the packages.
# However, the marketing cost was quite high because customers were contacted at random without looking at 
# the available information. The company is now planning to launch a new product i.e. Wellness Tourism Package.
# Wellness Tourism is defined as Travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle,
# and support or increase one's sense of well-being. 
# However, this time company wants to harness the available data of existing and potential
# customers to make the marketing expenditure more efficient.

# In[ ]:





# 2) Data Collection.
# 
# 
# https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction

# In[ ]:





# In[1]:


## importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


C:\Users\amolm\OneDrive\Desktop\DATASETS


# In[ ]:





# HANDLING MISSING VALUES
# 
# 1.Handling Missing Values
# 2.Handling Duplicates
# 3.Check data types
# 4.Understad dataset

# In[ ]:





# In[3]:


df = pd.read_csv(r"C:\Users\amolm\OneDrive\Desktop\DATASETS\Travel.csv")
df.head()


# In[ ]:


## Checking the missing values


# In[ ]:





# In[ ]:





# In[5]:


df.isnull()


# In[6]:


df.isna().sum()


# In[17]:


df.info()


# In[7]:


df.isna().sum(axis=1)


# In[8]:


df.isna().sum().sum()


# In[9]:


(df.isna().sum() / len(df)) * 100


# In[ ]:





# In[10]:


## CHECK ALL THE CATEGORIES

df['Gender'].value_counts()


# In[11]:


df['MaritalStatus'].value_counts()


# In[12]:


df['TypeofContact'].value_counts()


# In[ ]:





# In[13]:


df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')


# In[14]:


df.head()


# In[15]:


df['Gender'].value_counts()


# In[16]:


## Check Misssing Values
##these are the features with nan value
features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
for feature in features_with_na:
    print(feature,np.round(df[feature].isnull().mean()*100,5), '% missing values')


# In[18]:


df.info()


# In[20]:


df.describe()


# In[21]:


# statistics on numerical columns (Null cols)
df[features_with_na].select_dtypes(exclude='object').describe()


# In[ ]:





#  IMPUTING NULL VALUES
# 
# 1.Impute Median value for Age column
# 2.Impute Mode for Type of Contract
# 3.Impute Median for Duration of Pitch
# 4.Impute Mode for NumberofFollowup as it is Discrete feature
# 5.Impute Mode for PreferredPropertyStar
# 6.Impute Median for NumberofTrips
# 7.Impute Mode for NumberOfChildrenVisiting
# 8.Impute Median for MonthlyIncome
# 

# In[ ]:





# In[ ]:





# In[22]:


#Age
df.Age.fillna(df.Age.median(), inplace=True)

#TypeofContract
df.TypeofContact.fillna(df.TypeofContact.mode()[0], inplace=True)

#DurationOfPitch
df.DurationOfPitch.fillna(df.DurationOfPitch.median(), inplace=True)

#NumberOfFollowups
df.NumberOfFollowups.fillna(df.NumberOfFollowups.mode()[0], inplace=True)

#PreferredPropertyStar
df.PreferredPropertyStar.fillna(df.PreferredPropertyStar.mode()[0], inplace=True)

#NumberOfTrips
df.NumberOfTrips.fillna(df.NumberOfTrips.median(), inplace=True)

#NumberOfChildrenVisiting
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.mode()[0], inplace=True)

#MonthlyIncome
df.MonthlyIncome.fillna(df.MonthlyIncome.median(), inplace=True)


# In[23]:


df.head()


# In[24]:


df.isnull()


# In[25]:


df.isnull().sum()


# In[ ]:


## CustomerID is not important, so we should drop it.


# In[26]:


df.drop('CustomerID', inplace=True, axis=1)


# In[27]:


df.head()


# # Feature Engineering

# 
# Feature Extraction

# In[28]:


df.head()


# In[29]:


# create new column for feature
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)


# In[30]:


df.head()


# In[ ]:





# In[31]:


## get all the numeric features
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical Features :', len(num_features))


# In[32]:


##categorical features
cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))


# In[33]:


## Discrete features
discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
print('Num of Discrete Features :',len(discrete_features))


# In[34]:


## coontinuous features
continuous_features=[feature for feature in num_features if feature not in discrete_features]
print('Num of Continuous Features :',len(continuous_features))


# In[35]:


df.head()


# Train Test Split And Model Training

# In[36]:


df.head()


# In[ ]:





# In[37]:


from sklearn.model_selection import train_test_split
X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']


# In[38]:


y.value_counts()


# In[39]:


X.value_counts()


# In[40]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# In[41]:


X.info()


# In[42]:


df.head()


# In[ ]:





# In[43]:


# Create Column Transformer with 3 types of transformers
cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
         ("OneHotEncoder", oh_transformer, cat_features),
          ("StandardScaler", numeric_transformer, num_features)
    ]
)


# In[44]:


preprocessor


# In[45]:


## applying Trnsformation in training(fit_transform)
X_train=preprocessor.fit_transform(X_train)


# In[46]:


pd.DataFrame(X_train)


# In[47]:


## apply tansformation on test(transform)
X_test=preprocessor.transform(X_test)


# In[49]:


pd.DataFrame(X_test)


# Random Forest Classifier Training

# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 


# In[ ]:





# In[52]:


models={
    "Logisitic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boost":GradientBoostingClassifier()
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc


    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))



    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))


    print('='*35)
    print('\n')


# In[ ]:





# In[54]:


# Hyperparameter Training
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}


# In[ ]:





# In[55]:


rf_params


# In[56]:


# Models list for Hyperparameter tuning
randomcv_models = [
                   ("RF", RandomForestClassifier(), rf_params)

                   ]


# In[57]:


randomcv_models


# In[ ]:





# In[58]:


from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[ ]:





# In[59]:


models={

    "Random Forest":RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None)
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc


    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))



    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))


    print('='*35)
    print('\n')


# In[60]:


## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Random Forest Classifier',
    'model': RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    'auc':  0.8325
},

]
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train, y_train) # train the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show() 


# In[ ]:




