# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] _cell_guid="10d5cd49-c524-4dd9-8b4d-67497e413466" _uuid="55946cb1-4116-49db-9d6f-d1a448b615a6"
# ## Import Libraries
# Let's import some libraries to get started!

# + _cell_guid="f8b120b2-aecd-4c89-acf8-309de0623000" _uuid="1f38d2a8-b056-4950-9f4c-8b0c5e6c574d"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
# %matplotlib inline

# + [markdown] _cell_guid="ba3d03b5-e508-47c0-8232-ad59809bbaf2" _uuid="057e509e-bb25-40b2-9900-3ed890470bac"
# ## The Data
#

# + _cell_guid="a83e0dca-00a8-4f0c-91ee-8ff4827f18a8" _uuid="2fccc693-f638-4f54-9604-9768991566a2"
train = pd.read_csv('titanic.csv')

# + _cell_guid="2edc27e2-00b2-4f24-b9f4-eee0df2dccd3" _uuid="7269d290-49db-4c9b-b524-152864564846"
train.tail()

# + _cell_guid="2796034a-ff70-4cdd-99ad-3d8c066c25db" _uuid="27c9deb6-52bf-493f-adbe-a53963b1fc34"
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# -

train.isnull().sum().sort_values(ascending=False)

# + _cell_guid="65e26278-c3d2-4e26-9a5b-d1cce06b0ff6" _uuid="16085fa2-70c5-462d-946d-9ed54aa8cd1d"
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

# + _cell_guid="1dde7b2d-a2c8-4cd2-a1da-716776742b43" _uuid="b5916cd2-f3b4-4baf-b4fc-d0ef9bea7c7d"
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

# + _cell_guid="2029bd7d-2415-4055-bdb6-d5e0c6cc480f" _uuid="3c63bdd5-09b9-447d-adbd-f1efe69180b4"
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

# + _cell_guid="5684aa3c-2e6d-4668-ac3a-af2cb4890bf7" _uuid="d6c62306-3ac5-4cef-b97b-bf42581b1da0"
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

# + _cell_guid="e577a7b2-9d02-4b89-b3e1-86ec2992fb79" _uuid="186519f5-c924-4e05-9fb3-6be7d7b7ca59"
sns.countplot(x='SibSp',data=train)
# -

sns.countplot(x='Parch',data=train)

# + _cell_guid="9cfc90ba-5398-4139-864a-bc31a98e243a" _uuid="298860b1-dff8-44ca-b211-2237e3cedb55"
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# + [markdown] _cell_guid="9153792b-c727-4380-8d34-939686991bfb" _uuid="75563284-73ef-47c7-9542-f81075ed0c61"
# ___
# ## Data Cleaning
#

# + _cell_guid="12d1841d-d955-47f4-a003-79fe4dbbc831" _uuid="d23506c6-ff8b-4d03-ad3b-be2911571905"
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# + _cell_guid="ca213e42-43e3-4f69-bb6c-d3358d9a8934" _uuid="500a0190-6ce3-471a-9033-669e8996ef21"
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# + [markdown] _cell_guid="0b76ab42-3f75-408b-93dc-e5bd9d846e7f" _uuid="275055d1-38b2-4e06-afe1-5a1dd5e7a510"
# Now apply that function!

# + _cell_guid="8f711705-a3d8-46ad-bc16-7999bf470e2b" _uuid="f0971c93-45d0-4838-9bdd-d01918c4c69e"
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# -

train['Embarked'] = train['Embarked'].fillna('S')


# + _cell_guid="74b72a12-7335-4780-ae3c-b3ec3cdd6e2a" _uuid="4d541a02-d5f1-48f2-b5b6-2e0505f2ae17"
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# + _cell_guid="40bf2f1b-76d3-4e33-8e96-aea6b13f7ba6" _uuid="2a5ae069-b203-4152-90d5-aa3b214f5935"
train.drop('Cabin',axis=1,inplace=True)

# + _cell_guid="ed680301-6512-4631-b9fe-8533402920b1" _uuid="3eb5a7f2-7a68-42a7-b210-73347fde0118"
train.head()

# + _cell_guid="304e4df6-e578-4010-833f-0f1f93c0ca7b" _uuid="205d9e89-5135-40a9-baab-d13c1036f196"
train.dropna(inplace=True)

# + [markdown] _cell_guid="c4488457-0050-41c9-85a0-8b44d37561bb" _uuid="5a776be5-763f-49b8-9cf8-337ec4d18ec6"
# ## Converting Categorical Features 

# + _cell_guid="41462f98-6728-48d6-b6d8-b9d6e1da827b" _uuid="746d9c20-dbe6-4f3d-9dec-8d789e4e6c1b"
train.info()

# + _cell_guid="21c4dce6-8fa3-4c28-be1e-3178031629ad" _uuid="abc79025-8ad2-4fe2-804e-38517df0ac8a"
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# + _cell_guid="0f873b59-8484-446e-b5ee-4ef369deb427" _uuid="e8a00bfd-89e6-4ae2-b6f1-66d9776e4c5c"
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

# + _cell_guid="986df581-b6f7-4074-ac5f-c6fdb51f13e1" _uuid="701d86a7-2256-48e3-a865-80ce9af6420f"
train = pd.concat([train,sex,embark],axis=1)

# + _cell_guid="4f8d18d6-8170-4bb0-a98b-ee2fd2164e7a" _uuid="0930afba-ea1b-432f-ac55-38bf9454fbc6"
train.head()
# -

# ## Building a Logistic Regression model
#  
#
# ## Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.10, 
                                                    random_state=101)

# ## Training and Predicting

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
X_test.head()

predictions

# ## Evaluation

# We can check precision,recall,f1-score using classification report!

from sklearn.metrics import confusion_matrix , accuracy_score

# +
cm = confusion_matrix(y_test,predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

cm_display.plot()
plt.show()
# -

lr_accuracy = accuracy_score(y_test,predictions)
lr_accuracy

# # Decision Tree Classifiction

from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)

# +
cm = confusion_matrix(y_test,dt_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

cm_display.plot()
plt.show()
# -

dt_accuracy = accuracy_score(y_test,dt_pred)
dt_accuracy



# # Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)

rf_pre=rf.predict(X_test)

cm = confusion_matrix(y_test,rf_pre)

# +
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

cm_display.plot()
plt.show()
# -

rf_accuracy = accuracy_score(y_test,rf_pre)
rf_accuracy

# <h3>Compare the Test Accuracy of the 3 Classification Algorithms:</h3>

# <h3>Based on the Accuracy: </h3>

models_names = ["Logistic Regression","Decision Tree","Random Forest"]
models_scores = [lr_accuracy,dt_accuracy,rf_accuracy]
comp = pd.DataFrame()
comp['name'] = models_names
comp['score'] = models_scores
comp

cm = sns.light_palette("green", as_cmap=True)
s = comp.style.background_gradient(cmap=cm)
s

sns.set(style="whitegrid")
ax = sns.barplot(y="name", x="score", data=comp.sort_values(by="score", ascending=False))
