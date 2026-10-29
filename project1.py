import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('train.csv')

df = df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male":0, "female":1})
df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)

#print(df.info())

#print(df.head())

X = df.drop(["Survived"],axis=1)
y=df["Survived"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

score=accuracy_score(y_test,y_pred)

print(score)

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Predict: Die(0)","Predict: Alive(1)"],
            yticklabels=["Real: Die(0)","Real: Alive(1)"]
            )

plt.title("Titanic Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()






