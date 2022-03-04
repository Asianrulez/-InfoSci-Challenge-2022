import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

packetcsv = pd.read_csv(r"C:\Users\georg\Downloads\iscxIDS2012_subset.csv")
df = pd.DataFrame(packetcsv)

le = preprocessing.LabelEncoder()

df = df[["Tag","appName"'', "direction", "totalSourceBytes", "sourceTCPFlagsDescription", "totalSourcePackets"]] 

df["Tag"] = le.fit_transform(df["Tag"])
df["appName"] = le.fit_transform(df["appName"])
df["direction"] = le.fit_transform(df["direction"])
df["sourceTCPFlagsDescription"] = le.fit_transform(df["sourceTCPFlagsDescription"])
df["totalSourceBytes"] = le.fit_transform(df["totalSourceBytes"])
df["totalSourcePackets"] = le.fit_transform(df["totalSourcePackets"])

features = list(df.columns[1:])
y = df["Tag"]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy: " ,metrics.accuracy_score(y_test, y_pred))

col = ["appName", "direction", "totalSourceBytes", "totalSourcePackets", "sourceTCPFlagsDescription"]
y = model.feature_importances_
#plot
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False) 

plt.title("Feature importance ")
plt.xlabel("Relative importance")
plt.ylabel("Feature") 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
