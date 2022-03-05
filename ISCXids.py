import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# imports the iscx dataset
packetcsv = pd.read_csv(r"C:\Users\georg\Downloads\iscxIDS2012_subset.csv")
# converts the csv into a dataframe
df = pd.DataFrame(packetcsv)
# instantiates the label encoder object
le = preprocessing.LabelEncoder()
# narrows the columns to the features we want
df = df[["Tag","appName"'', "direction", "totalSourceBytes", "sourceTCPFlagsDescription", "totalSourcePackets"]] 
# encodes the dataframe to be readable by our classifier
df["Tag"] = le.fit_transform(df["Tag"])
df["appName"] = le.fit_transform(df["appName"])
df["direction"] = le.fit_transform(df["direction"])
df["sourceTCPFlagsDescription"] = le.fit_transform(df["sourceTCPFlagsDescription"])
df["totalSourceBytes"] = le.fit_transform(df["totalSourceBytes"])
df["totalSourcePackets"] = le.fit_transform(df["totalSourcePackets"])
# sets the features for the learning model
features = list(df.columns[1:])
# matches "attack" and "normal" with each feature
y = df["Tag"]
X = df[features]
# generates the test set
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.5)
# establishes the number of decision trees
model = RandomForestClassifier(n_estimators=100)
# fits the model to our training set
model.fit(X_train,y_train)
# predicts if the traffic corresponds with an attack
y_pred = model.predict(X_test)

print("Accuracy: " ,metrics.accuracy_score(y_test, y_pred))

# sets the column names
col = ["appName", "direction", "totalSourceBytes", "totalSourcePackets", "sourceTCPFlagsDescription"]
# sets the feature importances as the y axis
y = model.feature_importances_
# creates the plot and sets its variables
fig, ax = plt.subplots() 
width = 0.5  
ind = np.arange(len(y)) 
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False) 
# sets the labels for the plot
plt.title("Network Traffic Feature importance ")
plt.xlabel("Relative importance")
plt.ylabel("Feature") 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
