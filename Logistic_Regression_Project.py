# Import Libraries
import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sb
%matplotlib inline

# Get the Data
# Reading the advertising.csv file and setting it to a data frame called ad_data.
ad_data = pd.read_csv("advertising (1).csv")

# Check the head of ad_data
ad_data.head()

# Use info and describe() on ad_data
ad_data.info()
ad_data.describe()

# Exploratory Data Analysis
# Create a histogram of the Age
sb.set_style("whitegrid")
sb.distplot(ad_data['Age'],kde=False,bins=30,hist_kws={"alpha":0.75})

# Create a jointplot showing Area Income versus Age.
sb.jointplot(x="Age",y="Area Income",data=ad_data)

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age
sb.jointplot(x="Age",y="Daily Time Spent on Site",data=ad_data,kind="kde",color="red")

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sb.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=ad_data,color="green")

# Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sb.pairplot(ad_data,hue="Clicked on Ad")

# Splitting the data into training set and testing set using train_test_split
from sklearn.model_selection import train_test_split
X = ad_data[["Daily Time Spent on Site", 'Age', 'Area Income','Daily Internet Usage', 'Male']]
X_train, X_test, Y_train, Y_test = train_test_split(X,ad_data["Clicked on Ad"],test_size=0.30,random_state=42)

# Train and fit a logistic regression model on the training set.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

# Predict values for the testing data.
prediction = model.predict(X_test)

# Create a classification report for the model.
from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction))
