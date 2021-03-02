import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train_data = pd.read_excel('./Data_Train.xlsx')
# print(train_data.head())
# print(train_data.describe())

train_data.dropna(inplace=True)
# print(train_data.isnull().sum())

train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y', errors='ignore').dt.day
# print(train_data['Jourey_Day'].head(2))

train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y', errors='ignore').dt.month
# print(train_data['Jouney_Month'].head(2))

# print(train_data['Jourey_Day'].head())
train_data.drop(['Date_of_Journey'], axis=1, inplace=True)

# Convert Arrival and departure time into integer into hour minute and minute
#### ARRIVAL
train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
# print(train_data['dep_hour'].head(1))

train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
# print(train_data['dep_minute'].head(1))

#### DEPARTURE
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
# print(df['arrival_hour'].head(1))

train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
# print(df['arrival_minute'].head(1))

# NOW Drop all the unwanted column
train_data.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

print(train_data.dtypes)
print(f"Airline: {train_data['Airline'].unique()}")
print(f"Source: {train_data['Source'].unique()}")
print(f"Destination: {train_data['Destination'].unique()}")
print(f"Duration: {train_data['Duration'].unique()}")
print(f"Total stops: {train_data['Total_Stops'].unique()}")
print(f"Additional Info: {train_data['Additional_Info'].unique()}")

# Time taken by plane to reach destination is called Duration
# differnce betwwen Departure Time and Arrival time
# Assigning and converting Duration column into list
# list_duration=list(train_data['Duration'])
# print(list_duration)


duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]  # Adds 0 hour

duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
    # print(duration_hours)
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

# print(duration)

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
train_data.drop(["Duration"], axis=1, inplace=True)
# train_data.head()
# Replacing Total_Stops
# convert Gender from categorical to numeric
unique_stops = train_data['Total_Stops'].unique()
train_data['Total_Stops'].replace(unique_stops, list(range(1, len(unique_stops) + 1)), inplace=True)
# print(df['Total_Stops'])
# train_data["Airline"].value_counts()

# From graph we can see that Jet Airways Business have the highest Price.
# Apart from the first Airline almost all are having similar median

# Airline vs Price
sns.catplot(y="Price", x="Airline", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6,
            aspect=3)
# plt.show()

# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = train_data[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)
# Airline.head()

train_data["Source"].value_counts()
sns.catplot(y="Price", x="Source", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=4,
            aspect=3)
# plt.show()

Source = train_data[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()

# train_data["Destination"].value_counts()
Destination = train_data[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first=True)
# Destination.head()

# train_data["Route"]

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other

train_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)
train_data["Total_Stops"].value_counts()

# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with corresponding keys

# train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
# train_data.head()

# Concatenate dataframe --> train_data + Airline + Source + Destination

train_data = pd.concat([train_data, Airline, Source, Destination], axis=1)
train_data.head()
train_data.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
# print(data_train.shape)

## Find the correlation
# for column in train_data.columns:
#     correl = np.corrcoef(train_data[column], train_data['Price'])
#     # print(f"Correl {column} and Price: {correl[0][1]}")

# test set
test_data = pd.read_excel(r"./Test_set.xlsx")
test_data.head()
# Preprocessing
print("Test data Info")
print("-" * 75)
print(test_data.info())

print()
print()

print("Null values :")
print("-" * 75)
test_data.dropna(inplace=True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y", errors='ignore').dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format="%d/%m/%Y", errors='ignore').dt.month
test_data.drop(["Date_of_Journey"], axis=1, inplace=True)
# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis=1, inplace=True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis=1, inplace=True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]  # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis=1, inplace=True)

# Categorical data

print("Airline")
print("-" * 75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first=True)

print()

print("Source")
print("-" * 75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first=True)

print()

print("Destination")
print("-" * 75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first=True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)

data_test.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

print()
print()

print("Shape of test data : ", data_test.shape)

# data_test.head()

# data_train.shape
# data_train.columns

x = train_data.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
                       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
                       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                       'Airline_Jet Airways', 'Airline_Jet Airways Business',
                       'Airline_Multiple carriers',
                       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                       'Destination_Kolkata', 'Destination_New Delhi']]
x.head()

y = train_data.iloc[:, 1]
y.head()

# Fitting model using Random Forest

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# find the accuracy scorereg_rf.score(X_test, y_test)
from sklearn.metrics import accuracy_score
# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# plot the Scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

import pickle

# open a file, where you ant to store the data
file = open('flight_model.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)

model = open('flight_model.pkl', 'rb')
model2 = pickle.load(model)
y_prediction = model2.predict(x_test)
metrics.r2_score(y_test, y_prediction)


