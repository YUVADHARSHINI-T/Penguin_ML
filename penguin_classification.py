import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
    
#Load the dataset
data = pd.read_csv("penguins.csv")

#Display first 5 rows
print("Dataset:")
print(data.head())

#Remove rows with missing values
data = data.dropna()

#Histogram
data["flipper_length_mm"].hist()
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Number of Penguins")
plt.title("Histogram of Penguin Flipper Length")
plt.show()

#Scatter plot
plt.scatter(data["flipper_length_mm"], data["body_mass_g"])
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.title("Flipper Length vs Body Mass")
plt.show()

#Select input features
X = data[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

#Select target
y = data["species"]

#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

#Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

#Train model
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

#Give your own penguin measurements
print("\nEnter penguin measurements:")

bill_length = float(input("Bill length (mm): "))
bill_depth = float(input("Bill depth (mm): "))
flipper_length = float(input("Flipper length (mm): "))
body_mass = float(input("Body mass (g): "))

#Create new penguin data
new_penguin = [[bill_length,bill_depth,flipper_length,body_mass]]

# 15. Predict species
prediction = model.predict(new_penguin)

print("\nPredicted penguin species:", prediction[0])
