import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
    
# 1. Load the dataset
data = pd.read_csv("penguins.csv")

# 2. Display first 5 rows
print("Dataset:")
print(data.head())

# 3. Remove rows with missing values
data = data.dropna()

# 4. Histogram
data["flipper_length_mm"].hist()
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Number of Penguins")
plt.title("Histogram of Penguin Flipper Length")
plt.show()

# 5. Scatter plot
plt.scatter(data["flipper_length_mm"], data["body_mass_g"])
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.title("Flipper Length vs Body Mass")
plt.show()

# 6. Select input features
X = data[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
]

# 7. Select target
y = data["species"]

# 8. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

# 9. Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# 10. Train model
model.fit(X_train, y_train)

# 11. Predict
y_pred = model.predict(X_test)

# 12. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# 13. Give your own penguin measurements
print("\nEnter penguin measurements:")

bill_length = float(input("Bill length (mm): "))
bill_depth = float(input("Bill depth (mm): "))
flipper_length = float(input("Flipper length (mm): "))
body_mass = float(input("Body mass (g): "))

# 14. Create new penguin data
new_penguin = [[
    bill_length,
    bill_depth,
    flipper_length,
    body_mass
]]

# 15. Predict species
prediction = model.predict(new_penguin)

print("\nPredicted penguin species:", prediction[0])