import pandas as pd 
df = pd.read_csv("Churn_Modelling.csv")

print(df.head())
print(df.shape)  

print(df.info())
print(df.isnull().sum())
print(df["Exited"].value_counts())

# Step 7: Data Cleaning

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnecessary columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

print("\nDataset after cleaning:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 8: Encoding Categorical Data

df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

print("\nAfter Encoding:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

# Step 9: Separate Features and Target

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Step 10: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

 # Step 11: Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling Completed!")

print("Training Scaled Shape:", X_train_scaled.shape)
print("Testing Scaled Shape:", X_test_scaled.shape)

# Step 12: Build Machine Learning Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed!")

# Step 12: Build Machine Learning Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed!")
# Step 14: Save Model and Scaler

import pickle

with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("\nModel and Scaler Saved Successfully!")

# Step 15: Customer Churn Prediction

new_customer = X_test_scaled[0].reshape(1, -1)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nCustomer may leave the bank!")
else:
    print("\nCustomer is likely to stay!")

    with open("feature_names.pkl", "wb") as file:
        pickle.dump(list(X.columns), file) 

