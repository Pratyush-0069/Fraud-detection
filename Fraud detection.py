# 1. Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading the Dataset
df = pd.read_csv('credit_card_transactions.csv')

# 3. Data Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df['Class'].value_counts())  # Class column: 0 for legit, 1 for fraud

# 4. Data Preprocessing
# Handling missing values (if any)
df = df.dropna()

# Scaling numerical features (like Amount)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Splitting features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 5. Handling Imbalanced Data with SMOTE (Oversampling minority class)
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# 6. Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 7. Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Visualizing Class Distribution (Before and After SMOTE)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(x='Class', data=df, ax=ax[0])
ax[0].set_title('Before Resampling')

sns.countplot(x=y_res, ax=ax[1])
ax[1].set_title('After SMOTE Resampling')

plt.show()
