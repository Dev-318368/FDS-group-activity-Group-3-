import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. Load the Dataset
# Replace this filename with the exact name of the CSV you uploaded
file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
file_path = 'Monday-WorkingHours.pcap_ISCX.csv'
file_path = 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
file_path = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
file_path =  'Tuesday-WorkingHours.pcap_ISCX.csv'
file_path = 'Wednesday-workingHours.pcap_ISCX.csv'
df = pd.read_csv(file_path)

# Optional: Sample the dataset to prevent memory crashes in Colab
df = df.sample(n=50000, random_state=42)

# 2. Data Preprocessing (Cleaning & Transformation)
# Clean column names and handle infinite/missing values
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Separate the target label (y) from the dataset
y = df['Label']

# Create features (X) by dropping the Label column
X = df.drop('Label', axis=1)

# --- THE FIX ---
# Keep only numeric columns. This automatically drops 'Flow ID', 'Source IP',
# 'Destination IP', and 'Timestamp' which cause the ValueError.
X = X.select_dtypes(include=[np.number])

# Encode the categorical target labels (e.g., 'BENIGN', 'DDoS') into numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Transform and scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features ready for ML. Shape: {X_scaled.shape}")
# 3. Dimensionality Reduction (Matrix Decompositions)
# PCA (Principal Component Analysis)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print(f"Dataset shape after PCA: {X_pca.shape}")

# SVD (Singular Value Decomposition)
svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X_scaled)
print(f"Dataset shape after SVD: {X_svd.shape}")

# Split the dimensionality-reduced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42)

# 4. Machine Learning Algorithms

# --- Linear Regression ---
# (Note: Used here per syllabus requirements, though it is typically for continuous target variables)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
# Convert continuous regression predictions into binary classes based on a 0.5 threshold
lr_class_predictions = np.where(lr_predictions > 0.5, 1, 0)
print(f"Linear Regression MSE: {mean_squared_error(y_test, lr_predictions):.4f}")
print(f"Linear Regression Estimated Accuracy: {accuracy_score(y_test, lr_class_predictions):.4f}")

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logreg.predict(X_test)):.4f}")

# --- K-Nearest Neighbors (KNN) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"KNN Accuracy: {accuracy_score(y_test, knn.predict(X_test)):.4f}")

# --- Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")


# 1. Save the processed data to a new CSV file in Colab
# Setting index=False prevents pandas from adding an extra column of row numbers
X.to_csv('processed_dataset.csv', index=False)

# Re-attach the original text labels to your clean numerical data
tableau_df = X.copy()
tableau_df['Attack_Label'] = df['Label'].values # Using .values ensures it aligns perfectly

# Verify it worked right here in Colab
print("The last column is:", tableau_df.columns[-1])

# Save and download with a brand new name
tableau_df.to_csv('tableau_dataset_FIXED.csv', index=False)
from google.colab import files
files.download('tableau_dataset_FIXED.csv')