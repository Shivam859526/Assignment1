# 1. Import Packages
import pandas as pd
import numpy as np
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
import joblib
# Import the Logistic Regression model and evaluation metrics from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 2. Read Data
AML_df = pd.read_csv("../data/SAML-D.csv", header = 0, sep = ',')

print("Dataset shape:", AML_df.shape)


# 3. Data Preprocessing
AML_df['Time'] = pd.to_timedelta(AML_df['Time'].astype(str)).dt.total_seconds().astype('int64')

AML_df['Amount'] = pd.to_numeric(AML_df['Amount'], errors='coerce')

AML_df['Date'] = pd.to_datetime(AML_df['Date'], errors='coerce')
AML_df['Date'] = AML_df['Date'].dt.strftime('%Y%m%d').astype('Int64')

# Split The Data Into Train And Test Sets
X = AML_df.drop('Is_laundering', axis = 1)
Y = AML_df['Is_laundering']

print(AML_df.head())

# 5. Data Pre-Processing - Handling All Categorical Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
 
binary_cols = [col for col in AML_df.columns if AML_df[col].dtype == 'str' and
               AML_df[col].nunique()>1]
print(binary_cols)
 
for col in binary_cols:
    # fit() - train label encoder with input samples
    # transform() - apply changes and transform existing dataframe samples.
    AML_df[col] = le.fit_transform(AML_df[col])

# Split The Data Into Train And Test Sets
X = AML_df.drop('Is_laundering', axis = 1)
Y = AML_df['Is_laundering']

# Convert X to a NumPy array with float32 data type
# Convert X to a NumPy array with float32 data type
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="raise").astype("float32")

print(X.dtypes)


# 7. Add SMOTE - Handling Imbalanced Dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=2)
transformed_feature, transformed_label = smote.fit_resample(X, Y)

print("Before SMOTE:", X.shape, Y.shape)
print("After SMOTE:", transformed_feature.shape, transformed_label.shape)
# 9. Split Data into Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(transformed_feature, transformed_label, test_size = 0.2,
                                                    random_state = 2)

 
# 10. Decision Tree Classifier
clf_tree_best = DecisionTreeClassifier(ccp_alpha = 0.001, criterion = 'gini',
                                       max_depth = 20, random_state = 2)
 
clf_tree_best.fit(X_train, Y_train)
 
joblib.dump(clf_tree_best, "./model/decision_tree_model.pkl")
 
print("\nModel Saved Successfully!")

# save test sample as csv for prediction
X_test.to_csv("./data/SAML-D.csv", index=False)