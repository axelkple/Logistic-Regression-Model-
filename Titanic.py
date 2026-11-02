import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import os
path = r"C:/Users/axelk/Desktop/Intro_AI/assignment_Logistic_Regression"
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
axel = pd.read_csv(fullpath,sep=',')
print("--- First 3 Records ---")
print(axel.head(3))
print("\n--- Shape (Rows, Columns) ---")
print(axel.shape)
print("\n--- Column Names, Types, and Non-Null Counts ---")
axel.info()

print("\n--- unique values for 'Sex' and 'Pclass' ---")
print("Unique values in 'Sex':", axel['Sex'].unique())
print("Unique values in 'Pclass':", axel['Pclass'].unique())




ct_pclass = pd.crosstab(axel['Pclass'], axel['Survived'])
ax1 = ct_pclass.plot(kind='bar', figsize=(8, 5), color=['#ff9999', '#66b3ff'])
plt.title('Titanic Survival Count by Passenger Class - Axel')
plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Number of Passengers')
plt.legend(['Did Not Survive (0)', 'Survived (1)'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Plot B: Survival vs. Gender ---
ct_sex = pd.crosstab(axel['Sex'], axel['Survived'])
ax2 = ct_sex.plot(kind='bar', figsize=(8, 5), color=['#ff9999', '#66b3ff'])
plt.title('Titanic Survival Count by Gender - Axel')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.legend(['Did Not Survive (0)', 'Survived (1)'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


axel['Sex_numeric'] = axel['Sex'].map({'female': 1, 'male': 0})

# Select the requested columns
features = ['Survived', 'Sex_numeric', 'Pclass', 'Fare', 'SibSp', 'Parch']
subset_df = axel[features]

# Generate scatter matrix
plt.figure(figsize=(10, 10))
scatter_matrix(subset_df, alpha=0.3, figsize=(10, 10), diagonal='hist', marker='o')
plt.suptitle("Titanic Feature Relationships Scatter Matrix - Axel", fontsize=14, y=0.92)
plt.tight_layout()
plt.show()



# 1. Drop the four unhelpful columns identified earlier
axel = axel.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# 2 & 3. Using get_dummies to transform categorical variables ('Sex') into numeric columns (yielding 2 columns)
sex_dummies = pd.get_dummies(axel['Sex'])
axel = pd.concat([axel, sex_dummies], axis=1)

# 4. Remove the original categorical variables columns ('Sex' and 'Embarked') using pandas drop method
axel = axel.drop(columns=['Sex', 'Embarked'])

# 5. Replace missing values in 'Age' with the mean of the age column
axel['Age'] = axel['Age'].fillna(axel['Age'].mean())

# 6. Change all column types into float
axel = axel.astype(float)

# 7. Display axel.info() to verify the structure
axel.info()




# 1. Write the normalization function using pandas .min() and .max()
def normalize_df(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

# 2. Call the function passing your transformed dataframe
df_normalized = normalize_df(axel)

# 3. Display the first two records
print(df_normalized.head(2))



# 1. Generate a histogram plot for all variables with a figure size of 9x10 inches
fig, ax = plt.subplots(figsize=(9, 10))
df_normalized.hist(ax=ax, bins=20)

plt.suptitle("Titanic Normalized Variables Histograms - Axel", fontsize=14, y=0.92)
plt.tight_layout()
plt.show()


# Split features into x_firstname (dropping the target column 'Survived')
x_axel = df_normalized.drop(columns=['Survived'])

# Split the target class into y_firstname (keeping only 'Survived')
y_axel = df_normalized['Survived']

print("Features dataframe shape (x_axel):", x_axel.shape)
print("Target dataframe shape (y_axel):", y_axel.shape)



x_train_axel, x_test_axel, y_train_axel, y_test_axel = train_test_split(
    x_axel, 
    y_axel, 
    test_size=0.30, 
    random_state=42
)

# Verify the shapes of the resulting datasets
print("x_train shape:", x_train_axel.shape)
print("x_test shape:", x_test_axel.shape)
print("y_train shape:", y_train_axel.shape)
print("y_test shape:", y_test_axel.shape)


axel_model = LogisticRegression(random_state=42)

# Fit the model using the training data
axel_model.fit(x_train_axel, y_train_axel)

print("Model successfully trained!")
print("Model parameters:", axel_model.get_params())


coef_axel = pd.DataFrame(
    list(zip(x_train_axel.columns, np.transpose(axel_model.coef_))), 
    columns=['Feature', 'Coefficient']
)

# Display the coefficient table
print(coef_axel)


# Loop through test sizes from 0.10 to 0.50 (exclusive of 0.55), stepping by 0.05
for i in np.arange(0.10, 0.51, 0.05):
    # 1. Split data with varying test_size (i)
    # Ensure you use your features (x_axel) and target (y_axel), and your student ID seed (42)
    x_train_loop, x_test_loop, y_train_loop, y_test_loop = train_test_split(
        x_axel, y_axel, test_size=float(i), random_state=42
    )
    
    # Instantiate the model
    model_loop = LogisticRegression(random_state=42)
    
    # 2. Perform 10-fold cross-validation on the training data
    cv_scores = cross_val_score(model_loop, x_train_loop, y_train_loop, cv=10, scoring='accuracy')
    
    # 4. Print out minimum, mean, and maximum accuracy for each run
    test_percentage = int(round(i * 100))
    print(f"Test Size: {test_percentage}% (Train Size: {100 - test_percentage}%) | "
          f"Min Acc: {cv_scores.min():.4f} | "
          f"Mean Acc: {cv_scores.mean():.4f} | "
          f"Max Acc: {cv_scores.max():.4f}")



    # 1. Rebuild the model using 70% - 30% train/test split
x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    x_axel, y_axel, test_size=0.30, random_state=42
)

axel_model = LogisticRegression(random_state=42)
axel_model.fit(x_train_axel, y_train_axel)

# Training Accuracy for comparison later
train_acc = axel_model.score(x_train_axel, y_train_axel)

# 2. Store predicted probabilities for the test set
y_pred_axel = axel_model.predict_proba(x_test_axel)

# ---------------------------------------------------------
# EVALUATION AT THRESHOLD = 0.50
# ---------------------------------------------------------
# 3. Transform probabilities into boolean flags (Threshold 0.5)
y_pred_axel_flag = y_pred_axel[:, 1] > 0.5

print("==================================================")
print("             EVALUATION (THRESHOLD = 0.50)")
print("==================================================")

# 5. Print test accuracy
acc_050 = accuracy_score(y_test_axel, y_pred_axel_flag)
print(f"Test Accuracy (Threshold 0.50): {acc_050:.4f}\n")

# 6. Print confusion matrix
print("Confusion Matrix (Threshold 0.50):")
print(confusion_matrix(y_test_axel, y_pred_axel_flag))
print("\n")

# 7. Print classification report
print("Classification Report (Threshold 0.50):")
print(classification_report(y_test_axel, y_pred_axel_flag))


# ---------------------------------------------------------
# 9. EVALUATION AT THRESHOLD = 0.75
# ---------------------------------------------------------
# Transform probabilities into boolean flags (Threshold 0.75)
y_pred_axel_flag_75 = y_pred_axel[:, 1] > 0.75

print("\n==================================================")
print("             EVALUATION (THRESHOLD = 0.75)")
print("==================================================")

# Print test accuracy
acc_075 = accuracy_score(y_test_axel, y_pred_axel_flag_75)
print(f"Test Accuracy (Threshold 0.75): {acc_075:.4f}\n")

# Print confusion matrix
print("Confusion Matrix (Threshold 0.75):")
print(confusion_matrix(y_test_axel, y_pred_axel_flag_75))
print("\n")

# Print classification report
print("Classification Report (Threshold 0.75):")
print(classification_report(y_test_axel, y_pred_axel_flag_75))