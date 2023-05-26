import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocess import Preprocess
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import accuracy_score
from Classification import Classification

df = pd.read_csv('CustomersDataset.csv')
pre = Preprocess(df)  # Assuming Preprocess is the class name

dataset = pre.CleanData(df)  # Pass df as an argument to CleanData method
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(dataset.head())

########################################################################################################################

# Calculate basic statistics
print("Some Statistics about The Data")
print(df.describe())
print(df.info())
std_deviation = df[['tenure', 'MonthlyCharges', 'TotalCharges']].std()
print("Standard Deviation:")
print(std_deviation)

# Calculate correlations
correlations = df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()
print("Correlations:")
print(correlations)

########################################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns

c = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Box plots for selected columns
plt.style.use('seaborn-darkgrid')
for column in c:
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(df[column].values, patch_artist=True, medianprops={'color': 'black'}, vert=False)
    ax.set_title(f'Box Plot of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('')
    outliers = bp['fliers'][0].get_xdata()
    ax.scatter(outliers, [1] * len(outliers), c='red', marker='o', alpha=0.5)
    plt.show()

# Box plot of all data columns
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(dataset.values, patch_artist=True, medianprops={'color': 'black'}, vert=False)
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_yticklabels(dataset.columns)
ax.set_title('Box Plot of All Data')
plt.show()

# Histogram of tenure distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='tenure', bins=20, kde=True)
plt.xlabel('Tenure')
plt.ylabel('Count')
plt.title('Distribution of Tenure')
plt.show()

# Histogram of TotalCharges distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='TotalCharges', bins=20, kde=True)
plt.xlabel('TotalCharges')
plt.ylabel('Count')
plt.title('Distribution of TotalCharges')
plt.show()

# Histogram of monthly charges distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='MonthlyCharges', kde=True)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.show()

# Count plot of churned vs. non-churned customers
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Churned vs. Non-Churned Customers')
plt.show()

# Count plot of churn by contract type
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn', hue='Contract')
plt.title('Churn by Contract Type')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Count plot of churn by payment method
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn', hue='PaymentMethod')
plt.title('Churn by Payment Method')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Count plot of churn by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn', hue='gender')
plt.title('Churn by Gender')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Count plot of churn by internet service
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn', hue='InternetService')
plt.title('Churn by Internet Service')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Pie chart of internet service types
plt.figure(figsize=(8, 6))
df['InternetService'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Internet Service Types')
plt.ylabel('')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[c].corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Pairwise scatter plot
sns.pairplot(data=df, vars=['tenure', 'MonthlyCharges', 'TotalCharges'], hue='Churn')
plt.show()

# Scatter plot for tenure, MonthlyCharges, and TotalCharges
c = ['tenure', 'MonthlyCharges', 'TotalCharges']
plt.figure(figsize=(10, 8))
for i, column in enumerate(c):
    plt.subplot(2, 2, i+1)
    plt.scatter(df[column], df['Churn'], alpha=0.5)
    plt.xlabel(column)
    plt.ylabel('Churn')
    plt.title(f'Scatter Plot: {column} vs. Churn')
plt.tight_layout()
plt.show()

########################################################################################################################

# Feature Selection
X = dataset.drop('Churn', axis=1)
y = dataset['Churn']
selector = SelectKBest(f_classif, k=19)  # Specify the value of k
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]

print("Selected Columns:")
print(selected_columns)

# Models

def calcAccuracy(modelPred, yTest , modelName):

    score_model = accuracy_score(yTest, modelPred)
    print(f"accuracy score of {modelName} model : {score_model}")

X_train, X_test, y_train, y_test = train_test_split(X[selected_columns], y, test_size=0.2, random_state=42)
classification = Classification(X_train, y_train)
predictions = classification.Train()
print("############ Train ##############")
calcAccuracy(predictions[0], y_train, "RandomForestClassifier")
calcAccuracy(predictions[1], y_train, "KNeighborsClassifier")
calcAccuracy(predictions[2], y_train, "GradientBoostingClassifier")
calcAccuracy(predictions[3], y_train, "DecisionTreeClassifier")
calcAccuracy(predictions[4], y_train, "LogisticRegression")
calcAccuracy(predictions[5], y_train, "SupportVectorMachineClassifier")
calcAccuracy(predictions[6], y_train, "NaiveBayesClassifier")
calcAccuracy(predictions[7], y_train, "AdaBoostClassifier")
calcAccuracy(predictions[8], y_train, "LightGBMClassifier")

print("############ Test ##############")
classificationTest = Classification(X_test, y_test)
Test = classificationTest.Test()
calcAccuracy(Test[0], y_test, "RandomForestClassifier")
calcAccuracy(Test[1], y_test, "KNeighborsClassifier")
calcAccuracy(Test[2], y_test, "GradientBoostingClassifier")
calcAccuracy(Test[3], y_test, "DecisionTreeClassifier")
calcAccuracy(Test[4], y_test, "LogisticRegression")
calcAccuracy(Test[5], y_test, "SupportVectorMachineClassifier")
calcAccuracy(Test[6], y_test, "NaiveBayesClassifier")
calcAccuracy(Test[7], y_test, "AdaBoostClassifier")
calcAccuracy(Test[8], y_test, "LightGBMClassifier")