import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Read adult.data file into Pandas DataFrame
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income']

df = pd.read_csv('C:/Users/A/Desktop/Census Income/adult.data', sep=',', header=None, names=column_names, na_values=' ?',
                 skipinitialspace=True)

# Handle missing values
df = df.replace('?', pd.NA)
df = df.dropna()

# Ordinal encode the categorical variables
workclass_order = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
encoder = OrdinalEncoder(categories=[workclass_order])
df['workclass_encoded'] = encoder.fit_transform(df[['workclass']])

education_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                   'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters', 'Doctorate', 'Prof-school']
encoder = OrdinalEncoder(categories=[education_order])
df['education_encoded'] = encoder.fit_transform(df[['education']])

marital_status_order = ['Never-married', 'Separated', 'Divorced', 'Widowed',
                        'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
encoder = OrdinalEncoder(categories=[marital_status_order])
df['marital_status_encoded'] = encoder.fit_transform(df[['marital_status']])

relationship_order = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
encoder = OrdinalEncoder(categories=[relationship_order])
df['relationship_encoded'] = encoder.fit_transform(df[['relationship']])

race_order = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Black', 'Other']
encoder = OrdinalEncoder(categories=[race_order])
df['race_encoded'] = encoder.fit_transform(df[['race']])

sex_order = ['Female', 'Male']
encoder = OrdinalEncoder(categories=[sex_order])
df['sex_encoded'] = encoder.fit_transform(df[['sex']])

native_country_order = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
                        'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',
                        'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland',
                        'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                        'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',
                        'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']
encoder = OrdinalEncoder(categories=[native_country_order])
df['native_country_encoded'] = encoder.fit_transform(df[['native_country']])

# Exploratory Data Analysis
# Descriptive Analytics
numerical_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
pd.set_option('display.max_columns', None)
summary_statistics = df[numerical_features].describe()
print(summary_statistics)

# Count plot of education level by income category
plt.figure(figsize=(12, 8))
sns.countplot(x='education_encoded', hue='income', data=df)
plt.title('Count Plot of Education Level by Income')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Count plot of race by income category
plt.figure(figsize=(12, 8))
sns.countplot(x='race_encoded', hue='income', data=df)
plt.title('Count Plot of Race by Income')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()

# Count plot of gender by income category
plt.figure(figsize=(12, 8))
sns.countplot(x='sex_encoded', hue='income', data=df)
plt.title('Count Plot of Gender by Income')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Stacked Bar Chart of Workclass against Income
pd.crosstab(df['workclass_encoded'], df['income']).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Workclass against Income')
plt.xlabel('Workclass')
plt.ylabel('Count')
plt.show()

# K-clustering
X = df[['age', 'education_encoded', 'hours_per_week', 'race_encoded', 'sex_encoded']]
kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
df['cluster'] = kmeans.fit_predict(X)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='education', hue='cluster', data=df, palette='viridis', s=50)
plt.title('K-Means Clustering Results')
plt.xlabel('Age')
plt.ylabel('Education Level')
plt.show()

# Random Forest Classification
# Assuming X contains the features and y is the 'income' column
X = df[['age', 'education_encoded', 'workclass_encoded', 'hours_per_week', 'marital_status_encoded',
        'capital_gain', 'capital_loss']]
y = df[['income']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Create a DataFrame to view the predictions and actual values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, predictions))