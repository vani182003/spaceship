# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_data = pd.read_csv("/kaggle/input/dataset/train.csv")
test_data = pd.read_csv("/kaggle/input/dataset/test.csv")
sample_submission = pd.read_excel("/kaggle/input/dataset/spaceship_data.csv.xlsx")

# Data Preprocessing

# Fill missing values for numerical columns
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# Fill missing categorical columns with 'Unknown'
train_data.fillna({'HomePlanet': 'Unknown', 'CryoSleep': False, 'Cabin': 'Unknown/0/Unknown',
                   'Destination': 'Unknown', 'VIP': False}, inplace=True)
test_data.fillna({'HomePlanet': 'Unknown', 'CryoSleep': False, 'Cabin': 'Unknown/0/Unknown',
                  'Destination': 'Unknown', 'VIP': False}, inplace=True)

# Extract Deck and Side from Cabin column
train_data['Deck'] = train_data['Cabin'].apply(lambda x: x.split('/')[0])
train_data['Side'] = train_data['Cabin'].apply(lambda x: x.split('/')[-1])
test_data['Deck'] = test_data['Cabin'].apply(lambda x: x.split('/')[0])
test_data['Side'] = test_data['Cabin'].apply(lambda x: x.split('/')[-1])

# Create a new feature: TotalSpend (sum of all amenities spent)
train_data['TotalSpend'] = train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test_data['TotalSpend'] = test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

# Create new feature: Group Size based on PassengerId
train_data['Group'] = train_data['PassengerId'].apply(lambda x: x.split('_')[0])
test_data['Group'] = test_data['PassengerId'].apply(lambda x: x.split('_')[0])

# Calculate group size for each passenger
train_data['GroupSize'] = train_data.groupby('Group')['Group'].transform('count')
test_data['GroupSize'] = test_data.groupby('Group')['Group'].transform('count')

# Create a new feature: IsAlone (whether the passenger is traveling alone)
train_data['IsAlone'] = train_data['GroupSize'] == 1
test_data['IsAlone'] = test_data['GroupSize'] == 1

# Create a new feature: CabinGroup (count of passengers in the same cabin)
train_data['CabinGroup'] = train_data.groupby('Cabin')['Cabin'].transform('count')
test_data['CabinGroup'] = test_data.groupby('Cabin')['Cabin'].transform('count')

# Create a new feature: SpentMoney (whether the passenger spent money)
train_data['SpentMoney'] = train_data['TotalSpend'] > 0
test_data['SpentMoney'] = test_data['TotalSpend'] > 0

# Convert CryoSleep and VIP into binary values (True/False -> 1/0)
train_data['CryoSleep'] = train_data['CryoSleep'].astype(int)
train_data['VIP'] = train_data['VIP'].astype(int)
test_data['CryoSleep'] = test_data['CryoSleep'].astype(int)
test_data['VIP'] = test_data['VIP'].astype(int)

# One-hot encode categorical columns
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])

# Align the train and test data (ensure both have the same columns)
X = train_data.drop(columns=['Transported', 'Name', 'PassengerId', 'Cabin', 'Group'])
X_test = test_data.drop(columns=['Name', 'PassengerId', 'Cabin', 'Group'])
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)  # Align columns for both train and test

y = train_data['Transported']

# Impute any remaining missing values in numerical columns
imputer = SimpleImputer(strategy='median')  # Fill missing values with the median for numerical columns
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Split the training data into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test set
test_predictions = clf.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

# Visualization of transported passengers
# Count how many were transported (True) and how many were not (False)
transported_counts = submission['Transported'].value_counts()

# Set up the plot (bar chart)
plt.figure(figsize=(8, 5))
sns.barplot(x=transported_counts.index, y=transported_counts.values, palette="viridis")
plt.xlabel('Transported to Another Dimension', fontsize=14)
plt.ylabel('Number of Passengers', fontsize=14)
plt.title('Passengers Transported (True vs False)', fontsize=16)
plt.show()

# Pie chart visualization
plt.figure(figsize=(6, 6))
plt.pie(transported_counts, labels=['Not Transported', 'Transported'], autopct='%1.1f%%', colors=["lightcoral", "lightgreen"])
plt.title('Percentage of Passengers Transported', fontsize=16)
plt.show()

# Optional: Print only rows where 'Transported' is True
true_transports = submission[submission['Transported'] == True]
print("\nPassengers transported (True):")
print(true_transports)