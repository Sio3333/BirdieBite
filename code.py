#!/usr/bin/env python
# coding: utf-8

# # 1. Generate Dataset

# In[ ]:


import numpy as np
import pandas as pd
import random
#Data generate
np.random.seed(46)
num_samples = 422

F1 = np.random.normal(35, 13, num_samples).astype(int)
F1 = np.clip(F1, 9, 62)

F2 = np.round(np.random.uniform(12.00, 22.00, num_samples), 2)

F3 = np.round(np.random.uniform(0.90, 15.00, num_samples), 2)

F4 = np.random.randint(1, 6, num_samples)


F5_weights = [2/3] + [1/27 for _ in range(9)]
F5 = np.random.choice(10, num_samples, p=F5_weights)

labels = []
for i in range(num_samples):
    label = 2  # Neutral by default
    if 20 <= F1[i] <= 40 or F2[i] <= 18.5 and F2[i] >= 17.5 or F3[i] <= 1.5 or F4[i] == 3 or F5[i] == 0:
        label = 3  # Healthy
    if F1[i] > 50 or F1[i] < 12 or F4[i] == 1 or F4[i] == 5 or F5[i] > 7:
        label = 1  # Bad
    labels.append(label)


count_label_2 = labels.count(2)
while count_label_2 < num_samples * 0.4:
    idx = random.randint(0, num_samples-1)
    if labels[idx] != 2:
        labels[idx] = 2
        count_label_2 += 1
#Noise generate
noise_ratio = 0.05
num_noise_samples = int(num_samples * noise_ratio)
noise_indices = np.random.choice(num_samples, num_noise_samples, replace=False)

for idx in noise_indices:
    F1[idx] = random.randint(9, 62)
    F2[idx] = round(random.uniform(12.00, 22.00), 2)
    F3[idx] = round(random.uniform(0.90, 15.00), 2)
    F4[idx] = random.randint(1, 5)
    F5[idx] = random.randint(0, 9)
    labels[idx] = random.choice([1, 2, 3])
    
# Creating Dataframe
df = pd.DataFrame({
    'F1': F1,
    'F2': F2,
    'F3': F3,
    'F4': F4,
    'F5': F5,
    'Label': labels
})

df.to_excel('dataset_generated.xlsx', index=False)


# ##### Manually combine the generated data with the original data into a 500 volumed dataset 

# #  2.Learning&Comparing models

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_excel('dataset_500.xlsx')
X = data[['F1', 'F2', 'F3', 'F4', 'F5']]
y = data['Label']

#Split datas into train set and test set in ratio of 4：1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=33)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# # 3.Validation of RF：Adjusting n_estimator parameter

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_excel('dataset_500.xlsx')


X = data[['F1', 'F2', 'F3', 'F4', 'F5']]
y = data['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
estimators_range = range(10, 401, 10)
accuracies = []


for n_estimators in estimators_range:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
    accuracies.append(scores.mean())


plt.figure(figsize=(10, 6))
plt.plot(estimators_range, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# # 4.Validation of RF：Confusion matrix & Feature Importance

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_excel('dataset_500.xlsx')

X = data[['F1', 'F2', 'F3', 'F4', 'F5']]
y = data['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)


rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)


feature_importances = rf_classifier.feature_importances_
features = ['F1', 'F2', 'F3', 'F4', 'F5']

plt.bar(features, feature_importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance using Random Forest')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

