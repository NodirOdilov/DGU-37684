import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM

# Load train and test data
with open('train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)

with open('nemwaf.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

# Extract payload and injection data
X_train = [item.get("payload", []) for item in train]
X_test = [item.get("payload", []) for item in test]
y_train = [item.get("injection", []) for item in train]
y_test = [item.get("injection", []) for item in test]

# Create a list to hold the combined JSON elements
combined_json = []

# Iterate over the elements and create combined JSON
for x_train, y_train in zip(X_train, y_train):
    combined_element = {
        "payload": x_train,
        "injection": y_train
    }
    combined_json.append(combined_element)

for x_test, y_test in zip(X_test, y_test):
    combined_element = {
        "payload": x_test,
        "injection": y_test
    }
    combined_json.append(combined_element)

dat = [item.get("payload", []) for item in combined_json]
inj = [item.get("injection", []) for item in combined_json]

X_train, X_test, y_train, y_test = train_test_split(dat, inj, test_size=0.2, random_state=42)

# Создаем объект LabelEncoder и преобразуем значения y_train и y_test
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Создаем объект Tokenizer и преобразуем инъекцию в числовые данные
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train, mode='binary')
X_test = tokenizer.texts_to_matrix(X_test, mode='binary')

# Logistic Regression
lr = LogisticRegression(C=10.0)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Logistic Regression - Accuracy:", accuracy)
print("Logistic Regression - Precision:", precision)
print("Logistic Regression - Recall:", recall)
print("Logistic Regression - F1 score:", f1)

# Random Forest with class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: class_weights[0], 1: class_weights[1]}
model = RandomForestClassifier(class_weight=class_weights)

hyperparameters = {'n_estimators': [50, 75, 100, 125, 150, 175, 200]}
rf = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=2, verbose=1, n_jobs=-1)
rf.fit(X_train, y_train)

models = {}
models['random_forest'] = rf.best_estimator_
y_pred = models['random_forest'].predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Random Forest - Accuracy:", accuracy)
print("Random Forest - Precision:", precision)
print("Random Forest - Recall:", recall)
print("Random Forest - F1 score:", f1)

# Convolutional Neural Network
model = Sequential([
    Conv1D(128, 16, padding='same', activation='relu', input_shape=(1000, 1)),
    MaxPooling1D(2),
    Conv1D(64, 16, padding='same', activation='relu'),
    MaxPooling1D(2),
    Conv1D(32, 8, padding='same', activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("CNN - Accuracy:", accuracy)
print("CNN - Precision:", precision)
print("CNN - Recall:", recall)
print("CNN - F1 score:", f1)

# Support Vector Machine
clf = LinearSVC(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("SVM - Accuracy:", accuracy)
print("SVM - Precision:", precision)
print("SVM - Recall:", recall)
print("SVM - F1 score:", f1)

# GRU Network
model = Sequential([
    Conv1D(128, 16, padding='same', activation='relu', input_shape=(1000, 1)),
    MaxPooling1D(2),
    Conv1D(64, 16, padding='same', activation='relu'),
    MaxPooling1D(2),
    Conv1D(32, 8, padding='same', activation='relu'),
    MaxPooling1D(2),
    GRU(32),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("GRU - Accuracy:", accuracy)
print("GRU - Precision:", precision)
print("GRU - Recall:", recall)
print("GRU - F1 score:", f1)
