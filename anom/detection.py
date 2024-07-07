# Подготовка датасета для обучения
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, add, GlobalAveragePooling1D, Input
from keras import optimizers, losses
from keras.utils import to_categorical
import pickle

# Загрузка и предобработка данных
df_dataset = pd.read_csv("./live_dat.csv", low_memory=False)
df_dataset["Label"].value_counts()

train, test = train_test_split(df_dataset, test_size=0.3, random_state=420)
numerical_cols = df_dataset.columns[:-3]
min_max_scaler = MinMaxScaler().fit(train[numerical_cols])
train[numerical_cols] = min_max_scaler.transform(train[numerical_cols])
test[numerical_cols] = min_max_scaler.transform(test[numerical_cols])

y_train = np.array(train.pop("Label"))
X_train = train.values
y_test = np.array(test.pop("Label"))
X_test = test.values

# Определение моделей и их параметров
label = {0: 'BENIGN', 1: 'http', 2: 'udp'}
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in label.keys()}

# Модель случайного леса
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights_dict,
    n_jobs=-1
)
hyperparameters_rf = {'n_estimators': [75, 100, 125, 150, 175]}
rf_grid = GridSearchCV(estimator=rf_model, param_grid=hyperparameters_rf, cv=2, verbose=1, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Модель дерева решений
dt_model = DecisionTreeClassifier()
hyperparameters_dt = {'max_depth': [i for i in range(10, 20)]}
dt_grid = GridSearchCV(estimator=dt_model, param_grid=hyperparameters_dt, cv=2, verbose=1, n_jobs=-1)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

# Стекинг моделей
stacking_estimators = [('rf', best_rf), ('dt', best_dt)]
stacking_model = VotingClassifier(estimators=stacking_estimators, voting='hard')
stacking_model.fit(X_train, y_train)

# Оценка и сохранение модели
y_pred = stacking_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=list(label.values())))
with open('Models/log.pkl', 'wb') as file:
    pickle.dump(stacking_model, file)

# Функции для создания различных моделей
def create_gru_model():
    model = Sequential()
    model.add(GRU(128, input_shape=(71, 1), return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

def create_cnn_model():
    model = Sequential([
        Conv1D(128, 16, padding='same', activation='relu', input_shape=(71, 1)),
        MaxPooling1D(2),
        Conv1D(64, 16, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(32, 8, padding='same', activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def create_ensemble_model():
    input_layer = Input(shape=(71, 1))
    gru_model = create_gru_model()(input_layer)
    cnn_model = create_cnn_model()(input_layer)
    lstm_model = create_lstm_model()(input_layer)
    merged = concatenate([cnn_model, gru_model, lstm_model])
    dense = Dense(128, activation='relu')(merged)
    output_layer = Dense(3, activation='softmax')(dense)
    ensemble_model = Model(inputs=input_layer, outputs=output_layer)
    return ensemble_model

# Компиляция и обучение моделей
Y_train = to_categorical(y_train, num_classes=3)
Y_test = to_categorical(y_test, num_classes=3)

model = create_ensemble_model()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=20)

mod_pred = np.argmax(model.predict(X_test), axis=1)
mod_pred_categorical = to_categorical(mod_pred)
print(classification_report(Y_test, mod_pred_categorical, target_names=list(label.values())))
