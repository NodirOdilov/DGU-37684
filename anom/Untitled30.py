import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Загрузка данных
df = pd.read_csv('live_dat.csv', engine='python')
df.columns = df.columns.str.strip()

# Информация о данных
df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

# Кодирование строковых признаков
string_features = list(df.select_dtypes(include=['object']).columns)
string_features.remove('Label')

le = preprocessing.LabelEncoder()
df[string_features] = df[string_features].apply(lambda col: le.fit_transform(col))

# Статистика по меткам
benign_total = len(df[df['Label'] == "BENIGN"])
attack_total = len(df[df['Label'] != "BENIGN"])

# Балансировка датасета
indexes = []
benign_included_count = 0
dos = 0
ddos = 0
port_scan = 0

for index, row in df.iterrows():
    if row['Label'] == "DoS":
        if dos == 40000: continue
        dos += 1
        indexes.append(index)
    elif row['Label'] == "DDoS":
        if ddos == 30000: continue
        ddos += 1
        indexes.append(index)
    elif row['Label'] == "PortScan":
        if port_scan == 20000: continue
        port_scan += 1
        indexes.append(index)
    else:
        if benign_included_count == 180000: continue
        benign_included_count += 1
        indexes.append(index)

df = df.loc[indexes]
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Удаление ненужных признаков
excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
df = df.drop(columns=excluded, errors='ignore')

# Визуализация признаков
if 'init_bwd_win_byts' in df.columns:
    df['init_bwd_win_byts'].hist(figsize=(6,4), bins=10)
    plt.title("init_bwd_win_byts")
    plt.xlabel("Value bins")
    plt.ylabel("Density")
    plt.savefig('init_bwd_win_byts.png', dpi=300)

if 'init_fwd_win_byts' in df.columns:
    df['init_fwd_win_byts'].hist(figsize=(6,4), bins=10)
    plt.title("init_fwd_win_byts")
    plt.xlabel("Value bins")
    plt.ylabel("Density")
    plt.savefig('init_fwd_win_byts.png', dpi=300)

# Удаление признаков с низкой информативностью
excluded2 = ['init_bwd_win_byts', 'init_fwd_win_byts']
df = df.drop(columns=excluded2, errors='ignore')

# Разделение данных на признаки и метки
y = df['Label'].values
X = df.drop(columns=['Label'])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение дерева решений
decision_tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
decision_tree = decision_tree.fit(X_train, y_train)

# Вывод дерева решений
r = export_text(decision_tree, feature_names=X_train.columns.to_list())

# Обучение случайного леса
rf = RandomForestClassifier(n_estimators=250, random_state=42, oob_score=True)
rf.fit(X_train, y_train)

# Важность признаков
features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
webattack_features = []

for index, i in enumerate(indices[:40]):
    webattack_features.append(features[i])

# Визуализация важности признаков
indices = np.argsort(importances)[-30:]
plt.rcParams['figure.figsize'] = (10, 6)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#cccccc', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.grid()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

# Предсказание меток
y_pred = rf.predict(X_test)

# Отбор признаков
max_features = 40
webattack_features = webattack_features[:max_features]

# Корреляционная матрица
corr_matrix = df[webattack_features].corr()
plt.rcParams['figure.figsize'] = (16, 5)
g = sns.heatmap(corr_matrix, annot=True, fmt='.1g', cmap='Greys')
g.set_xticklabels(g.get_xticklabels(), verticalalignment='top', horizontalalignment='right', rotation=30)
plt.savefig('corr_heatmap.png', dpi=300, bbox_inches='tight')

# Удаление сильно коррелирующих признаков
cor = df[webattack_features].corr()
upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

to_be_removed = to_drop
webattack_features = [item for item in webattack_features if item not in to_be_removed]
webattack_features = webattack_features[:15]

# Итоговая корреляционная матрица
corr_matrix = df[webattack_features].corr()
plt.rcParams['figure.figsize'] = (10, 5)
sns.heatmap(corr_matrix, annot=True, fmt='.1g', cmap='Greys')

# Гиперпараметры для обучения модели
parameters = {'n_estimators': [30, 50, 70, 100], 
              'min_samples_leaf': [3, 5, 7, 10],
              'max_features': [3, 5, 7, 10, 12], 
              'max_depth': [5, 10, 17, 23]}
X = df[webattack_features]

parameters = {'n_estimators': [10],
              'min_samples_leaf': [3],
              'max_features': [3], 
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 30, 50]}
scoring = {'f1', 'accuracy'}
gcv = GridSearchCV(RandomForestClassifier(random_state=1), parameters, scoring=scoring, refit='f1', cv=10, return_train_score=True)
%time gcv.fit(X, y)
results = gcv.cv_results_

# Результаты GridSearchCV
cv_results = pd.DataFrame(gcv.cv_results_)

plt.figure(figsize=(8, 5))
plt.title("GridSearchCV results", fontsize=14)
plt.xlabel("max_depth")
plt.ylabel("f1")

ax = plt.gca()
ax.set_xlim(1, 30)
ax.set_ylim(0.9, 1)

X_axis = np.array(results['param_max_depth'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7)

# Вывод результатов
print(df.shape)
print(df['Label'].unique())
print(benign_total, attack_total)
print(df['Label'].value_counts())
print(df.columns)
print(X.shape, y.shape)
print(np.unique(y_train, return_counts=True))
print(X_train.shape)
print(cross_val_score(decision_tree, X_train, y_train, cv=10))
print(r)
print(np.unique(y_test, return_counts=True))
print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f} \nOut-of-bag Score: {:.2f}'
      .format(rf.score(X_train, y_train), rf.score(X_test, y_test), rf.oob_score_))
print(confusion_matrix(y_test, y_pred))
print(webattack_features)
print(to_drop)
print(webattack_features)
print(cv_results.head())