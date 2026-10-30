"""
Diagnostic : la CV Stratified aléatoire fuit-elle ?
Compare StratifiedKFold (aléatoire) vs GroupKFold par date (out-of-time simulé)
sur le pipeline benchmark officiel, avec et sans la variable pred_date.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


def parse(df):
    df = df.copy()
    dt = pd.to_datetime(df['prediction_time'].astype(str), errors='coerce')
    df['pred_hour'] = dt.dt.hour
    df['pred_dow'] = dt.dt.day_of_week
    df['pred_date'] = dt.dt.date.astype(str)
    return df


train, test = parse(train), parse(test)
print('Dates train :', train['pred_date'].min(), '->', train['pred_date'].max(),
      f"({train['pred_date'].nunique()} jours)")
print('Dates test  :', test['pred_date'].min(), '->', test['pred_date'].max(),
      f"({test['pred_date'].nunique()} jours)")
print('Dates test vues en train :', len(set(test['pred_date']) & set(train['pred_date'])))
print('Communautés test inconnues :',
      sorted(set(test['community'].str.strip().str.lower()) - set(train['community'].str.strip().str.lower())))


def make_pipe(cols):
    num = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat)])
    return Pipeline([('pre', pre), ('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))])


base = ['user_id', 'confidence', 'predicted_intensity', 'community', 'district',
        'indicator', 'forecast_length', 'pred_hour', 'pred_dow']
configs = {'benchmark (avec pred_date)': base + ['pred_date'], 'sans pred_date': base}
y = train['Target']

for name, cols in configs.items():
    X = train[cols]
    for cv_name, splitter, groups in [
        ('Stratified aléatoire', StratifiedKFold(5, shuffle=True, random_state=42), None),
        ('GroupKFold par date ', GroupKFold(5), train['pred_date'])]:
        oof = pd.Series(index=y.index, dtype=object)
        for tr, va in splitter.split(X, y, groups):
            p = make_pipe(cols).fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[va] = p.predict(X.iloc[va])
        print(f'{name:28s} | {cv_name} | Macro F1 = {f1_score(y, oof, average="macro"):.4f}')
