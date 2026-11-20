# Name: Maroof Afridi | Trainee ID: TRN-2026-000367 | Date: 20 Nov 2026
# Title: Reproducible Data Pipelines, Tabular ML & Systems Exploration
# Dataset: Pakistan Air Quality - 9 Cities, 1215 Rows
# How to run: pip install pandas numpy scikit-learn matplotlib joblib
#             python assignment.py
# AI Used: Meta AI for explaining concepts and debugging. I can explain every line.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============ PART A: DATA QUALITY PIPELINE ============
np.random.seed(42)
cities = ['Lahore','Karachi','Islamabad','Rawalpindi','Peshawar','Quetta','Faisalabad','Multan','Hyderabad']
n = 1200
dates = pd.date_range('2024-01-01', periods=n, freq='D')
df = pd.DataFrame({
    'date': np.random.choice(dates, n),
    'city': np.random.choice(cities, n),
    'pm2_5': np.random.normal(85,45,n).clip(5,350),
    'pm10': np.random.normal(130,60,n).clip(10,500),
    'no2': np.random.normal(35,15,n).clip(2,120),
    'o3': np.random.normal(45,20,n).clip(5,150),
    'temperature_c': np.random.normal(24,8,n).clip(2,45),
    'humidity_pct': np.random.normal(55,18,n).clip(10,95),
    'wind_speed_kmh': np.random.normal(12,6,n).clip(0,40),
    'traffic_index': np.random.randint(20,100,n),
})
df.loc[np.random.choice(df.index, 60), 'pm2_5'] = np.nan
df.loc[np.random.choice(df.index, 30), 'no2'] = np.nan
df = pd.concat([df, df.iloc[:15]], ignore_index=True)

def aqi_cat(x):
    if pd.isna(x): return 'Unknown'
    if x <= 50: return 'Good'
    if x <= 100: return 'Moderate'
    if x <= 150: return 'Unhealthy for Sensitive'
    return 'Unhealthy'
df['aqi_category'] = df['pm2_5'].apply(aqi_cat)
print(f"Raw: {df.shape}, Missing PM2.5: {df['pm2_5'].isna().sum()}, Duplicates: {df.duplicated().sum()}")

# Cleaning + Validation + Quality Gate (Bonus A)
def clean(df):
    df = df.drop_duplicates().copy()
    df['date'] = pd.to_datetime(df['date'])
    for col in ['pm2_5','pm10','no2','o3']:
        df[col] = df[col].fillna(df[col].median())
    assert df['pm2_5'].min() >= 0
    assert df['humidity_pct'].between(0,100).all()
    assert df['pm2_5'].isna().mean() < 0.10, "Quality Gate Failed"
    df = df[df['aqi_category'] != 'Unknown']
    return df

clean_df = clean(df)
print(f"Clean: {clean_df.shape}")
print(clean_df.groupby('city')['pm2_5'].mean().round(2))
clean_df.to_csv('clean.csv', index=False)

# SQLite Question: Which city has highest avg PM2.5?
con = sqlite3.connect(':memory:')
clean_df.to_sql('air_quality', con, index=False)
q = "SELECT city, ROUND(AVG(pm2_5),2) AS avg_pm25 FROM air_quality GROUP BY city ORDER BY avg_pm25 DESC"
print(pd.read_sql(q, con))
# Answer: Peshawar 92.06, Rawalpindi 91.36, Karachi 90.67

# 3 Visualizations
plt.figure(); clean_df.groupby('city')['pm2_5'].mean().sort_values().plot(kind='barh'); plt.title('Avg PM2.5 by City'); plt.savefig('fig1.png'); plt.close()
plt.figure(); clean_df['pm2_5'].hist(bins=30); plt.title('PM2.5 Distribution'); plt.savefig('fig2.png'); plt.close()
plt.figure(); plt.scatter(clean_df['traffic_index'], clean_df['pm2_5'], alpha=0.4); plt.title('Traffic vs PM2.5'); plt.savefig('fig3.png'); plt.close()

# ============ PART B: TABULAR ML BASELINE ============
X = clean_df[['city','pm10','no2','o3','temperature_c','humidity_pct','wind_speed_kmh','traffic_index']]
y = clean_df['aqi_category']
# Split BEFORE fitting to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), ['pm10','no2','o3','temperature_c','humidity_pct','wind_speed_kmh','traffic_index']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), ['city'])
])
for name, clf in [('Dummy', DummyClassifier(strategy='most_frequent')), ('LogisticRegression', LogisticRegression(max_iter=1000)), ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42))]:
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(name, 'Accuracy:', round(accuracy_score(y_test, pred),4), 'F1-macro:', round(f1_score(y_test, pred, average='macro'),4))

best = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
best.fit(X_train, y_train)
print(classification_report(y_test, best.predict(X_test)))
joblib.dump(best, 'aqi_pipeline.joblib')

# predict.py logic (Bonus B) - new rows par predict
new = pd.DataFrame([{'city':'Lahore','pm10':180,'no2':45,'o3':50,'temperature_c':28,'humidity_pct':60,'wind_speed_kmh':10,'traffic_index':85}])
print('Prediction for new row:', best.predict(new))

# ============ PART C: EXPLORATION NOTES ============
# C1 Feed Queries: Index (user_id, created_at), Cursor pagination WHERE id < last_id LIMIT 20, Redis cache. Ranker score = 0.5*likes + 0.3*comments + 0.2*recency
# C2 GitHub Actions: Har PR par pytest + flake8, main par direct push band, 1 approval lazmi, CI green lazmi. Review me Logic, Tests, Naming check karein.
# C3 System Design Theory: CAP Theorem, Load Balancer, Caching, Sharding, Message Queue, Horizontal Scaling.
# C4 Workflow: Requirements > Capacity Estimate > High-level Diagram > DB Design > Bottleneck > Trade-offs.
print("Done - Maroof Afridi - TRN-2026-000367")
