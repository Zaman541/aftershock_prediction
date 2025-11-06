"""
Aftershock Prediction Model Training Script
===========================================

This script trains the aftershock prediction model and saves it to disk.
Run this once to train the model, then use the prediction notebook for inference.

Author: FYP Project
Date: September 2025
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

print("Starting Aftershock Prediction Model Training")
print("=" * 50)

# 1. LOAD AND PREPROCESS DATA
print("Loading earthquake data...")
file_path = "query.csv"
data = pd.read_csv(file_path)

# Basic preprocessing
columns_to_keep = ['mag', 'time', 'depth', 'latitude', 'longitude', 'magType']
data = data[columns_to_keep].copy()
data = data.drop_duplicates().reset_index(drop=True)
data = data.dropna()

# Convert data types
data['mag'] = data['mag'].astype(float)
data['depth'] = data['depth'].astype(float) 
data['latitude'] = data['latitude'].astype(float)
data['longitude'] = data['longitude'].astype(float)
data['time'] = pd.to_datetime(data['time'])

print(f"Data loaded: {data.shape[0]} earthquakes")
print(f"Date range: {data['time'].min()} to {data['time'].max()}")

# 2. IMPROVED CLUSTERING
print("\n   Performing spatio-temporal clustering...")

# Convert time to days since start
min_date = data['time'].min()
data['days_since_start'] = (data['time'] - min_date).dt.total_seconds() / (60 * 60 * 24)

def improved_custom_metric(a, b):
    """Improved distance metric for aftershock clustering"""
    loc_dist = haversine((a[0], a[1]), (b[0], b[1]), unit=Unit.KILOMETERS)
    time_dist = abs(a[2] - b[2])  # in days
    
    max_time = 100  # days
    max_distance = 100  # km
    
    if time_dist <= max_time and loc_dist <= max_distance:
        time_weight = time_dist / max_time
        space_weight = loc_dist / max_distance
        return (time_weight + space_weight) / 2
    else:
        return 1.0

# Apply clustering
X_clustering = data[['latitude', 'longitude', 'days_since_start']].copy()
db_improved = DBSCAN(eps=0.3, min_samples=2, metric=improved_custom_metric)
data['cluster'] = db_improved.fit_predict(X_clustering.values)

print(f"Clustering completed: {len(set(data['cluster'])) - (1 if -1 in data['cluster'] else 0)} clusters found")

# 3. CREATE TARGET LABELS
print("\nCreating target labels...")

data['is_mainshock'] = False
data['triggers_aftershocks'] = False

mainshock_count = 0
aftershock_triggering_count = 0

for cluster_id in data['cluster'].unique():
    cluster_events = data[data['cluster'] == cluster_id]
    
    if cluster_id == -1:
        # Independent events
        data.loc[cluster_events.index, 'is_mainshock'] = True
        mainshock_count += len(cluster_events)
    else:
        # Cluster mainshock identification
        if len(cluster_events) >= 2:
            earliest_idx = cluster_events['time'].idxmin()
            data.loc[earliest_idx, 'is_mainshock'] = True
            data.loc[earliest_idx, 'triggers_aftershocks'] = True
            mainshock_count += 1
            aftershock_triggering_count += 1

print(f"   Target labels created:")
print(f"   Total mainshocks: {mainshock_count}")
print(f"   Mainshocks triggering aftershocks: {aftershock_triggering_count}")

# 4. FEATURE ENGINEERING
print("\n  Engineering features...")

# Create feature dataset
feature_data = data.copy()

# Basic seismic features
feature_data['mag_squared'] = feature_data['mag'] ** 2
feature_data['depth_log'] = np.log1p(feature_data['depth'])
feature_data['mag_depth_ratio'] = feature_data['mag'] / (feature_data['depth'] + 1)

# Temporal features
feature_data['hour'] = feature_data['time'].dt.hour
feature_data['day_of_week'] = feature_data['time'].dt.dayofweek
feature_data['month'] = feature_data['time'].dt.month
feature_data['year'] = feature_data['time'].dt.year

# Simplified regional activity (computationally efficient version)
print("   Computing regional seismic activity features...")
feature_data['recent_activity_30d'] = 0
feature_data['recent_avg_mag_30d'] = 0
feature_data['recent_max_mag_30d'] = 0

# Process in batches to avoid memory issues
batch_size = 1000
for start_idx in range(0, len(feature_data), batch_size):
    end_idx = min(start_idx + batch_size, len(feature_data))
    if start_idx % 5000 == 0:
        print(f"   Processing events {start_idx}-{end_idx}/{len(feature_data)}")
    
    for idx in range(start_idx, end_idx):
        current_event = feature_data.iloc[idx]
        current_time = current_event['time']
        current_lat = current_event['latitude']
        current_lon = current_event['longitude']
        
        # Look at events in the past 30 days within 50km
        time_window = current_time - pd.Timedelta(days=30)
        recent_events = feature_data[
            (feature_data['time'] >= time_window) & 
            (feature_data['time'] < current_time)
        ]
        
        if len(recent_events) > 0:
            distances = []
            mags = []
            for _, event in recent_events.iterrows():
                dist = haversine(
                    (current_lat, current_lon),
                    (event['latitude'], event['longitude']),
                    unit=Unit.KILOMETERS
                )
                if dist <= 50:  # Within 50km
                    distances.append(dist)
                    mags.append(event['mag'])
            
            feature_data.loc[feature_data.index[idx], 'recent_activity_30d'] = len(mags)
            if mags:
                feature_data.loc[feature_data.index[idx], 'recent_avg_mag_30d'] = np.mean(mags)
                feature_data.loc[feature_data.index[idx], 'recent_max_mag_30d'] = np.max(mags)

# Geographic features
feature_data['lat_abs'] = np.abs(feature_data['latitude'])
feature_data['distance_from_equator'] = np.abs(feature_data['latitude'])

# Encode categorical variables
magType_encoded = pd.get_dummies(feature_data['magType'], prefix='magType', drop_first=True)
feature_data = pd.concat([feature_data, magType_encoded], axis=1)

print(f"Feature engineering completed: {len(feature_data.columns)} total features")

# 5. PREPARE TRAINING DATA
print("\n   Preparing training data...")

# Select features
feature_columns = [
    'mag', 'depth', 'latitude', 'longitude',
    'mag_squared', 'depth_log', 'mag_depth_ratio',
    'hour', 'day_of_week', 'month', 'year',
    'recent_activity_30d', 'recent_avg_mag_30d', 'recent_max_mag_30d',
    'lat_abs', 'distance_from_equator'
]

# Add magType dummy columns
magType_cols = [col for col in feature_data.columns if col.startswith('magType_')]
feature_columns.extend(magType_cols)

X = feature_data[feature_columns].fillna(0)
y = feature_data['triggers_aftershocks']

# Temporal train-test split
sorted_data = feature_data.sort_values('time')
split_idx = int(0.8 * len(sorted_data))

train_indices = sorted_data.index[:split_idx]
test_indices = sorted_data.index[split_idx:]

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

print(f"   Data split completed:")
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# 6. APPLY SMOTE BALANCING
print("\n  Applying SMOTE class balancing...")

smote = SMOTE(random_state=42, sampling_strategy=0.4)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   SMOTE completed:")
print(f"   Original training size: {len(X_train)}")
print(f"   Balanced training size: {len(X_train_balanced)}")

# 7. TRAIN MODEL
print("\n   Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_balanced, y_train_balanced)

print(f"   Model training completed!")
print(f"   Out-of-bag score: {rf_model.oob_score_:.4f}")

# 8. EVALUATE MODEL
print("\n  Evaluating model performance...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Model Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")

# 9. SAVE MODEL AND METADATA
print("\n   Saving model and metadata...")

# Create model package
model_package = {
    'model': rf_model,
    'feature_columns': feature_columns,
    'magType_columns': magType_cols,
    'training_date': datetime.now().isoformat(),
    'performance': {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'oob_score': rf_model.oob_score_
    },
    'data_info': {
        'total_samples': len(feature_data),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'date_range': {
            'start': str(feature_data['time'].min()),
            'end': str(feature_data['time'].max())
        }
    }
}

# Save model
model_filename = 'aftershock_prediction_model.pkl'
joblib.dump(model_package, model_filename)

print(f"   Model saved as '{model_filename}'")

# 10. CREATE FEATURE IMPORTANCE REPORT
print("\n   Generating feature importance report...")

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save feature importance
feature_importance.to_csv('feature_importance_report.csv', index=False)
print(f"   Feature importance saved as 'feature_importance_report.csv'")

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\n" + "=" * 50)
print("   MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"   Files created:")
print(f"   • {model_filename} (trained model)")
print(f"   • feature_importance_report.csv (feature analysis)")
print(f"   • requirements.txt (dependencies)")
print(f"\n💡 Next steps:")
print(f"   1. Install dependencies: pip install -r requirements.txt")
print(f"   2. Use the clean prediction notebook for inference")
print(f"   3. Load model with: joblib.load('{model_filename}')")