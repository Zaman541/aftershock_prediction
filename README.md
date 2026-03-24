# Aftershock Prediction System

A machine learning system for predicting whether an earthquake will trigger aftershocks, built using DBSCAN clustering and Random Forest classification.

## Model Performance

- **Accuracy**: 92.52%
- **F1-Score**: 0.26 (balanced precision-recall)
- **ROC-AUC**: 0.87 (strong discriminative ability)
- **Training Data**: 19,044 global earthquakes


## Methodology

1. **DBSCAN Clustering**: Groups earthquakes into spatio-temporal sequences
2. **Mainshock Identification**: Earliest earthquake in each cluster
3. **Feature Engineering**: 17 features including magnitude, depth, location, temporal, and regional activity
4. **Random Forest Classification**: Predicts aftershock triggering probability
5. **SMOTE Balancing**: Handles class imbalance in training data

## Technical Details

- **Algorithm**: Random Forest (500 trees)
- **Features**: 17 engineered features from basic earthquake parameters
- **Training**: 80-20 temporal split with SMOTE balancing
- **Validation**: Out-of-bag scoring and test set evaluation

## Applications

- Emergency response planning
- Earthquake risk assessment
- Seismological research
- Insurance and risk modeling
- Public warning systems

## Model Features

- **Original input features**
    - `mag` (magnitude)
    - `depth`
    - `latitude`
    - `longitude`

- **Derived seismic features**
    - `mag_squared` (`mag`^2)
    - `depth_log` (`log1p(depth)`)
    - `mag_depth_ratio` (`mag / (depth + 1)`)

- **Temporal features**
    - `hour`
    - `day_of_week`
    - `month`
    - `year`

- **Regional activity features (past 30 days, within 50 km)**
    - `recent_activity_30d`
    - `recent_avg_mag_30d`
    - `recent_max_mag_30d`

- **Geographic features**
    - `lat_abs`
    - `distance_from_equator`

- **Categorical encoded features**
    - One-hot encoded magnitude type columns from `magType` using `drop_first=True`
    - Included as all columns matching `magType_*` in the training data

## Troubleshooting

**Model not loading?**
- Run `python train_model.py` first
- Check that `aftershock_prediction_model.pkl` exists

**Missing packages?**
- Run `pip install -r requirements.txt`

**Prediction errors?**
- Ensure all required parameters are provided
- Check numeric value ranges (positive depth, valid coordinates)
- Use supported magnitude types only

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- imbalanced-learn >= 0.8.0
- haversine >= 2.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- joblib >= 1.1.0

## File Structure

- **`aftershock_prediction_clean.ipynb`** - Clean prediction interface (USE THIS)
- **`train_model.py`** - Model training script (run once)
- **`setup.py`** - Automated setup script
- **`requirements.txt`** - Python dependencies

## Risk Levels

- **🟢 LOW RISK** (< 40%): Minimal aftershock activity expected
- **🟡 MODERATE RISK** (40-70%): Possible aftershock activity, monitor closely
- **🔴 HIGH RISK** (> 70%): Expect significant aftershock activity


## Quick Start

### Option 1: Automated Setup
To install the necessary dependencies and set up the model automatically, run:

```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (one-time setup)
python train_model.py

# 3. Open the clean prediction notebook
# aftershock_prediction_clean.ipynb
```

### Example Usage:
```python
result = predict_aftershock_triggering(
    magnitude=7.2,
    depth=8.5,
    latitude=34.052,
    longitude=-118.243,
    magType_str='mw'
)

print(f"Will trigger aftershocks: {result['will_trigger_aftershocks']}")
print(f"Probability: {result['probability_percent']}")
print(f"Risk Level: {result['risk_level']}")
```

## Academic Context

This system was developed as part of a Final Year Project (FYP) combining:
- **Seismological Science**: Based on established earthquake physics
- **Machine Learning**: Advanced classification and clustering techniques
- **Practical Application**: Real-world earthquake risk assessment

**Ready to predict earthquake aftershocks!**