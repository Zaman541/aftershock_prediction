"""
Setup Script for Aftershock Prediction System
=============================================

This script will set up your aftershock prediction system by:
1. Installing required dependencies
2. Training the model (if not already trained)
3. Providing usage instructions

Run this script once to set up everything.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("   Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("   All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Error installing packages: {e}")
        return False

def check_model_exists():
    """Check if trained model exists"""
    model_file = Path("aftershock_prediction_model.pkl")
    return model_file.exists()

def train_model():
    """Run model training script"""
    print("   Training the aftershock prediction model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("   Model training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Error training model: {e}")
        return False

def main():
    print("   Setting up Aftershock Prediction System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("   requirements.txt not found!")
        print("Please run this script from the project directory.")
        return
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed at package installation step.")
        return
    
    # Step 2: Check/train model
    if check_model_exists():
        print("   Trained model found - no need to retrain!")
    else:
        print("   No trained model found - training now...")
        if not train_model():
            print("Setup failed at model training step.")
            return
    
    # Step 3: Success message
    print("\n" + "=" * 50)
    print("   SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n   Files available:")
    print("   • aftershock_prediction_model.pkl (trained model)")
    print("   • aftershock_prediction_clean.ipynb (clean prediction notebook)")
    print("   • feature_importance_report.csv (model analysis)")
    print("   • requirements.txt (dependencies)")
    print("   • train_model.py (model training script)")
    
    print("\n   Next steps:")
    print("   1. Open 'aftershock_prediction_clean.ipynb' in Jupyter/VS Code")
    print("   2. Run the cells to start predicting aftershocks!")
    print("   3. Modify the example predictions with your own earthquake data")
    
    print("\n   Usage:")
    print("   • For predictions: Use the clean notebook")
    print("   • To retrain: Run 'python train_model.py'")
    print("   • To reinstall: Run 'python setup.py'")
    
    print("\n   Ready to predict earthquake aftershocks!")

if __name__ == "__main__":
    main()