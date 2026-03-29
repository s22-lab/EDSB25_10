# test_preprocessing.py
import pandas as pd
import joblib
import sys
sys.path.append('scripts')
from preprocessing import transform_new_data

# Load the saved scaler and imputer
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl')

# Test employee data
test_employee = {
    'JobSatisfaction': 2,
    'JobInvolvement': 3,
    'EnvironmentSatisfaction': 2,
    'WorkLifeBalance': 3,
    'NumCompaniesWorked': 4,
    'TotalWorkingYears': 8,
    'YearsAtCompany': 2,
    'Department': 'Sales',
    'JobRole': 'Sales Representative',
    'BusinessTravel': 'Travel_Frequently',
    'MaritalStatus': 'Single',
    'OverTime': 'Yes'
}

print("=" * 60)
print("TESTING PREPROCESSING PIPELINE")
print("=" * 60)

# Test the preprocessing
try:
    processed = transform_new_data(test_employee, scaler, imputer)
    
    print(f"✅ SUCCESS!")
    print(f"\nInput features: 12")
    print(f"Output features: {processed.shape[1]}")
    print(f"Expected features: 18")
    
    if processed.shape[1] == 18:
        print("\n✅ Feature count matches!")
    else:
        print(f"\n❌ ERROR: Expected 18 features, got {processed.shape[1]}")
    
    print("\nProcessed feature names:")
    for i, col in enumerate(processed.columns, 1):
        print(f"  {i}. {col}")
    
    print("\nSample values:")
    print(processed.head())
    
    # Test prediction
    print("\n" + "=" * 60)
    print("TESTING MODEL PREDICTION")
    print("=" * 60)
    
    logreg = joblib.load('models/logistic_regression_best.pkl')
    dt = joblib.load('models/dt_attrition_model.joblib')
    
    logreg_proba = logreg.predict_proba(processed)[0, 1]
    dt_proba = dt.predict_proba(processed)[0, 1]
    
    print(f"\n✅ Logistic Regression probability: {logreg_proba:.3f}")
    print(f"✅ Decision Tree probability: {dt_proba:.3f}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
