"""
HR Attrition Prediction App
Models: Logistic Regression (Conservative) & Decision Tree (Aggressive)
Uses recall-optimized thresholds for maximum attrition detection
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import sys

# Add scripts directory to path
sys.path.append('scripts')
from preprocessing import transform_new_data

# ============================================
# 1. LOAD MODELS AND METADATA
# ============================================

logreg = joblib.load("models/logistic_regression_best.pkl")
dt = joblib.load("models/dt_attrition_model.joblib")
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl') 

with open("models/logreg_model_metadata.json", 'r') as f:
    logreg_metadata = json.load(f)
with open("models/dt_model_metadata.json", 'r') as f:
    dt_metadata = json.load(f)

# Recall-optimized thresholds (FIXED)
THRESHOLD_LOGREG = logreg_metadata['thresholds']['recall_optimized']  # 0.400
THRESHOLD_DT = dt_metadata['thresholds']['recall_optimized']  # 0.320

# ============================================
# 2. PREDICTION FUNCTION
# ============================================

def predict_attrition(
    job_satisfaction, job_involvement, environment_satisfaction, work_life_balance,
    num_companies_worked, total_working_years, years_at_company,
    department, job_role, business_travel, marital_status, overtime,
    model_name
):
    """
    Predict attrition using RAW employee data (18 features only)
    """
    
    # Create input dictionary with RAW values (no EducationField - not in final 18 features)
    input_data = {
        'JobSatisfaction': job_satisfaction,
        'JobInvolvement': job_involvement,
        'EnvironmentSatisfaction': environment_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'NumCompaniesWorked': num_companies_worked,
        'TotalWorkingYears': total_working_years,  # Will be log-transformed
        'YearsAtCompany': years_at_company,        # Will be log-transformed
        'Department': department,
        'JobRole': job_role,
        'BusinessTravel': business_travel,
        'MaritalStatus': marital_status,
        'OverTime': overtime
    }
    
    try:
        # Apply preprocessing (uses your preprocessing.py transform_new_data!)
        df_processed = transform_new_data(input_data, scaler, imputer)
        
        # Select model and threshold
        if model_name == "Logistic Regression (Conservative)":
            model = logreg
            threshold = THRESHOLD_LOGREG
            model_desc = "Conservative"
        else:
            model = dt
            threshold = THRESHOLD_DT
            model_desc = "Aggressive"
        
        # Get probability
        proba = model.predict_proba(df_processed)[0, 1]
        
        # Apply recall-optimized threshold
        prediction = int(proba >= threshold)
        label = "🔴 LEAVE" if prediction == 1 else "🟢 STAY"
        
        # Risk level and action
        if proba < 0.30:
            risk_level = "🟢 Low Risk"
            action = "Continue regular check-ins and maintain engagement."
        elif proba < 0.50:
            risk_level = "🟡 Medium Risk"
            action = "Schedule a 1-on-1 to discuss career goals and satisfaction."
        elif proba < 0.70:
            risk_level = "🟠 High Risk"
            action = "Conduct retention interview. Review compensation and work-life balance."
        else:
            risk_level = "🔴 Critical Risk"
            action = "**URGENT:** Immediate manager intervention required. Consider counter-offer if valuable employee."
        
        # Explanation
        explanation = f"""
        ### 📊 Prediction Result
        
        **Model:** {model_name}  
        **Strategy:** {model_desc} (Recall-Optimized)  
        **Threshold:** {threshold:.3f}
        
        ---
        
        **Attrition Probability:** {proba:.1%}  
        **Prediction:** {label}  
        **Risk Level:** {risk_level}
        
        ---
        
        ### 💡 Recommended Action
        
        {action}
        """
        
        return proba, label, risk_level, explanation
    
    except Exception as e:
        return 0.0, "❌ Error", "Error", f"**Error occurred:**\n\n{str(e)}\n\nPlease check that all inputs are valid."

# ============================================
# 3. BATCH PREDICTION
# ============================================

def predict_batch(csv_file, model_name):
    """
    Batch predictions from CSV (18 features only)
    """
    if csv_file is None:
        return "⚠️ Please upload a CSV file", None
    
    try:
        df = pd.read_csv(csv_file.name)
        
        # Only 12 raw columns needed (18 features after transformation)
        required_cols = [
            'JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction', 'WorkLifeBalance',
            'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany',
            'Department', 'JobRole', 'BusinessTravel', 'MaritalStatus', 'OverTime'
        ]
        
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"❌ **Missing columns:** {', '.join(missing)}", None
        
        # Select model and threshold
        if model_name == "Logistic Regression (Conservative)":
            model = logreg
            threshold = THRESHOLD_LOGREG
        else:
            model = dt
            threshold = THRESHOLD_DT
        
        # Process all rows at once using transform_new_data
        df_processed = transform_new_data(df[required_cols], scaler, imputer)
        
        # Get predictions
        probas = model.predict_proba(df_processed)[:, 1]
        predictions = (probas >= threshold).astype(int)
        
        # Create results
        results_df = df.copy()
        results_df['Attrition_Probability'] = np.round(probas, 4)
        results_df['Prediction'] = predictions
        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Stay', 1: 'Leave'})
        results_df['Risk_Level'] = pd.cut(
            probas, 
            bins=[0, 0.3, 0.5, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Summary
        total = len(results_df)
        leavers = predictions.sum()
        avg_risk = probas.mean()
        
        critical_count = (probas >= 0.70).sum()
        high_count = ((probas >= 0.50) & (probas < 0.70)).sum()
        medium_count = ((probas >= 0.30) & (probas < 0.50)).sum()
        low_count = (probas < 0.30).sum()
        
        summary = f"""
        ### 📊 Batch Prediction Summary
        
        **Model:** {model_name}  
        **Threshold:** {threshold:.3f} (Recall-Optimized)
        
        ---
        
        **Total Employees:** {total}  
        **Predicted Leavers:** {leavers} ({leavers/total:.1%})  
        **Average Risk Score:** {avg_risk:.1%}
        
        ---
        
        ### Risk Distribution
        
        - 🔴 **Critical Risk:** {critical_count} employees ({critical_count/total:.1%})
        - 🟠 **High Risk:** {high_count} employees ({high_count/total:.1%})
        - 🟡 **Medium Risk:** {medium_count} employees ({medium_count/total:.1%})
        - 🟢 **Low Risk:** {low_count} employees ({low_count/total:.1%})
        """
        
        return summary, results_df
    
    except Exception as e:
        return f"❌ **Error processing file:**\n\n{str(e)}", None

# ============================================
# 4. GRADIO INTERFACE
# ============================================

with gr.Blocks(title="HR Attrition Prediction") as demo:
    
    gr.Markdown("""
    # 🎯 HR Attrition Prediction System
    
    Predict employee attrition risk using **18 key features** and **recall-optimized** thresholds.
    
    **Choose Your Model:**
    - **Logistic Regression (Conservative):** More reliable, fewer false alarms (58% precision, 77% recall)
    - **Decision Tree (Aggressive):** More sensitive, catches edge cases (25% precision, 79% recall)
    """)
    
    with gr.Tabs():
        
        # ============================================
        # TAB 1: SINGLE PREDICTION
        # ============================================
        with gr.TabItem("🔍 Single Employee Prediction"):
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Employee Information")
                    
                    model_selector = gr.Dropdown(
                        choices=["Logistic Regression (Conservative)", "Decision Tree (Aggressive)"],
                        value="Logistic Regression (Conservative)",
                        label="Select Model"
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("#### 😊 Satisfaction Levels (1=Low, 4=High)")
                    
                    job_satisfaction = gr.Slider(1, 4, value=3, step=1, label="Job Satisfaction")
                    job_involvement = gr.Slider(1, 4, value=3, step=1, label="Job Involvement")
                    environment_satisfaction = gr.Slider(1, 4, value=3, step=1, label="Environment Satisfaction")
                    work_life_balance = gr.Slider(1, 4, value=3, step=1, label="Work-Life Balance")
                    
                    gr.Markdown("#### 💼 Work Experience")
                    
                    num_companies_worked = gr.Slider(0, 9, value=2, step=1, label="Number of Companies Worked")
                    total_working_years = gr.Slider(0, 40, value=10, step=1, label="Total Working Years")
                    years_at_company = gr.Slider(0, 40, value=5, step=1, label="Years at Current Company")
                    
                    gr.Markdown("#### 🏢 Job Details")
                    
                    department = gr.Dropdown(
                        ["Sales", "Research & Development", "Human Resources"],
                        value="Research & Development",
                        label="Department"
                    )
                    
                    # Only the 6 job roles that matter in the model
                    job_role = gr.Dropdown(
                        ["Research Director", "Sales Representative", "Laboratory Technician",
                         "Manufacturing Director", "Healthcare Representative", "Manager"],
                        value="Laboratory Technician",
                        label="Job Role"
                    )
                    
                    business_travel = gr.Dropdown(
                        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
                        value="Travel_Rarely",
                        label="Business Travel Frequency"
                    )
                    
                    marital_status = gr.Dropdown(
                        ["Single", "Married", "Divorced"],
                        value="Married",
                        label="Marital Status"
                    )
                    
                    overtime = gr.Dropdown(
                        ["No", "Yes"],
                        value="No",
                        label="Works Overtime?"
                    )
                    
                    predict_btn = gr.Button("🎯 Predict Attrition Risk", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Results")
                    
                    proba_output = gr.Number(label="Attrition Probability", precision=4)
                    prediction_output = gr.Textbox(label="Prediction")
                    risk_output = gr.Textbox(label="Risk Level")
                    explanation_output = gr.Markdown()
            
            # Connect button
            predict_btn.click(
                fn=predict_attrition,
                inputs=[
                    job_satisfaction, job_involvement, environment_satisfaction, work_life_balance,
                    num_companies_worked, total_working_years, years_at_company,
                    department, job_role, business_travel, marital_status, overtime,
                    model_selector
                ],
                outputs=[proba_output, prediction_output, risk_output, explanation_output]
            )
        
        # ============================================
        # TAB 2: BATCH PREDICTION
        # ============================================
        with gr.TabItem("📁 Batch Prediction (CSV Upload)"):
            
            gr.Markdown("""
            ### Upload CSV File with Employee Data
            
            **Required columns (12 raw columns → 18 features after transformation):**
            
            **Numeric (raw values):**
            - `JobSatisfaction`, `JobInvolvement`, `EnvironmentSatisfaction`, `WorkLifeBalance` (1-4)
            - `NumCompaniesWorked` (0-9)
            - `TotalWorkingYears`, `YearsAtCompany` (actual years - will be log-transformed)
            
            **Categorical:**
            - `Department`: Sales, Research & Development, Human Resources
            - `JobRole`: Research Director, Sales Representative, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager
            - `BusinessTravel`: Travel_Rarely, Travel_Frequently, Non-Travel
            - `MaritalStatus`: Single, Married, Divorced
            - `OverTime`: Yes, No
            
            **Note:** System automatically handles log transformations, scaling, and one-hot encoding.
            """)
            
            with gr.Row():
                with gr.Column():
                    csv_file = gr.File(label="Upload CSV File", file_types=[".csv"])
                    batch_model = gr.Dropdown(
                        ["Logistic Regression (Conservative)", "Decision Tree (Aggressive)"],
                        value="Logistic Regression (Conservative)",
                        label="Select Model"
                    )
                    batch_btn = gr.Button("📊 Predict All Employees", variant="primary", size="lg")
                
                with gr.Column():
                    batch_summary = gr.Markdown()
                    batch_results = gr.Dataframe(label="Prediction Results")
            
            batch_btn.click(
                fn=predict_batch,
                inputs=[csv_file, batch_model],
                outputs=[batch_summary, batch_results]
            )
        
        # ============================================
        # TAB 3: MODEL INFORMATION
        # ============================================
        with gr.TabItem("ℹ️ Model Information"):
            gr.Markdown("""
            ## 📚 Model Performance (18-Feature Subset)
            
            ### 🎯 Logistic Regression (Conservative)
            - **Threshold:** 0.400 (recall-optimized)
            - **Validation Performance:** 58% precision, 77% recall
            - **Test AUC:** 0.808
            - **Best for:** General HR planning, balanced predictions
            
            ### 🌳 Decision Tree (Aggressive)
            - **Threshold:** 0.320 (recall-optimized)
            - **Validation Performance:** 25% precision, 79% recall
            - **Test AUC:** 0.724
            - **Best for:** Critical roles, proactive retention programs
            
            ---
            
            ## 🔑 18 Key Features Used
            
            **After VIF analysis and feature selection:**
            
            1. **Satisfaction Metrics (4):** JobSatisfaction, JobInvolvement, EnvironmentSatisfaction, WorkLifeBalance
            2. **Experience (3):** NumCompaniesWorked, TotalWorkingYearsLog, YearsAtCompanyLog
            3. **Department (1):** DepartmentResearch & Development
            4. **Job Roles (6):** Research Director, Sales Representative, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager
            5. **Travel (1):** BusinessTravelTravel_Frequently
            6. **Personal (2):** MaritalStatusSingle, MaritalStatusMarried
            7. **Overtime (1):** OverTimeYes
            
            **Excluded features:** EducationField, Gender, Age, MonthlyIncome (due to multicollinearity/low importance)
            
            ---
            
            ## 🎚️ Threshold Strategy
            
            Both models use **recall-optimized thresholds** to minimize missed departures:
            - Prioritizes catching potential leavers (70%+ recall)
            - Accepts more false alarms (lower precision)
            - Business rationale: Missing a leaver costs more than investigating a false alarm
            
            ---
            
            ## 📊 Risk Level Guidelines
            
            | Probability | Risk Level | Action Required |
            |------------|------------|-----------------|
            | < 30% | 🟢 Low | Regular engagement |
            | 30-50% | 🟡 Medium | Career discussion |
            | 50-70% | 🟠 High | Retention interview |
            | ≥ 70% | 🔴 Critical | Immediate intervention |
            
            ---
            
            ## 📥 CSV Format Example
            
            ```
            JobSatisfaction,JobInvolvement,EnvironmentSatisfaction,WorkLifeBalance,NumCompaniesWorked,TotalWorkingYears,YearsAtCompany,Department,JobRole,BusinessTravel,MaritalStatus,OverTime
            3,3,2,3,2,10,5,Research & Development,Laboratory Technician,Travel_Rarely,Married,Yes
            2,2,3,2,5,15,3,Sales,Sales Representative,Travel_Frequently,Single,No
            ```
            
            **All transformations are automatic!** Just provide raw values.
            """)

# ============================================
# 5. LAUNCH APP
# ============================================

if __name__ == "__main__":
    demo.launch()
