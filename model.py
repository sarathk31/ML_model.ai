import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# For Ai Insights
from analysis import generate_summary, generate_improvement_suggestions as suggest_improvements
# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(page_title="ML & AI Insights", layout="wide")
st.title("Machine Learning & AI Insights App")
st.subheader("Explore, Analyze, and Visualize Your Data with ML Models")

File = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="file_uploader") 

if File is not None:
    df = pd.read_csv(File)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    target = st.selectbox("Select Target Variable", options=df.columns, key="target_select")
    
    if target:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()
        
        # Pre Processing
        num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna('Missing')
        
        # Encoding Categorical Variables
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        
        if len(df[target].unique()) < 15:
            problem_type = 'Classification'
        else:
            problem_type = 'Regression'
            
        st.write(f"Detected Problem Type: {problem_type}")
        
        #Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler=StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
        # ========
        # MODELS
        # ========
        results=[]
        
        if problem_type == 'Regression':
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
            }
            
            for name,model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({'Model Name': name, 'R2 Score': round(r2_score(y_test, y_pred), 4), 'RMSE': round(mean_squared_error(y_test, y_pred), 4)})
            
        else: 
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                 "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)}
            for name,model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                    
                results.append({'Model Name': name, 
                                'Accuracy': round(accuracy_score(y_test, y_pred), 4), 
                                'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4), 
                                'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4), 
                                'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4)})
        
        
        results_df = pd.DataFrame(results)
        st.write("Model Performance Comparison:")
        st.dataframe(results_df)
        
        if problem_type == 'Regression':
            st.bar_chart(results_df.set_index('Model Name')[['R2 Score', 'RMSE']])
        
        else:
            st.bar_chart(results_df.set_index('Model Name')[['Accuracy', 'Precision', 'Recall', 'F1 Score']])   
        
        #============
        # AI Insights
        #============
        
        if st.button(":blue[Generate Summary]"):
            summary = generate_summary(results_df)
            st.write(summary)

        if st.button(":blue[Suggest Improvements]"):
            improve = suggest_improvements(results_df)
            st.write(improve)

        # DOWNLOAD
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV here", csv, "model_results.csv")       