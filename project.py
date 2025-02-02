import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import io

import warnings
warnings.filterwarnings("ignore")

st.title("AI Impact on Software Development Industry and Job Transformation")

# Upload File
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Load Data Button
load_data_button = st.button("Load Data")

if load_data_button and uploaded_file is not None:
    # Load Data
    data = pd.read_excel(uploaded_file)
    data = data.iloc[:, 1:]  # Drop index column

    # Show basic details about the uploaded dataset
    st.write("### Preview of Dataset:")
    st.write(data.head())

    # Basic EDA
    st.write("## Basic Exploratory Data Analysis (EDA)")

    st.write("### Dataset Shape:")
    st.write(f"Number of Rows: {data.shape[0]}")
    st.write(f"Number of Columns: {data.shape[1]}")

    # Dataset Information
    st.write("### Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    # Check for Null Values
    st.write("### Null Values in the Dataset:")
    null_values = data.isnull().sum()
    st.write(null_values)

    # Check for Duplicates
    st.write("### Duplicate Rows in the Dataset:")
    duplicate_rows = data.duplicated().sum()
    st.write(f"Number of Duplicate Rows: {duplicate_rows}")

    # Dataset Columns
    st.write("### Dataset Columns:")
    st.write(data.columns.tolist())

    # Data Cleaning & Processing
    st.write("## Data Cleaning & Processing")

    def generate_short_column_name(full_column_name):
        short_name_map = {
            "Your age?": "Age",
            "How familiar are you with AI technologies?": "AI_Familiarity",
            "What is your current role in the software development industry?  ": "Current_Role",
            "How many years of experience do you have in the software industry?  ": "Years_Exp",
            "Have you used AI tools (e.g., GitHub Copilot, ChatGPT, AI-based code generation, or testing tools)?  ": "Used_AI_Tools",
            "If yes, which tasks do you use AI tools for? (Select all that apply)": "AI_Tasks_Usage",
            "How has the use of AI tools impacted your productivity?": "AI_Impact_Productivity",
            "What are the main benefits you’ve experienced using AI tools? (Select all that apply)": "AI_Benefits",
            "Has your organization provided any training programs to prepare employees for AI-driven changes?": "AI_Training_Org",
            "How prepared do you feel for the ongoing transformation driven by AI in the software industry?": "AI_Preparedness",
            "In your opinion, what is the most significant challenge in adapting to AI-driven transformations in the software industry?": "AI_Challenge",
            "Do you believe AI will lead to job displacement in the software development industry?": "AI_Job_Displacement",
            "Have you observed any new roles or job opportunities emerging because of AI in your field?": "New_AI_Roles",
            "Which roles do you think are most vulnerable to automation by AI? (Select all that apply)": "Vulnerable_Roles_AI",
            "Which new skills do you think software professionals need to remain relevant in the AI-driven landscape? (Select all that apply)": "New_Skills_Required",
            "What is your overall opinion on the integration of AI in software development?": "AI_Opinion",
        }
        return short_name_map.get(full_column_name, full_column_name[:10])

    # Before Encoding Data
    st.write("### Before Encoding Data:")
    st.write(data.head())

    # Clean and preprocess the dataset
    data.columns = [generate_short_column_name(col) for col in data.columns]
    data["AI_Tasks_Usage"] = data["AI_Tasks_Usage"].str.split(';').str[0]
    mode_value = data["AI_Tasks_Usage"].mode()[0]
    data["AI_Tasks_Usage"].fillna(mode_value, inplace=True)
    data["AI_Benefits"] = data["AI_Benefits"].str.split(';').str[0]
    data["Vulnerable_Roles_AI"] = data["Vulnerable_Roles_AI"].str.split(';').str[0]
    data["New_Skills_Required"] = data["New_Skills_Required"].str.split(';').str[0]
    data["AI_Familiarity"] = data["AI_Familiarity"].str.split('(').str[0]

    # After Encoding Data
    st.write("### After Encoding Data:")
    st.write(data.head())

    # Data Preprocessing
    def preprocess_data(df):
        encoded_data = pd.DataFrame()
        label_mappings = {}

        for column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[column].fillna("Unknown"))
                encoded_data[column] = encoded_values
                label_mappings[column] = dict(zip(range(len(le.classes_)), le.classes_))
            else:
                encoded_data[column] = df[column]

        return encoded_data, label_mappings

    # Encode the data
    st.write("### Encoding Data:")
    encoded_data, label_mappings = preprocess_data(data)
    st.write(encoded_data.head())

    # EDA: Visualizations
    st.write("## Advanced Exploratory Data Analysis")

    if st.checkbox("Show Dataset Shape"):
        st.write("### Dataset Shape:")
        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")

    if st.checkbox("Show Dataset Information"):
        st.write("### Dataset Information:")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

    if st.checkbox("Show Null Values in the Dataset"):
        st.write("### Null Values in the Dataset:")
        null_values = data.isnull().sum()
        st.write(null_values)

    if st.checkbox("Show Duplicate Rows in the Dataset"):
        st.write("### Duplicate Rows in the Dataset:")
        duplicate_rows = data.duplicated().sum()
        st.write(f"Number of Duplicate Rows: {duplicate_rows}")

    if st.checkbox("Show Dataset Columns"):
        st.write("### Dataset Columns:")
        st.write(data.columns.tolist())

    # Correlation Matrix with Plotly
    if st.checkbox("Show interactive correlation matrix"):
        corr_matrix = encoded_data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis'
        ))
        fig.update_layout(title="Interactive Correlation Matrix")
        st.plotly_chart(fig)

    # Histograms with Plotly
    if st.checkbox("Show interactive histogram for numerical features"):
        num_cols = encoded_data.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig = px.histogram(encoded_data, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig)

    # Count Plots for Categorical Features with Plotly
    if st.checkbox("Show interactive count plots for categorical features"):
        cat_cols = data.select_dtypes(include="object").columns
        for col in cat_cols:
            fig = px.bar(data[col].value_counts(), title=f"Count Plot of {col}")
            st.plotly_chart(fig)

    # Box Plots with Plotly
    if st.checkbox("Show interactive box plots for numerical features"):
        num_cols = encoded_data.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig = px.box(encoded_data, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig)

    # Violin Plots with Plotly
    if st.checkbox("Show interactive violin plots for numerical features"):
        num_cols = encoded_data.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig = px.violin(encoded_data, y=col, box=True, points="all", title=f"Violin Plot of {col}")
            st.plotly_chart(fig)

    # Feature Importance (Random Forest)
    target_column = st.selectbox("Select target column for feature importance", encoded_data.columns)
    if st.button("Show feature importance"):
        X = encoded_data.drop(columns=[target_column])
        y = encoded_data[target_column]

        model = RandomForestClassifier()
        model.fit(X, y)

        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        fig = px.bar(feature_importance, x=feature_importance.values, y=feature_importance.index, title="Feature Importance (Random Forest)")
        st.plotly_chart(fig)

    # PCA for Dimensionality Reduction
    if st.checkbox("Show PCA for dimensionality reduction"):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(encoded_data)
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA of Dataset")
        st.plotly_chart(fig)

    # t-SNE for Dimensionality Reduction
    if st.checkbox("Show t-SNE for dimensionality reduction"):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(encoded_data)
        tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
        fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', title="t-SNE of Dataset")
        st.plotly_chart(fig)

    # Hypothesis Testing (Fixed)
    st.write("## Hypothesis Testing")

    test_type = st.selectbox("Select Hypothesis Test", ["Chi-Square", "T-Test"])

    if test_type == "Chi-Square":
        col1 = st.selectbox("Select first categorical variable", encoded_data.columns)
        col2 = st.selectbox("Select second categorical variable", encoded_data.columns)

        contingency_table = pd.crosstab(encoded_data[col1], encoded_data[col2])
        chi2, p, _, _ = stats.chi2_contingency(contingency_table)

        st.write(f"Chi-Square Test: p-value = {p}")
        if p < 0.05:
            st.write("**Result:** Reject null hypothesis (Significant relationship exists)")
        else:
            st.write("**Result:** Fail to reject null hypothesis (No significant relationship)")

    elif test_type == "T-Test":
        col = st.selectbox("Select numerical variable", encoded_data.columns)
        group1 = encoded_data[col][encoded_data[col] > encoded_data[col].median()]
        group2 = encoded_data[col][encoded_data[col] <= encoded_data[col].median()]

        t_stat, p = stats.ttest_ind(group1, group2)

        st.write(f"T-Test: p-value = {p}")
        if p < 0.05:
            st.write("**Result:** Reject null hypothesis (Significant difference found)")
        else:
            st.write("**Result:** Fail to reject null hypothesis (No significant difference)")

    # Machine Learning
    st.write("## Train Machine Learning Model")

    target_column = st.selectbox("Select the target column", encoded_data.columns)

    if st.button("Train Model"):
        X = encoded_data.drop(columns=[target_column])
        y = encoded_data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = st.selectbox("Select model type", ["Random Forest", "Logistic Regression", "Naive Bayes", "Decision Tree"])

        if model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Naive Bayes":
            model = GaussianNB()
        else:
            model = DecisionTreeClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Evaluation:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

else:
    st.warning("Please click the 'Load Data' button to proceed.")
