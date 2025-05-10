import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Kidney Disease Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #FAFAFA; }
    .stButton>button {
        color: white;
        background-color: #1f77b4;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Kidney Disease - Blood Glucose Predictor (bgr)")

@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")
    df.dropna(thresh=len(df.columns) - 3, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df.drop(columns=[col for col in ['id', 'classification'] if col in df.columns], inplace=True)
    df.dropna(subset=['bgr'], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df

df = load_data()
X = df.drop(columns=['bgr'])
y = df['bgr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)

tab1, tab2, tab3 = st.tabs(["📈 Model Overview", "📊 Visualizations", "🔍 Make Predictions"])

with tab1:
    st.header("Model Evaluation")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R-squared Score (R²)", f"{r2:.2f}", f"{r2 * 100:.2f}%" )

    st.subheader("Top Features Affecting Blood Glucose")
    st.dataframe(
        feature_importance.head(10).rename("Coefficient").to_frame()
        .style.background_gradient(cmap="coolwarm")
    )

    st.subheader("Sample of Cleaned Dataset")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

with tab2:
    st.header("Visual Analysis")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis', ax=ax1)
    ax1.set_title("Feature Importance (Linear Regression Coefficients)", color='black')
    ax1.set_facecolor('#F0EAF6')
    fig1.patch.set_facecolor('#F0EAF6')
    st.pyplot(fig1)

    st.markdown("---")

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("Correlation Heatmap", color='black')
    ax2.set_facecolor('#F0EAF6')
    fig2.patch.set_facecolor('#F0EAF6')
    st.pyplot(fig2)

with tab3:
    st.header("Predict Blood Glucose Level (bgr)")

    user_input = {}
    col1, col2 = st.columns(2)

    for idx, col in enumerate(X.columns):
        with col1 if idx % 2 == 0 else col2:
            if len(X[col].unique()) <= 10:
                user_input[col] = st.selectbox(f"{col}", sorted(X[col].unique()))
            else:
                user_input[col] = st.slider(
                    f"{col}",
                    min_value=float(X[col].min()),
                    max_value=float(X[col].max()),
                    value=float(X[col].mean())
                )

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.success(f"🧾 Predicted Blood Glucose Level: **{prediction:.2f}**")

            csv = input_df.copy()
            csv["Predicted_bgr"] = prediction
            csv_file = csv.to_csv(index=False)

            st.download_button(
                label="📥 Download Prediction as CSV",
                data=csv_file,
                file_name="bgr_prediction.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred while predicting: {e}")

st.markdown("---")
st.markdown("Created with ❤️ using Streamlit | Author: *MAHESH P* |")
