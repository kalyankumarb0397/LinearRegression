import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set up Streamlit app title
st.title("Real Estate Valuation: Linear Regression with Standard Scaling")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Display dataset information
    st.subheader("Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("Columns:", list(df.columns))

    # User selects features and target variable
    st.subheader("Select Features and Target")
    target_column = st.selectbox("Select the target column (Y)", df.columns)
    feature_columns = st.multiselect("Select feature columns (X)", [col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        # Prepare the data
        X = df[feature_columns]
        y = df[target_column]

        # Train-test split
        st.subheader("Train-Test Split")
        test_size = st.slider("Select test size percentage", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Standard Scaling
        st.subheader("Standard Scaling")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.write("Feature scaling applied using StandardScaler.")

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Model evaluation
        st.subheader("Model Evaluation")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Display coefficients
        st.subheader("Model Coefficients")
        coefficients = pd.DataFrame({
            "Feature": feature_columns,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)
        st.write(coefficients)

        # Scatter plot: Actual vs Predicted
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to get started.")
