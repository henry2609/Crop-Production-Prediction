import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("FAOSTAT_data_cleaned.csv")
    df.columns = df.columns.str.strip()
    df = df[df['Metric'].isin(['Area harvested', 'Yield', 'Production'])]
    df = df.pivot_table(index=['Country', 'Item', 'Year'], 
                        columns='Metric', values='Value').reset_index()
    df = df.dropna(subset=['Area harvested', 'Yield', 'Production'])
    return df

df = load_data()

# Title
st.title("🌾 Crop Production Prediction App with Model Comparison")

# Sidebar filters
country = st.sidebar.selectbox("Select Country", sorted(df["Country"].unique()))
crop = st.sidebar.selectbox("Select Crop", sorted(df["Item"].unique()))

# Filter data
filtered_df = df[(df["Country"] == country) & (df["Item"] == crop)]

if filtered_df.empty:
    st.warning("No data available for the selected combination.")
    st.stop()

# Display data
st.subheader(f"📊 Historical Data for {crop} in {country}")
st.dataframe(filtered_df[['Year', 'Area harvested', 'Yield', 'Production']].sort_values('Year'))

# Features & target
X = filtered_df[['Area harvested', 'Yield']]
y = filtered_df['Production']

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# Model evaluation
st.subheader("📈 Model Evaluation")
results = []
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    results.append({
        "Model": name,
        "R² Score": r2_score(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred)
    })

result_df = pd.DataFrame(results).set_index("Model")
st.table(result_df.style.format({"R² Score": "{:.3f}", "MAE": "{:.2f}", "MSE": "{:.2f}"}))

# Let user pick a model
selected_model_name = st.selectbox("Select model for prediction", list(models.keys()))
selected_model = models[selected_model_name]
selected_model.fit(X, y)

# Prediction input
st.subheader("🧮 Make a Production Prediction")
default_area = float(filtered_df['Area harvested'].mean())
default_yield = float(filtered_df['Yield'].mean())

area_input = st.number_input("Area harvested (ha)", value=default_area, format="%.2f")
yield_input = st.number_input("Yield (kg/ha)", value=default_yield, format="%.2f")

prediction = selected_model.predict([[area_input, yield_input]])[0]
st.success(f"{selected_model_name} Prediction: **{prediction:,.2f} tons**")

# Plot production trend
st.subheader("📊 Production Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_df['Year'], filtered_df['Production'], label='Actual', marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Production (tons)")
ax.set_title(f"{crop} Production Trend in {country}")
ax.legend()
st.pyplot(fig)

# Export predictions
st.subheader("⬇️ Export Prediction")
if st.button("Download as CSV"):
    export_df = pd.DataFrame({
        "Country": [country],
        "Crop": [crop],
        "Area harvested (ha)": [area_input],
        "Yield (kg/ha)": [yield_input],
        "Predicted Production (tons)": [prediction],
        "Model Used": [selected_model_name]
    })
    st.download_button("📁 Download CSV", data=export_df.to_csv(index=False), file_name="crop_production_prediction.csv", mime="text/csv")
