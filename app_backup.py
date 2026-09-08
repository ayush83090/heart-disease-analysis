import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/heart_cleaned.csv")


@st.cache_resource
def load_model():
    model = joblib.load("models/heart_disease_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


df = load_data()
model, scaler = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("❤️ Heart Disease Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Prediction",
        "SQL Analytics"
    ]
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("❤️ Heart Disease Analytics Dashboard")

    st.write(
        "Interactive analysis of patient data using Python, "
        "SQL and Machine Learning."
    )

    st.divider()

    # -------------------------------
    # KPI CARDS
    # -------------------------------

    total_patients = len(df)

    disease_cases = int(df["target"].sum())

    disease_rate = disease_cases / total_patients * 100

    average_age = df["age"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Patients",
        total_patients
    )

    col2.metric(
        "Heart Disease Cases",
        disease_cases
    )

    col3.metric(
        "Disease Rate",
        f"{disease_rate:.1f}%"
    )

    col4.metric(
        "Average Age",
        f"{average_age:.1f}"
    )

    st.divider()

    # -------------------------------
    # DISEASE DISTRIBUTION
    # -------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Heart Disease Distribution")

        disease_data = pd.DataFrame({
            "Outcome": [
                "No Heart Disease",
                "Heart Disease"
            ],
            "Patients": [
                int((df["target"] == 0).sum()),
                int((df["target"] == 1).sum())
            ]
        })

        fig = px.pie(
            disease_data,
            names="Outcome",
            values="Patients",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------
    # AGE DISTRIBUTION
    # -------------------------------

    with col2:

        st.subheader("Age Distribution")

        fig = px.histogram(
            df,
            x="age",
            color="target",
            nbins=20,
            labels={
                "target": "Heart Disease"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------
    # CHOLESTEROL
    # -------------------------------

    st.subheader("Cholesterol vs Heart Disease")

    fig = px.box(
        df,
        x="target",
        y="chol",
        labels={
            "target": "Heart Disease",
            "chol": "Cholesterol"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------
    # MAX HEART RATE
    # -------------------------------

    st.subheader("Maximum Heart Rate by Outcome")

    heart_rate = (
        df.groupby("target")["thalach"]
        .mean()
        .reset_index()
    )

    heart_rate["target"] = heart_rate["target"].map({
        0: "No Heart Disease",
        1: "Heart Disease"
    })

    fig = px.bar(
        heart_rate,
        x="target",
        y="thalach",
        labels={
            "target": "Outcome",
            "thalach": "Average Maximum Heart Rate"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.title("🤖 Heart Disease Prediction")

    st.write(
        "Enter patient information to generate a model prediction."
    )

    st.warning(
        "This is an educational machine-learning project and "
        "is not a medical diagnostic tool."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=20,
            max_value=100,
            value=55
        )

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=220,
            value=130
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=240
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [1, 2, 3]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7]
        )

    st.divider()

    predict_button = st.button(
        "🔍 Predict",
        type="primary"
    )

    if predict_button:

        sex_value = 1 if sex == "Male" else 0

        cp_value = {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-anginal Pain": 3,
            "Asymptomatic": 4
        }[cp]

        fbs_value = 1 if fbs == "Yes" else 0

        exang_value = 1 if exang == "Yes" else 0

        input_data = pd.DataFrame([[
            age,
            sex_value,
            cp_value,
            trestbps,
            chol,
            fbs_value,
            restecg,
            thalach,
            exang_value,
            oldpeak,
            slope,
            ca,
            thal
        ]], columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.divider()

        if prediction == 1:

            st.error(
                "Model prediction: Heart Disease"
            )

        else:

            st.success(
                "Model prediction: No Heart Disease"
            )

        st.metric(
            "Predicted Probability",
            f"{probability * 100:.2f}%"
        )


# ============================================================
# SQL ANALYTICS PAGE
# ============================================================

elif page == "SQL Analytics":

    st.title("🗄️ SQL Analytics")

    st.write(
        "Analytics generated directly from the SQLite database."
    )

    connection = sqlite3.connect(
        "data/heart_disease.db"
    )

    # -------------------------------
    # Disease distribution
    # -------------------------------

    query1 = """
    SELECT
        target,
        COUNT(*) AS patient_count
    FROM patients
    GROUP BY target;
    """

    disease_result = pd.read_sql_query(
        query1,
        connection
    )

    disease_result["target"] = disease_result["target"].map({
        0: "No Heart Disease",
        1: "Heart Disease"
    })

    st.subheader("Heart Disease Distribution")

    st.dataframe(
        disease_result,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------
    # Age group analysis
    # -------------------------------

    query2 = """
    SELECT
        pg.age_group,
        COUNT(p.patient_id) AS total_patients,
        SUM(p.target) AS heart_disease_cases,
        ROUND(
            100.0 * SUM(p.target)
            / COUNT(p.patient_id),
            2
        ) AS disease_rate
    FROM patients p
    JOIN patient_groups pg
        ON p.patient_id = pg.patient_id
    GROUP BY pg.age_group
    ORDER BY disease_rate DESC;
    """

    age_result = pd.read_sql_query(
        query2,
        connection
    )

    st.subheader(
        "Heart Disease Rate by Age Group"
    )

    st.dataframe(
        age_result,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------
    # Chest pain analysis
    # -------------------------------

    query3 = """
    SELECT
        cp,
        COUNT(*) AS total_patients,
        SUM(target) AS heart_disease_cases
    FROM patients
    GROUP BY cp
    ORDER BY heart_disease_cases DESC;
    """

    chest_result = pd.read_sql_query(
        query3,
        connection
    )

    chest_result["cp"] = chest_result["cp"].map({
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    })

    st.subheader(
        "Heart Disease Cases by Chest Pain Type"
    )

    st.dataframe(
        chest_result,
        use_container_width=True,
        hide_index=True
    )

    connection.close()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Heart Disease Analytics | Python • SQL • Machine Learning"
)