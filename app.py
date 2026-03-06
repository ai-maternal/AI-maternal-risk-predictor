import streamlit as st
import pandas as pd
import joblib
import sqlite3
import hashlib

# Explainability + plotting
import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# PDF report (with graph + metadata)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Maternal Risk AI", page_icon="🤰", layout="wide")

# -----------------------------
# Local assets (banner)
# -----------------------------
HERO_IMAGE_PATH = os.path.join("assets", "maternal_banner.jpg")

# -----------------------------
# UI theme + Hide sidebar
# -----------------------------
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      [data-testid="stSidebarNav"] {display: none !important;}
      section[data-testid="stSidebar"] {display: none !important;}
      button[kind="header"] {display: none !important;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .stApp {
        background:
          radial-gradient(1100px 550px at 10% 10%, rgba(255, 95, 162, 0.14), rgba(0,0,0,0)),
          radial-gradient(1100px 550px at 90% 15%, rgba(138, 92, 255, 0.14), rgba(0,0,0,0)),
          radial-gradient(900px 450px at 55% 95%, rgba(0, 179, 179, 0.10), rgba(0,0,0,0)),
          linear-gradient(180deg, #0b1020 0%, #070a12 100%);
        color: #f4f2ff;
      }
      html, body, [class*="css"]  { color: #f4f2ff !important; }

      .main .block-container {
        padding-top: 1.0rem;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px;
      }

      .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin: 10px 0 18px 0;
      }
      .muted { color: rgba(244,242,255,0.74); font-size: 0.95rem; }

      .hero {
        border-radius: 22px;
        padding: 18px 18px;
        background: linear-gradient(135deg, rgba(255,95,162,0.18), rgba(138,92,255,0.12));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 35px rgba(0,0,0,0.35);
        margin-bottom: 16px;
      }

      div.stButton > button {
        background: linear-gradient(90deg, #ff5fa2 0%, #8a5cff 100%);
        color: white; border: none; border-radius: 14px;
        padding: 0.65rem 1.1rem; font-weight: 850;
        box-shadow: 0 10px 25px rgba(255, 95, 162, 0.18);
      }
      div.stButton > button:hover { filter: brightness(1.06); transform: translateY(-1px); }

      div.stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.07);
        color: #f4f2ff;
      }

      [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 12px 14px;
        border-radius: 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Settings
# -----------------------------
DB_PATH = "users.db"
MODEL_PATH = "maternal_risk_model.pkl"
COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
ADMIN_USERS = {"admin"}

# -----------------------------
# Helpers
# -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Model Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def get_page():
    try:
        qp = st.query_params
        return qp.get("page", "home")
    except Exception:
        qp = st.experimental_get_query_params()
        return qp.get("page", ["home"])[0]

def set_page(page: str):
    try:
        st.query_params["page"] = page
    except Exception:
        st.experimental_set_query_params(page=page)

def is_admin_user() -> bool:
    u = st.session_state.get("username")
    return bool(u) and u.strip().lower() in ADMIN_USERS

def ensure_column_exists(cur, table: str, column: str, coltype: str):
    cur.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            created_at TEXT,
            patient_name TEXT,
            patient_id TEXT,
            age REAL,
            systolicbp REAL,
            diastolicbp REAL,
            bs REAL,
            bodytemp REAL,
            heartrate REAL,
            risk_score REAL,
            category TEXT
        );
        """
    )

    ensure_column_exists(cur, "patient_history", "patient_name", "TEXT")
    ensure_column_exists(cur, "patient_history", "patient_id", "TEXT")

    conn.commit()
    conn.close()

def user_exists(username: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def create_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    pw_hash = hash_password(password)
    cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
    conn.commit()
    conn.close()

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return hash_password(password) == row[0]

def save_prediction_and_get_patient_id(
    username: str,
    created_at: str,
    patient_name: str,
    age, sbp, dbp, bs, temp, hr,
    risk_score: float,
    category: str
) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO patient_history
        (username, created_at, patient_name, patient_id, age, systolicbp, diastolicbp, bs, bodytemp, heartrate, risk_score, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (username, created_at, patient_name, None, age, sbp, dbp, bs, temp, hr, risk_score, category),
    )

    new_id = cur.lastrowid
    patient_id = f"PAT{new_id:06d}"

    cur.execute(
        "UPDATE patient_history SET patient_id = ? WHERE id = ?",
        (patient_id, new_id),
    )

    conn.commit()
    conn.close()
    return patient_id

def load_user_history(username: str, limit: int = 100) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT created_at, patient_name, patient_id, age, systolicbp, diastolicbp, bs, bodytemp, heartrate, risk_score, category
        FROM patient_history
        WHERE username = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(username, limit),
    )
    conn.close()
    return df

def load_all_history(limit: int = 500) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT username, created_at, patient_name, patient_id, age, systolicbp, diastolicbp, bs, bodytemp, heartrate, risk_score, category
        FROM patient_history
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )
    conn.close()
    return df

def search_patient_by_id(patient_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT username, created_at, patient_name, patient_id,
               age, systolicbp, diastolicbp, bs, bodytemp,
               heartrate, risk_score, category
        FROM patient_history
        WHERE patient_id = ?
        ORDER BY id DESC
        """,
        conn,
        params=(patient_id,),
    )
    conn.close()
    return df

def load_patient_trend(patient_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT created_at, patient_name, patient_id, risk_score, category
        FROM patient_history
        WHERE patient_id = ?
        ORDER BY created_at ASC
        """,
        conn,
        params=(patient_id,),
    )

    conn.close()
    return df

def get_shap_vector_for_class1(shap_values, n_features: int):
    if isinstance(shap_values, list):
        return np.array(shap_values[1]).reshape(-1)

    sv = np.array(shap_values)
    if sv.ndim == 3:
        if sv.shape[1] == 2 and sv.shape[2] == n_features:
            return sv[0, 1, :]
        if sv.shape[2] == 2 and sv.shape[1] == n_features:
            return sv[0, :, 1]
        return sv.reshape(-1)
    if sv.ndim == 2:
        return sv[0, :]
    return sv.reshape(-1)

# -----------------------------
# PDF builder
# -----------------------------
def build_pdf_report(
    *,
    patient_name: str,
    patient_id: str,
    risk_score: float,
    category: str,
    recommendation: str,
    confidence: float,
    inputs: dict,
    top3_feats: list,
    signed_series: pd.Series,
    username_for_meta: str,
    chart_png_bytes: bytes,
):
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="AI Maternal Risk Predictor Report",
        author="Maternal Risk AI",
    )

    def add_pdf_meta(c: canvas.Canvas, d):
        c.setTitle("AI Maternal Risk Predictor Report")
        c.setAuthor("Maternal Risk AI")
        c.setSubject("Maternal risk prediction with Explainable AI (SHAP)")
        c.setCreator(f"Maternal Risk AI • User: {username_for_meta}")

    story = []
    story.append(Paragraph("AI Maternal Risk Predictor Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}</i>", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles["Normal"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.2f} %", styles["Normal"]))
    story.append(Paragraph(f"<b>Category:</b> {category}", styles["Normal"]))
    story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles["Normal"]))
    story.append(Paragraph(f"<b>Model Confidence:</b> {confidence:.2f} %", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Patient Inputs</b>", styles["Heading2"]))
    table_data = [["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()]
    t = Table(table_data, hAlign="LEFT", colWidths=[260, 120])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ff5fa2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Top Factors (SHAP)</b>", styles["Heading2"]))
    for f in top3_feats:
        direction = "increases risk" if signed_series[f] > 0 else "decreases risk"
        story.append(Paragraph(f"- {f}: {direction} (SHAP: {signed_series[f]:.3f})", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>SHAP Impact Chart</b>", styles["Heading2"]))
    story.append(Image(BytesIO(chart_png_bytes), width=480, height=260))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Note: Higher bar = stronger influence on this prediction.", styles["Normal"]))

    doc.build(story, onFirstPage=add_pdf_meta, onLaterPages=add_pdf_meta)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# -----------------------------
# Init
# -----------------------------
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# -----------------------------
# Navbar buttons
# -----------------------------
def nav_btn(label, page_name):
    if st.button(label, key=f"nav_{page_name}"):
        set_page(page_name)
        st.rerun()

def navbar():
    cols = st.columns([3, 1.2, 1.4, 1.4, 1.4, 1.2, 1.2])

    with cols[0]:
        st.markdown("### **Maternal Risk AI**")

    with cols[1]:
        nav_btn("Home", "home")

    with cols[2]:
        nav_btn("Predictor", "predictor")

    with cols[3]:
        if st.session_state.logged_in and is_admin_user():
            nav_btn("Dashboard", "dashboard")
        else:
            st.write("")

    with cols[4]:
        if st.session_state.logged_in:
            nav_btn("My History", "history")
        else:
            st.write("")

    with cols[5]:
        if st.session_state.logged_in and is_admin_user():
            nav_btn("Admin", "admin")
        elif not st.session_state.logged_in:
            nav_btn("Sign Up", "signup")
        else:
            st.write("")

    with cols[6]:
        if st.session_state.logged_in:
            nav_btn("Logout", "logout")
        else:
            nav_btn("Login", "login")

    st.divider()

navbar()

# -----------------------------
# Pages
# -----------------------------
def home_page():
    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0">AI Maternal Mortality Risk Predictor</h1>
          <div class="muted" style="margin-top:8px; font-size: 1.02rem;">
            Predict high-risk pregnancies using antenatal parameters + Explainable AI (SHAP) + PDF report.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(HERO_IMAGE_PATH):
        st.image(HERO_IMAGE_PATH, width="stretch")
    else:
        st.warning(f"Banner image not found: {HERO_IMAGE_PATH}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>Fast Risk Screening</h3><div class="muted">Instant score + category using ML.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>Explainable AI</h3><div class="muted">Top SHAP factors with direction.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>History Tracking</h3><div class="muted">Save predictions & view later.</div></div>', unsafe_allow_html=True)

    st.info("Login → Predictor → Predict → Download PDF and view History.")

def login_page():
    st.markdown('<div class="card"><h2>Login</h2><div class="muted">Access the predictor dashboard.</div></div>', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        u = username.strip()
        if u == "" or password == "":
            st.error("Please fill username and password.")
        elif verify_user(u, password):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.success("Logged in successfully")
            set_page("predictor")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.write("No account? Go to **Sign Up** page.")
    if st.button("Go to Sign Up"):
        set_page("signup")
        st.rerun()

def signup_page():
    st.markdown('<div class="card"><h2>Sign Up</h2><div class="muted">Create a user account to access the predictor.</div></div>', unsafe_allow_html=True)
    new_user = st.text_input("Choose a username")
    new_pass = st.text_input("Choose a password", type="password")
    new_pass2 = st.text_input("Confirm password", type="password")

    if st.button("Create Account"):
        u = new_user.strip()
        if len(u) < 3:
            st.error("Username must be at least 3 characters.")
            return
        if len(new_pass) < 4:
            st.error("Password must be at least 4 characters.")
            return
        if new_pass != new_pass2:
            st.error("Passwords do not match.")
            return
        if user_exists(u):
            st.error("Username already exists. Try another.")
            return
        create_user(u, new_pass)
        st.success("Account created. Please login now.")
        set_page("login")
        st.rerun()

def logout_action():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("Logged out")
    set_page("home")
    st.rerun()

def predictor_page():
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        set_page("login")
        st.rerun()

    st.markdown(
        f"""
        <div class="hero">
          <h2 style="margin:0">Maternal Risk Predictor Dashboard</h2>
          <div class="muted">Logged in as <b>{st.session_state.username}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)

    left, right = st.columns([1, 1])

    # ---------------- LEFT SIDE ----------------
    with left:
        st.markdown(
            '<div class="card"><h3>Patient Details</h3><div class="muted">Select patient type</div></div>',
            unsafe_allow_html=True
        )

        patient_type = st.radio(
            "Patient Type",
            ["New Patient", "Returning Patient"],
            key="patient_type"
        )

        if patient_type == "New Patient":
            patient_name = st.text_input(
                "Patient Name",
                placeholder="Example: Anitha",
                key="patient_name_new"
            )
            existing_patient_id = None
            st.caption("Patient ID will be generated automatically")

        else:
            existing_patient_id = st.text_input(
                "Enter Existing Patient ID",
                placeholder="Example: PAT000021",
                key="existing_patient_id"
            )
            patient_name = "Returning Patient"

        st.markdown(
            '<div class="card"><h3>Clinical Inputs</h3><div class="muted">Enter values and predict risk.</div></div>',
            unsafe_allow_html=True
        )

        age = st.number_input("Age (years)", min_value=10, max_value=60, value=25)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        bs = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=7.0)
        temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.0)
        hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=140, value=80)

        predict = st.button("Predict Risk")

    # ---------------- RIGHT SIDE ----------------
    with right:
        st.markdown(
            '<div class="card"><h3>Results</h3><div class="muted">Risk score + SHAP + PDF report</div></div>',
            unsafe_allow_html=True
        )

        if predict:
            if patient_type == "New Patient" and patient_name.strip() == "":
                st.error("Please enter Patient Name.")
                return

            if patient_type == "Returning Patient" and existing_patient_id.strip() == "":
                st.error("Please enter Existing Patient ID.")
                return

            input_data = pd.DataFrame([[age, sbp, dbp, bs, temp, hr]], columns=COLUMNS)
            prob = model.predict_proba(input_data)[0][1]
            risk_score = float(prob * 100)
            confidence = float(max(prob, 1 - prob) * 100)

            if risk_score < 30:
                category = "Low Risk"
                recommendation = "Routine monitoring"
                st.success(category)
            elif risk_score < 70:
                category = "Moderate Risk"
                recommendation = "Please consult a doctor for further evaluation and treatment."
                st.warning(category)
            else:
                category = "High Risk"
                recommendation = "Patient identified as high risk and should undergo immediate medical evaluation and necessary diagnostic tests."
                st.error(category)

            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if patient_type == "New Patient":
                patient_id = save_prediction_and_get_patient_id(
                    st.session_state.username,
                    created_at,
                    patient_name.strip(),
                    age, sbp, dbp, bs, temp, hr,
                    risk_score,
                    category
                )
            else:
                patient_id = existing_patient_id.strip()

                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO patient_history
                    (username, created_at, patient_name, patient_id, age, systolicbp, diastolicbp, bs, bodytemp, heartrate, risk_score, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        st.session_state.username,
                        created_at,
                        patient_name,
                        patient_id,
                        age, sbp, dbp, bs, temp, hr,
                        risk_score,
                        category
                    ),
                )
                conn.commit()
                conn.close()

            st.metric("AI Risk Score", f"{risk_score:.2f}%")
            st.write(f"**Patient:** {patient_name} | **ID:** {patient_id}")
            st.warning(f"Recommendation: {recommendation}")
            show_confidence_gauge(confidence)
            st.success("Prediction saved to History")

            st.markdown("### Explainability (SHAP)")

            try:
                shap_values = explainer.shap_values(input_data)
                sv = get_shap_vector_for_class1(shap_values, n_features=len(COLUMNS))

                if len(sv) != len(COLUMNS):
                    st.warning("SHAP output shape mismatch; explainability hidden.")
                    return

                signed_series = pd.Series(sv, index=COLUMNS)
                top3 = signed_series.abs().sort_values(ascending=False).head(3).index.tolist()

                for feat in top3:
                    direction = "increases risk" if signed_series[feat] > 0 else "decreases risk"
                    st.write(f"- **{feat}**: {direction} (SHAP: {signed_series[feat]:.3f})")

                impact = signed_series.abs().sort_values(ascending=False)

                fig = plt.figure()
                plt.bar(impact.index, impact.values)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("Impact on Prediction")
                plt.title("Feature Importance (SHAP Impact)")
                plt.tight_layout()
                st.pyplot(fig)

                img_buf = BytesIO()
                fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
                chart_png_bytes = img_buf.getvalue()
                img_buf.close()
                plt.close(fig)

                inputs = {
                    "Patient Name": patient_name,
                    "Patient ID": patient_id,
                    "Age": age,
                    "SystolicBP": sbp,
                    "DiastolicBP": dbp,
                    "Blood Sugar (BS)": bs,
                    "Body Temperature": temp,
                    "Heart Rate": hr,
                }

                pdf_bytes = build_pdf_report(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    risk_score=risk_score,
                    category=category,
                    recommendation=recommendation,
                    confidence=confidence,
                    inputs=inputs,
                    top3_feats=top3,
                    signed_series=signed_series,
                    username_for_meta=st.session_state.username,
                    chart_png_bytes=chart_png_bytes,
                )

                safe_user = "".join([c for c in st.session_state.username if c.isalnum() or c in ("_", "-")])[:20]
                filename = f"maternal_risk_report_{safe_user}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )

            except Exception as e:
                st.warning(f"SHAP explainability failed: {e}")

def history_page():
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        set_page("login")
        st.rerun()

    st.markdown(
        f"""
        <div class="hero">
          <h2 style="margin:0">My Prediction History</h2>
          <div class="muted">User: <b>{st.session_state.username}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_user_history(st.session_state.username, limit=200)
    if df.empty:
        st.info("No predictions saved yet. Go to Predictor and run a prediction.")
        return

    st.dataframe(df, width="stretch", hide_index=True)
def run_patient_search():
    patient_id = st.session_state.get("patient_search_id", "").strip()

    if patient_id == "":
        st.session_state["patient_search_error"] = "Please enter a Patient ID"
        st.session_state["patient_search_result"] = None
        st.session_state["patient_trend_result"] = None
        return

    result = search_patient_by_id(patient_id)
    trend_df = load_patient_trend(patient_id)

    st.session_state["patient_search_error"] = None
    st.session_state["patient_search_result"] = result
    st.session_state["patient_trend_result"] = trend_df

def admin_page():
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        set_page("login")
        st.rerun()

    if not is_admin_user():
        st.error("Admin access only.")
        return

    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0">Admin Dashboard</h2>
          <div class="muted">View predictions from all users</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # init search state
    if "patient_search_result" not in st.session_state:
        st.session_state["patient_search_result"] = None
    if "patient_trend_result" not in st.session_state:
        st.session_state["patient_trend_result"] = None
    if "patient_search_error" not in st.session_state:
        st.session_state["patient_search_error"] = None
    if "patient_search_id" not in st.session_state:
        st.session_state["patient_search_id"] = ""

    st.markdown("### Patient Search")

    st.text_input(
        "Enter Patient ID",
        placeholder="Example: PAT000011",
        key="patient_search_id",
        on_change=run_patient_search,   # pressing Enter triggers search
    )

    st.button("Search Patient", on_click=run_patient_search)  # clicking button also triggers search

    # show error
    if st.session_state["patient_search_error"]:
        st.warning(st.session_state["patient_search_error"])

    # show results
    result = st.session_state["patient_search_result"]
    trend_df = st.session_state["patient_trend_result"]

    if result is not None:
        if result.empty:
            st.error("No patient found with this ID")
        else:
            st.success("Patient record found")
            st.dataframe(result, use_container_width=True, hide_index=True)

            if trend_df is not None and not trend_df.empty and len(trend_df) > 1:
                trend_df = trend_df.copy()
                trend_df["created_at"] = pd.to_datetime(trend_df["created_at"])

                st.markdown("### Risk Trend Chart")

                fig = px.line(
                    trend_df,
                    x="created_at",
                    y="risk_score",
                    markers=True,
                    title=f"Risk Trend for {st.session_state['patient_search_id'].strip()}",
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Risk Score (%)",
                )

                st.plotly_chart(fig, use_container_width=True)

            elif trend_df is not None and not trend_df.empty:
                st.info("Only one record found for this patient. More visits are needed to show a trend chart.")

    df = load_all_history(limit=1000)
    if df.empty:
        st.info("No predictions saved yet.")
        return

    st.markdown("### All Patient Records")
    users = ["All"] + sorted(df["username"].dropna().unique().tolist())
    pick = st.selectbox("Filter by user", users)

    if pick == "All":
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[df["username"] == pick], use_container_width=True, hide_index=True)

def dashboard_page():
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        set_page("login")
        st.rerun()

    if not is_admin_user():
        st.error("Access denied. Admin privileges required.")
        set_page("home")
        st.rerun()

    st.markdown(
        """
        <div class="hero">
          <h2>Maternal Risk Analytics Dashboard</h2>
          <div class="muted">Real-time statistics from patient predictions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_all_history(limit=1000)

    if df.empty:
        st.info("No prediction data available yet.")
        return

    total_patients = len(df)
    high_risk = len(df[df["category"] == "High Risk"])
    moderate_risk = len(df[df["category"] == "Moderate Risk"])
    low_risk = len(df[df["category"] == "Low Risk"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", total_patients)
    col2.metric("High Risk", high_risk)
    col3.metric("Moderate Risk", moderate_risk)
    col4.metric("Low Risk", low_risk)

    st.markdown("### Risk Distribution")
    risk_counts = df["category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]

    fig = px.pie(
        risk_counts,
        values="Count",
        names="Risk Category",
        title="Maternal Risk Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk Score Distribution")
    fig2 = px.histogram(
        df,
        x="risk_score",
        nbins=20,
        title="Risk Score Distribution",
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Router
# -----------------------------
page = get_page()

if page == "home":
    home_page()
elif page == "login":
    login_page()
elif page == "signup":
    signup_page()
elif page == "logout":
    logout_action()
elif page == "predictor":
    predictor_page()
elif page == "history":
    history_page()
elif page == "admin":
    admin_page()
elif page == "dashboard":
    dashboard_page()
else:
    set_page("home")
    st.rerun()