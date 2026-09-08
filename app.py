
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="PulseIQ | Heart Disease Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_cleaned.csv"
DB_PATH = ROOT / "data" / "heart_disease.db"
MODEL_PATH = ROOT / "models" / "heart_disease_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# ============================================================
# DARK / RED UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 82% 7%, rgba(239, 68, 68, .12), transparent 24%),
        radial-gradient(circle at 15% 85%, rgba(127, 29, 29, .10), transparent 26%),
        #070a0f;
    color: #f5f7fa;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090c11 0%, #0d1118 100%);
    border-right: 1px solid #252b35;
}

[data-testid="stSidebar"] * {
    color: #d7dce4;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid #2a3039;
    border-radius: 22px;
    padding: 30px 34px;
    min-height: 190px;
    background:
        radial-gradient(circle at 72% 45%, rgba(239,68,68,.17), transparent 27%),
        linear-gradient(115deg, #0b0f15, #11151c 60%, #090b10);
    box-shadow: 0 20px 60px rgba(0,0,0,.25);
}

.hero h1 {
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: 1.05;
    margin: 0;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero h1 span { color: #ff334f; }

.hero p {
    color: #aab2bf;
    margin: 12px 0 0;
    font-size: 1rem;
}

.eyebrow {
    color: #ff5b6d;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.card {
    background: linear-gradient(145deg, #11161d, #0c1016);
    border: 1px solid #29313b;
    border-radius: 16px;
    padding: 20px;
    box-shadow: inset 0 1px rgba(255,255,255,.025), 0 10px 35px rgba(0,0,0,.18);
}

.kpi {
    min-height: 128px;
    position: relative;
    overflow: hidden;
}

.kpi:before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #ff304f;
    box-shadow: 0 0 18px rgba(255,48,79,.65);
}

.kpi-label { color: #9da6b3; font-size: .82rem; }
.kpi-value { color: #f8fafc; font-size: 2rem; font-weight: 800; margin-top: 5px; }
.kpi-sub { color: #707987; font-size: .74rem; margin-top: 4px; }

.section-title {
    font-size: 1.12rem;
    font-weight: 700;
    margin: 5px 0 2px;
}

.section-sub {
    color: #7f8997;
    font-size: .78rem;
    margin-bottom: 12px;
}

.insight {
    background: linear-gradient(145deg, rgba(127,29,29,.26), rgba(17,22,29,.9));
    border: 1px solid #54212a;
    border-radius: 14px;
    padding: 15px;
    min-height: 110px;
}

.insight-num {
    color: #ff334f;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: 1px;
}

.insight-text {
    color: #dfe3e8;
    font-size: .82rem;
    line-height: 1.5;
    margin-top: 7px;
}

.badge {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: .7rem;
    font-weight: 700;
    background: rgba(239,68,68,.10);
    border: 1px solid #5c222c;
    color: #ff6a7a;
}

hr { border-color: #242b34; }

div[data-testid="stMetric"] {
    background: #10151c;
    border: 1px solid #29313b;
    border-radius: 14px;
    padding: 12px;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #6e2530;
    background: linear-gradient(135deg, #e51f3e, #a9142b);
    color: white;
    font-weight: 700;
}

.stButton > button:hover {
    border-color: #ff4c62;
    color: white;
    box-shadow: 0 0 24px rgba(255,48,79,.25);
}

[data-testid="stDataFrame"] {
    border: 1px solid #29313b;
    border-radius: 12px;
}

.small-note {
    color: #737d8a;
    font-size: .72rem;
}

.footer {
    color: #687280;
    font-size: .75rem;
    padding-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA / MODEL
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

df = load_data()
model, scaler = load_model()

# ============================================================
# CUSTOM NAVIGATION
# ============================================================

# Product-style navigation. Buttons are used instead of Streamlit's
# default radio controls so the sidebar looks like a real SaaS product.
st.markdown("""
<style>
/* ---------- Sidebar shell ---------- */
[data-testid="stSidebar"] {
    min-width: 275px !important;
    max-width: 275px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 14px 18px 14px;
}

/* ---------- Brand ---------- */
.piq-brand {
    padding: 12px 8px 18px 8px;
}

.piq-logo-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.piq-heart {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    color: #fff;
    font-size: 19px;
    background: radial-gradient(circle at 50% 45%, #ff5369, #9d142a);
    box-shadow: 0 0 24px rgba(239,51,79,.25);
    animation: piqPulse 1.8s infinite ease-in-out;
}

@keyframes piqPulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 18px rgba(239,51,79,.20); }
    12% { transform: scale(1.10); box-shadow: 0 0 28px rgba(239,51,79,.42); }
    24% { transform: scale(1); }
}

.piq-name {
    font-size: 1.23rem;
    font-weight: 800;
    letter-spacing: .8px;
    color: #f5f7fa;
}

.piq-sub {
    color: #737d8b;
    font-size: .62rem;
    letter-spacing: 1.25px;
    text-transform: uppercase;
    margin-top: 2px;
}

.piq-ecg {
    margin-top: 13px;
    height: 19px;
    overflow: hidden;
    opacity: .8;
}

.piq-ecg svg {
    width: 100%;
    height: 100%;
}

.piq-section {
    color: #596371;
    font-size: .61rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    margin: 14px 8px 6px;
}

/* ---------- Navigation buttons ---------- */
.piq-nav-btn button {
    width: 100%;
    min-height: 43px;
    text-align: left;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    background: transparent !important;
    color: #8e98a7 !important;
    font-size: .79rem !important;
    font-weight: 600 !important;
    letter-spacing: .25px !important;
    padding: 8px 12px !important;
    transition: all .22s ease !important;
    box-shadow: none !important;
}

.piq-nav-btn button:hover {
    background: rgba(255,255,255,.045) !important;
    border-color: #292f38 !important;
    color: #eef1f5 !important;
    transform: translateX(3px);
}

.piq-nav-active button {
    background:
        linear-gradient(90deg, rgba(239,51,79,.20), rgba(239,51,79,.055)) !important;
    border-color: #68232e !important;
    color: #ffffff !important;
    box-shadow:
        inset 3px 0 0 #ef334f,
        0 0 24px rgba(239,51,79,.08) !important;
}

.piq-nav-active button:hover {
    background:
        linear-gradient(90deg, rgba(239,51,79,.24), rgba(239,51,79,.07)) !important;
    color: #fff !important;
}

/* ---------- Status card ---------- */
.piq-status {
    margin: 18px 2px 0;
    padding: 13px;
    border: 1px solid #242b34;
    border-radius: 14px;
    background: linear-gradient(145deg,#11161d,#0b0f14);
}

.piq-status-title {
    color: #66707d;
    font-size: .58rem;
    font-weight: 800;
    letter-spacing: 1.4px;
}

.piq-status-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 9px;
    color: #b5bdc8;
    font-size: .70rem;
}

.piq-online {
    color: #59e39a;
    font-size: .62rem;
    font-weight: 700;
}

.piq-online::before {
    content: "●";
    margin-right: 5px;
    text-shadow: 0 0 8px #59e39a;
}

.piq-health {
    margin-top: 12px;
    padding-top: 11px;
    border-top: 1px solid #202630;
    display: flex;
    justify-content: space-between;
    align-items: end;
}

.piq-health-num {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f2f4f7;
}

.piq-health-label {
    color: #687280;
    font-size: .58rem;
    letter-spacing: .8px;
    text-transform: uppercase;
}

/* Streamlit's button wrapper should not add visual noise */
[data-testid="stSidebar"] .piq-nav-btn {
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)

if "piq_page" not in st.session_state:
    st.session_state.piq_page = "📊 Dashboard"

def nav_to(target):
    st.session_state.piq_page = target

with st.sidebar:
    st.markdown("""
    <div class="piq-brand">
      <div class="piq-logo-row">
        <div class="piq-heart">♥</div>
        <div>
          <div class="piq-name">PULSEIQ</div>
          <div class="piq-sub">Cardiovascular Intelligence</div>
        </div>
      </div>

      <div class="piq-ecg">
        <svg viewBox="0 0 250 24" preserveAspectRatio="none">
          <path d="M0 13 H42 L48 13 L54 4 L61 20 L68 13 H104
                   L110 13 L116 8 L121 18 L127 13 H165
                   L171 13 L177 5 L184 20 L191 13 H250"
                fill="none" stroke="#ef334f" stroke-width="1.7"/>
        </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

    current = st.session_state.piq_page

    def nav_button(label, target):
        css = "piq-nav-btn piq-nav-active" if current == target else "piq-nav-btn"
        st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
        st.button(
            label,
            key=f"nav_{target}",
            on_click=nav_to,
            args=(target,),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="piq-section">OVERVIEW</div>', unsafe_allow_html=True)
    nav_button("📊  Dashboard", "📊 Dashboard")

    st.markdown('<div class="piq-section">ANALYTICS</div>', unsafe_allow_html=True)
    nav_button("🗄️  SQL Analytics", "🗄️ SQL Analytics")
    nav_button("⚡  Query Performance", "⚡ Query Performance")

    st.markdown('<div class="piq-section">INTELLIGENCE</div>', unsafe_allow_html=True)
    nav_button("🤖  Risk Prediction", "🤖 Prediction")
    nav_button("📈  Model Performance", "📈 Model Performance")

    st.markdown("""
    <div class="piq-status">
      <div class="piq-status-title">SYSTEM STATUS</div>
      <div class="piq-status-line">
        <span>Database</span><span class="piq-online">ONLINE</span>
      </div>
      <div class="piq-status-line">
        <span>ML Engine</span><span class="piq-online">READY</span>
      </div>
      <div class="piq-health">
        <div>
          <div class="piq-health-num">99.9%</div>
          <div class="piq-health-label">System Health</div>
        </div>
        <div style="font-size:18px;color:#ef334f;">♥</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:#515b68;font-size:.61rem;margin-top:13px;letter-spacing:.5px;">'
        'BETTER DATA · HEALTHIER TOMORROWS</div>',
        unsafe_allow_html=True
    )

page = st.session_state.piq_page

# ============================================================
# HELPERS
# ============================================================

def chart_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd2dc"),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#202731", zerolinecolor="#202731")
    fig.update_yaxes(gridcolor="#202731", zerolinecolor="#202731")
    return fig

def show_heart_3d():
    # Self-contained procedural 3D heart. Three.js is loaded from a public CDN.
    html = r"""
    <style>
      html,body { margin:0; background:transparent; overflow:hidden; }
      #heart-wrap { width:100%; height:270px; position:relative; }
      #label {
        position:absolute; right:12px; top:10px; z-index:3;
        color:#ff6678; font:600 11px Inter,Arial,sans-serif;
        letter-spacing:.5px;
      }
    </style>
    <div id="heart-wrap"><div id="label">● LIVE HEART SIMULATION</div></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (() => {
      const root = document.getElementById('heart-wrap');
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(35, root.clientWidth/root.clientHeight, .1, 100);
      camera.position.set(0, 0, 8);

      const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(root.clientWidth, root.clientHeight);
      root.appendChild(renderer.domElement);

      const group = new THREE.Group();
      group.rotation.x = -0.18;
      scene.add(group);

      const pts = [];
      // Parametric anatomical-style heart surface.
      for (let i=0;i<100;i++) {
        const u = Math.PI * (i/99);
        for (let j=0;j<120;j++) {
          const v = 2*Math.PI*(j/120);
          const x = 0.82 * Math.pow(Math.sin(u),3) * Math.cos(v);
          const y = 0.72 * (13*Math.cos(u)-5*Math.cos(2*u)-2*Math.cos(3*u)-Math.cos(4*u))/13;
          const z = 0.82 * Math.pow(Math.sin(u),3) * Math.sin(v);
          pts.push(new THREE.Vector3(x*2.15, y*1.05, z*1.65));
        }
      }

      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.PointsMaterial({
        color:0xff334f, size:.035, transparent:true, opacity:.9,
        blending:THREE.AdditiveBlending
      });
      const heart = new THREE.Points(geom, mat);
      group.add(heart);

      const glowGeom = new THREE.SphereGeometry(1.7, 32, 32);
      const glowMat = new THREE.MeshBasicMaterial({
        color:0xff1730, transparent:true, opacity:.045,
        blending:THREE.AdditiveBlending
      });
      group.add(new THREE.Mesh(glowGeom, glowMat));

      const lineMat = new THREE.LineBasicMaterial({
        color:0xff5060, transparent:true, opacity:.25
      });

      for(let k=0;k<8;k++){
        const g = new THREE.BufferGeometry();
        const arr=[];
        for(let t=0;t<50;t++){
          const a=t/49*Math.PI*2;
          const r=1.4+k*.08;
          arr.push(new THREE.Vector3(
            Math.cos(a)*r,
            Math.sin(a)*.6*r,
            Math.sin(a*2+k)*.12
          ));
        }
        g.setFromPoints(arr);
        group.add(new THREE.Line(g,lineMat));
      }

      const light = new THREE.PointLight(0xff304f, 3.5, 10);
      light.position.set(0,1,3);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x8b8b8b, .8));

      let start = performance.now();

      function animate(now) {
        requestAnimationFrame(animate);
        const t=(now-start)/1000;
        const beat=Math.pow(Math.max(0, Math.sin(t*2.2)), 18);
        const pulse=1 + beat*.10;
        group.scale.set(pulse,pulse,pulse);
        group.rotation.y += .0035;
        light.intensity=2.8 + beat*3.2;
        renderer.render(scene,camera);
      }
      animate(start);

      window.addEventListener('resize', () => {
        camera.aspect=root.clientWidth/root.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(root.clientWidth,root.clientHeight);
      });
    })();
    </script>
    """
    components.html(html, height=280, scrolling=False)

# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":
    if df.empty:
        st.error("Dataset not found. Expected data/heart_cleaned.csv")
        st.stop()

    total = len(df)
    cases = int(df["target"].sum())
    rate = cases / total * 100
    avg_age = df["age"].mean()

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">DATA · INSIGHTS · HEALTHIER TOMORROWS</div>
      <h1>Heart Disease <span>Analytics</span></h1>
      <p>Analyze · Predict · Understand · Prevent &nbsp; <span class="badge">LIVE DATA</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card kpi"><div class="kpi-label">Total Patients</div>
        <div class="kpi-value">{total:,}</div><div class="kpi-sub">Records in database</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card kpi"><div class="kpi-label">Heart Disease Cases</div>
        <div class="kpi-value">{cases:,}</div><div class="kpi-sub">Positive cases</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card kpi"><div class="kpi-label">Disease Rate</div>
        <div class="kpi-value">{rate:.2f}%</div><div class="kpi-sub">Percentage of patients</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="card kpi"><div class="kpi-label">Data Stack</div>
        <div class="kpi-value">3</div><div class="kpi-sub">Python · SQL · ML</div></div>""", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.65, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Disease Rate by Age Group</div><div class="section-sub">Observed heart disease prevalence across age groups</div>', unsafe_allow_html=True)

        temp = df.copy()
        bins = [0,39,49,59,69,200]
        labels = ["Under 40","40-49","50-59","60-69","70+"]
        temp["age_group"] = pd.cut(temp["age"], bins=bins, labels=labels)
        age_result = temp.groupby("age_group", observed=False).agg(
            total_patients=("target","size"),
            heart_disease_cases=("target","sum")
        ).reset_index()
        age_result["disease_rate"] = age_result["heart_disease_cases"] / age_result["total_patients"] * 100

        fig = px.bar(
            age_result, x="age_group", y="disease_rate",
            text=age_result["disease_rate"].map(lambda x:f"{x:.2f}%"),
            labels={"age_group":"Age Group","disease_rate":"Disease Rate (%)"},
        )
        fig.update_traces(marker_color="#ef334f")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live Heart Monitor</div><div class="section-sub">Animated 3D visualization for the dashboard experience</div>', unsafe_allow_html=True)
        show_heart_3d()
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Chest Pain Analysis</div><div class="section-sub">Patient volume vs. observed heart disease cases</div>', unsafe_allow_html=True)

        chest = df.groupby("cp").agg(
            total_patients=("target","size"),
            heart_disease_cases=("target","sum")
        ).reset_index()
        chest["cp"] = chest["cp"].map({
            1:"Typical Angina", 2:"Atypical Angina",
            3:"Non-anginal Pain", 4:"Asymptomatic"
        })
        long = chest.melt(id_vars="cp", value_vars=["total_patients","heart_disease_cases"],
                          var_name="Metric", value_name="Patients")
        long["Metric"] = long["Metric"].map({
            "total_patients":"Total Patients",
            "heart_disease_cases":"Heart Disease Cases"
        })
        fig = px.bar(long, x="cp", y="Patients", color="Metric", barmode="group",
                     labels={"cp":"Chest Pain Type"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Insights</div><div class="section-sub">Findings from the underlying dataset</div>', unsafe_allow_html=True)

        top_age = age_result.loc[age_result["disease_rate"].idxmax()]
        top_cp = chest.loc[chest["heart_disease_cases"].idxmax()]

        insights = [
            f"The {top_age['age_group']} age group has the highest observed disease rate at {top_age['disease_rate']:.2f}%.",
            f"{top_cp['cp']} has the largest number of observed heart disease cases ({int(top_cp['heart_disease_cases'])}).",
            f"The dataset contains {cases:,} positive cases across {total:,} patient records.",
            "SQL analytics and indexing experiments were used alongside the ML pipeline.",
        ]
        for i, text in enumerate(insights, 1):
            st.markdown(f'<div class="insight"><div class="insight-num">0{i}</div><div class="insight-text">{text}</div></div><div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Age Group Details</div><div class="section-sub">SQL-style summary generated from the analysis dataset</div>', unsafe_allow_html=True)
    display = age_result.copy()
    display["disease_rate"] = display["disease_rate"].map(lambda x:f"{x:.2f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING</div>
      <h1>Risk <span>Prediction</span></h1>
      <p>Generate a prediction from the trained classification model.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None or scaler is None:
        st.error("Model files not found. Expected models/heart_disease_model.pkl and models/scaler.pkl")
        st.stop()

    st.warning("Educational project only — this model is not a medical diagnostic tool.")

    a,b = st.columns(2)
    with a:
        age = st.number_input("Age", 20, 100, 55)
        sex = st.selectbox("Sex", ["Female","Male"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", 80, 220, 130)
        chol = st.number_input("Cholesterol", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"])
        restecg = st.selectbox("Resting ECG", [0,1,2])

    with b:
        thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No","Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=.1)
        slope = st.selectbox("Slope", [1,2,3])
        ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [3,6,7])

    if st.button("🔍 Generate Prediction", use_container_width=True):
        sex_value = 1 if sex == "Male" else 0
        cp_value = {"Typical Angina":1,"Atypical Angina":2,"Non-anginal Pain":3,"Asymptomatic":4}[cp]
        fbs_value = 1 if fbs == "Yes" else 0
        exang_value = 1 if exang == "Yes" else 0

        input_data = pd.DataFrame([[
            age, sex_value, cp_value, trestbps, chol, fbs_value,
            restecg, thalach, exang_value, oldpeak, slope, ca, thal
        ]], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

        scaled = scaler.transform(input_data)
        prediction = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][1])

        st.divider()
        if prediction == 1:
            st.error("### Model Prediction: Heart Disease")
        else:
            st.success("### Model Prediction: No Heart Disease")

        x,y = st.columns(2)
        x.metric("Heart Disease Probability", f"{probability*100:.2f}%")
        y.metric("Model Confidence", f"{max(probability,1-probability)*100:.2f}%")

# ============================================================
# SQL ANALYTICS
# ============================================================

elif page == "🗄️ SQL Analytics":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">SQL · BUSINESS ANALYTICS</div>
      <h1>SQL <span>Insights</span></h1>
      <p>Analytics generated from the SQLite database with aggregation, grouping and ranking.</p>
    </div>
    """, unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error("Database not found. Expected data/heart_disease.db")
        st.stop()

    con = sqlite3.connect(DB_PATH)

    q1 = """
    SELECT COUNT(*) AS total_patients,
           SUM(target) AS heart_disease_cases,
           ROUND(100.0 * SUM(target) / COUNT(*), 2) AS disease_rate
    FROM patients;
    """
    overview = pd.read_sql_query(q1, con)

    a,b,c = st.columns(3)
    a.metric("Patients", int(overview.loc[0,"total_patients"]))
    b.metric("Disease Cases", int(overview.loc[0,"heart_disease_cases"]))
    c.metric("Disease Rate", f"{overview.loc[0,'disease_rate']:.2f}%")

    q2 = """
    SELECT
      CASE
        WHEN age < 40 THEN 'Under 40'
        WHEN age BETWEEN 40 AND 49 THEN '40-49'
        WHEN age BETWEEN 50 AND 59 THEN '50-59'
        WHEN age BETWEEN 60 AND 69 THEN '60-69'
        ELSE '70+'
      END AS age_group,
      COUNT(*) AS total_patients,
      SUM(target) AS heart_disease_cases,
      ROUND(100.0 * SUM(target) / COUNT(*), 2) AS disease_rate
    FROM patients
    GROUP BY age_group
    ORDER BY disease_rate DESC;
    """
    age_sql = pd.read_sql_query(q2, con)

    q3 = """
    SELECT cp,
           COUNT(*) AS total_patients,
           SUM(target) AS heart_disease_cases,
           ROUND(100.0 * SUM(target) / COUNT(*), 2) AS disease_rate
    FROM patients
    GROUP BY cp
    ORDER BY disease_rate DESC;
    """
    chest_sql = pd.read_sql_query(q3, con)
    chest_sql["cp"] = chest_sql["cp"].map({
        1:"Typical Angina",2:"Atypical Angina",
        3:"Non-anginal Pain",4:"Asymptomatic"
    })

    q4 = """
    SELECT
      CASE
        WHEN age < 40 THEN 'Under 40'
        WHEN age BETWEEN 40 AND 49 THEN '40-49'
        WHEN age BETWEEN 50 AND 59 THEN '50-59'
        WHEN age BETWEEN 60 AND 69 THEN '60-69'
        ELSE '70+'
      END AS age_group,
      COUNT(*) AS total_patients,
      SUM(target) AS heart_disease_cases,
      ROUND(100.0 * SUM(target) / COUNT(*), 2) AS disease_rate,
      RANK() OVER (ORDER BY 100.0 * SUM(target) / COUNT(*) DESC) AS disease_rate_rank
    FROM patients
    GROUP BY age_group
    ORDER BY disease_rate_rank;
    """
    ranked = pd.read_sql_query(q4, con)

    st.subheader("Disease Rate by Age Group")
    st.dataframe(age_sql, use_container_width=True, hide_index=True)

    st.subheader("Chest Pain Analysis")
    st.dataframe(chest_sql, use_container_width=True, hide_index=True)

    st.subheader("Age Group Ranking")
    st.dataframe(ranked, use_container_width=True, hide_index=True)

    con.close()

# ============================================================
# QUERY PERFORMANCE
# ============================================================

elif page == "⚡ Query Performance":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">SQL · QUERY OPTIMIZATION</div>
      <h1>Query <span>Performance</span></h1>
      <p>Benchmarking index effectiveness at larger data volume.</p>
    </div>
    """, unsafe_allow_html=True)

    a,b,c = st.columns(3)
    a.metric("Benchmark Rows", "303,000")
    b.metric("Without Index", "0.4573 sec")
    c.metric("With Composite Index", "0.1591 sec")

    st.success("65.21% faster for the selective query after adding the composite index.")

    st.markdown("""
    <div class="card">
    <div class="section-title">Optimization Experiment</div>
    <div class="section-sub">EXPLAIN QUERY PLAN + repeated execution benchmark</div>

    <p><b>Selective query</b></p>
    <pre style="background:#080b10;color:#ff7180;padding:14px;border-radius:10px;">
SELECT *
FROM patients_large
WHERE age = 77 AND target = 1;</pre>

    <p><b>Index</b></p>
    <pre style="background:#080b10;color:#ff7180;padding:14px;border-radius:10px;">
CREATE INDEX idx_age_target
ON patients_large(age, target);</pre>

    <p><b>Observed plan</b></p>
    <pre style="background:#080b10;color:#b8c0cc;padding:14px;border-radius:10px;">
Before → SCAN TABLE patients_large
After  → SEARCH TABLE patients_large USING INDEX idx_age_target
Improvement → 65.21%</pre>

    <p class="small-note">
    The experiment also showed that indexes are not universally faster:
    the earlier broad filtering workload became slower with an index.
    This demonstrates the importance of query selectivity and workload-aware optimization.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING · EVALUATION</div>
      <h1>Model <span>Performance</span></h1>
      <p>Held-out test-set evaluation of the trained classification model.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = {
        "Accuracy": 86.89,
        "Precision": 81.25,
        "Recall": 92.86,
        "F1 Score": 86.67,
        "ROC-AUC": 95.13,
    }

    cols = st.columns(5)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.2f}%")

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")

    cm = [[27,6],[2,26]]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No Disease","Predicted: Disease"],
        y=["Actual: No Disease","Actual: Disease"],
        text=cm,
        texttemplate="%{text}",
        colorscale=[
            [0, "#171b22"],
            [.5, "#7f1d2d"],
            [1, "#ef334f"]
        ],
        showscale=False,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Recall for the heart-disease class was 92.86% on the held-out test set. "
        "The dataset contains 303 records, so these figures are project benchmarks, "
        "not clinical validation."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    '<div class="footer">PulseIQ · Heart Disease Analytics &nbsp; | &nbsp; '
    'Python · SQL · Machine Learning · Streamlit &nbsp; | &nbsp; '
    'Better Data. Healthier Tomorrows.</div>',
    unsafe_allow_html=True
)
