import streamlit as st
import numpy as np
import os
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load API key from .env file
load_dotenv()

# --- Model Setup ---
# Cache the model so it doesn't retrain on every button click
@st.cache_resource
def train_model():
    # Simulated training data
    # Features: [claims, citations, office_actions, days_to_grant, breadth_score]
    X = np.array([
        [15, 10, 1, 700,  8],   # Valid
        [12, 15, 2, 800,  7],   # Valid
        [18, 8,  1, 600,  9],   # Valid
        [5,  30, 5, 1100, 3],   # Weak
        [4,  25, 4, 950,  4],   # Weak
        [6,  35, 6, 1200, 2],   # Weak
        [8,  55, 7, 1400, 4],   # Disputed
        [7,  60, 8, 1500, 3],   # Disputed
        [9,  50, 7, 1300, 5],   # Disputed
        [3,  75, 9, 1800, 1],   # Invalid
        [2,  80, 10,1900, 2],   # Invalid
        [4,  70, 9, 1700, 1],   # Invalid
    ])
    y = np.array([0,0,0, 1,1,1, 2,2,2, 3,3,3])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=4, random_state=42, stratify=y)

    clf = LogisticRegression(solver='lbfgs', max_iter=500)
    clf.fit(X_train, y_train)
    
    return scaler, clf

scaler, clf = train_model()
labels = ['Valid', 'Weak', 'Disputed', 'Invalid']

# --- API Fetching Logic ---
def fetch_patent_data(app_number):
    url = f"https://api.uspto.gov/api/v1/patent/applications/{app_number}"
    api_key = os.getenv("USPTO_API_KEY")
    
    if not api_key:
        st.error("Error: USPTO_API_KEY not found in environment variables. Please check your .env file or host secrets.")
        return None, None
        
    try:
        headers = {'x-api-key': api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch data from USPTO API. Check if the Application Number is valid. Status code: {response.status_code}")
            return None, None
            
        data = response.json()
        bag = data.get('patentFileWrapperDataBag', [{}])[0]
        
        # Application Meta Data
        meta = bag.get('applicationMetaData', {})
        title = meta.get('inventionTitle', 'Unknown Title')
        
        # Calculate office actions
        events = bag.get('eventDataBag', [])
        office_actions = sum(1 for e in events if 'rejection' in e.get('eventDescriptionText', '').lower() or 'office action' in e.get('eventDescriptionText', '').lower())
        
        # Simulate structural metrics based on application number hash
        np.random.seed(sum(ord(c) for c in app_number))
        claims = np.random.randint(5, 40)
        citations = np.random.randint(5, 100)
        days_to_grant = np.random.randint(500, 2000)
        breadth_score = np.random.randint(1, 10)
        
        office_actions = max(1, office_actions) 
        
        return title, [claims, citations, office_actions, days_to_grant, breadth_score]
        
    except Exception as e:
        st.error(f"Error connecting to USPTO API: {e}")
        return None, None

# --- UI Setup ---
st.set_page_config(page_title="Patent Viability Classifier", page_icon="🏛️", layout="wide")

# Custom CSS for a US Corporate/Legal Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3, h4 { color: #0A2540; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E4E8;
        padding: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #005A9C;
    }
    div[data-testid="stMetric"] label { color: #5C6D7E; font-weight: 600; font-size: 0.9rem; }
    div[data-testid="stMetric"] div { color: #0A2540; }
    .stButton>button {
        background-color: #005A9C; color: white; border-radius: 3px; font-weight: bold;
        border: none; padding: 0.5rem 2rem; width: 100%; transition: 0.2s;
    }
    .stButton>button:hover { background-color: #004080; color: white; border: 1px solid #004080; }
    div[data-testid="stTextInput"] label { color: #0A2540; font-weight: bold; }
    .stAlert { border-left-color: #005A9C; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Seal_of_the_United_States_Patent_and_Trademark_Office.svg/200px-Seal_of_the_United_States_Patent_and_Trademark_Office.svg.png", width=120)
    st.markdown("### 🏛️ Legal IP Assessment Tool")
    st.markdown("This enterprise tool connects to the **USPTO Open Data Portal** to extract prosecution history and applies predictive modeling for Freedom-To-Operate (FTO) and litigation risk analysis.")
    st.divider()
    st.caption("Designed for Corporate IP Counsel & R&D Strategy Teams.")

# Main Body
st.title("Patent Enforceability & Risk Classifier")
st.markdown("Enter a **USPTO Application Number** to instantly retrieve the file wrapper and execute predictive structural analysis.")

# Layout: 2 Columns for input to keep it contained
col_input, col_empty = st.columns([1, 1])
with col_input:
    app_number = st.text_input("Application Number (e.g., 14123456)", "")
    analyze_btn = st.button("Execute Analysis")

if analyze_btn:
    if app_number.strip():
        with st.spinner('Establishing secure connection to USPTO Open Data Portal...'):
            title, features = fetch_patent_data(app_number.strip())
            
        if features:
            st.success(f"**Target Asset:** {title}")
            
            st.markdown("### Prosecution Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Claims", features[0])
            m2.metric("Citations", features[1])
            m3.metric("Office Actions", features[2], help="Sourced directly from USPTO Event History")
            m4.metric("Days to Grant", features[3])
            m5.metric("Breadth Score", f"{features[4]}/10")
            
            # ML Prediction
            new_patent = np.array([features])
            new_scaled = scaler.transform(new_patent)
            pred_class = clf.predict(new_scaled)[0]
            pred_probs = clf.predict_proba(new_scaled)[0]
            
            st.divider()
            st.markdown("### 🧠 Probabilistic Risk Classification")
            
            # Strict Corporate Color Mapping
            color_map = {"Valid": "#2E7D32", "Weak": "#F57C00", "Disputed": "#D32F2F", "Invalid": "#B71C1C"}
            pred_color = color_map.get(labels[pred_class], "#0A2540")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"<div style='padding: 20px; border: 2px solid {pred_color}; border-radius: 5px; text-align: center;'>"
                            f"<h4 style='color: #0A2540; margin:0;'>Risk Category</h4>"
                            f"<h2 style='color: {pred_color}; margin:0;'>{labels[pred_class].upper()}</h2>"
                            f"</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("**Confidence Distribution:**")
                for label, prob in zip(labels, pred_probs):
                    st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("*Disclaimer: Office Action metrics are sourced live from the USPTO API. Other fields are simulated for proof-of-concept modeling.*")
    else:
        st.warning("Please provide a valid Application Number.")
