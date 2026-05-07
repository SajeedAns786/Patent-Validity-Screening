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
st.set_page_config(page_title="Patent Viability Classifier", page_icon="⚖️", layout="centered")

st.title("⚖️ AI Patent Viability Classifier")
st.markdown("Enter a **USPTO Application Number** below. The system will query the real-time USPTO File Wrapper, analyze prosecution metrics, and predict the patent's structural enforceability.")

# User Input
app_number = st.text_input("Application Number (e.g., 14123456)", "")

if st.button("Analyze Patent"):
    if app_number.strip():
        with st.spinner('Querying USPTO Open Data Portal...'):
            title, features = fetch_patent_data(app_number.strip())
            
        if features:
            st.success(f"**Invention Title:** {title}")
            
            # Display Features Dashboard
            st.subheader("📊 Extracted Prosecution Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Claims", features[0])
            col2.metric("Citations", features[1])
            col3.metric("Office Actions", features[2], help="Real data extracted from File Wrapper history")
            col4.metric("Days", features[3])
            col5.metric("Breadth", features[4])
            
            # ML Prediction
            new_patent = np.array([features])
            new_scaled = scaler.transform(new_patent)
            pred_class = clf.predict(new_scaled)[0]
            pred_probs = clf.predict_proba(new_scaled)[0]
            
            st.divider()
            st.subheader("🧠 Machine Learning Classification")
            
            # Map classes to colors for styling
            color_map = {"Valid": "green", "Weak": "orange", "Disputed": "red", "Invalid": "darkred"}
            pred_color = color_map.get(labels[pred_class], "black")
            
            st.markdown(f"### Risk Category: <span style='color:{pred_color}'>{labels[pred_class]}</span>", unsafe_allow_html=True)
            
            st.write("**Probability Breakdown:**")
            for label, prob in zip(labels, pred_probs):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            
            st.caption("*Disclaimer: Only Office Action count is extracted live from the USPTO API. Other fields are simulated for this demonstration model.*")
    else:
        st.warning("Please enter a valid Application Number.")
