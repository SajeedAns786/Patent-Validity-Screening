import numpy as np
import os
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

# Load API key from .env file
load_dotenv()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Class labels ────────────────────────────────────────────────
# 0 = Valid, 1 = Weak, 2 = Disputed, 3 = Invalid

# ── Simulated training data ──────────────────────────────────────
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

# ── Preprocess ───────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=4, random_state=42, stratify=y
)

# ── Train multi-class linear classifier ──────────────────────────
# multi_class='multinomial' uses softmax → probability per class
clf = LogisticRegression(
    solver='lbfgs',
    max_iter=500
)
clf.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print(classification_report(
    y_test, y_pred,
    target_names=['Valid', 'Weak', 'Disputed', 'Invalid']
))

# ── Predict a new patent ─────────────────────────────────────────
def fetch_patent_data(app_number):
    print(f"\nFetching File Wrapper data for Application {app_number} from USPTO API...")
    url = f"https://api.uspto.gov/api/v1/patent/applications/{app_number}"
    
    api_key = os.getenv("USPTO_API_KEY")
    if not api_key:
        print("Error: USPTO_API_KEY not found. Please ensure it is in the .env file.")
        return None
        
    try:
        headers = {'x-api-key': api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch data from USPTO API. Status code: {response.status_code}")
            return None
            
        data = response.json()
        
        # Parse data from the first patentFileWrapperDataBag item
        bag = data.get('patentFileWrapperDataBag', [{}])[0]
        
        # Extract Application Meta Data
        meta = bag.get('applicationMetaData', {})
        title = meta.get('inventionTitle', 'Unknown Title')
        print(f"Found Invention Title: {title}")
        
        # Calculate office actions by parsing the event history
        events = bag.get('eventDataBag', [])
        office_actions = 0
        for e in events:
            desc = e.get('eventDescriptionText', '').lower()
            if 'rejection' in desc or 'office action' in desc:
                office_actions += 1
                
        # The USPTO File Wrapper API provides deep prosecution history, 
        # but claims/citations require a separate Full-Text API query.
        # We supplement the actual Office Action count with simulated structural metrics:
        np.random.seed(sum(ord(c) for c in app_number))
        claims = np.random.randint(5, 40)
        citations = np.random.randint(5, 100)
        days_to_grant = np.random.randint(500, 2000)
        breadth_score = np.random.randint(1, 10)
        
        # Ensure at least 1 office action for baseline scaling if none found
        office_actions = max(1, office_actions) 
        
        return [claims, citations, office_actions, days_to_grant, breadth_score]
        
    except Exception as e:
        print(f"Error connecting to USPTO API: {e}")
        return None

print("\n" + "="*50)
app_number = input("Enter Application Number (e.g., 14123456): ").strip()

if app_number:
    features = fetch_patent_data(app_number)
    if features:
        print(f"\nExtracted Features:")
        print(f"  Claims         : {features[0]} (Simulated)")
        print(f"  Citations      : {features[1]} (Simulated)")
        print(f"  Office Actions : {features[2]} (Extracted from USPTO File Wrapper)")
        print(f"  Days to Grant  : {features[3]} (Simulated)")
        print(f"  Breadth Score  : {features[4]} (Simulated)")
        
        new_patent = np.array([features])
        new_scaled  = scaler.transform(new_patent)

        pred_class  = clf.predict(new_scaled)[0]
        pred_probs  = clf.predict_proba(new_scaled)[0]

        labels = ['Valid', 'Weak', 'Disputed', 'Invalid']
        print(f"\n--- Prediction Results ---")
        print(f"Predicted class : {labels[pred_class]}")
        print(f"Probabilities   :")
        for label, prob in zip(labels, pred_probs):
            print(f"  {label:10s} -> {prob*100:.1f}%")