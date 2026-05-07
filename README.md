# AI-Powered Patent Viability & Risk Classifier

🚀 **[Live Demo: Try the Patent Classifier on Hugging Face Spaces](https://huggingface.co/spaces/SajeedAns786/Patent-Classifier-v1)**

## Executive Summary
This project provides a machine learning prototype built with Python and `scikit-learn`, tailored specifically for Intellectual Property (IP) attorneys, in-house counsel, and corporate R&D strategy teams. 

Using a **Multinomial Logistic Regression** model, the system evaluates the structural and historical prosecution data of a target patent to predict its legal robustness and enforceability. This predictive modeling serves as a vital preliminary tool for Freedom-to-Operate (FTO) analysis, patent valuation, and litigation risk assessment.

## The Classification Matrix

The model evaluates prosecution metrics (e.g., prosecution length, claim count, citation density, office action frequency, and scope breadth) and classifies the target patent into one of four distinct risk categories:

### 1. `Valid` (High Enforceability)
* **Definition:** A robust patent that has survived standard prosecution with minimal resistance and exhibits a healthy balance of claim scope and prior art citations.
* **Strategic Use Case:** These patents form the backbone of a strong defensive portfolio or offensive litigation campaign. Companies should confidently invest in licensing these assets, asserting them against competitors, or using them as core collateral.

### 2. `Weak` (Narrow or Vulnerable)
* **Definition:** A patent that, while legally active, likely possesses overly narrow claims (due to heavy prosecution amendments) or lacks substantial breadth to block workarounds.
* **Strategic Use Case:** For FTO analysis, finding a competitor's patent in this category suggests a high probability of successfully designing around it. For internal portfolio management, these assets may be prime candidates for abandonment to reduce maintenance fees.

### 3. `Disputed` (High Litigation Risk)
* **Definition:** A patent characterized by anomalous prosecution history—such as an unusually high number of office actions, excessive citations, or extended delays to grant. These traits often indicate heavy prior art crowding or ambiguous claim language.
* **Strategic Use Case:** Signals a high likelihood of post-grant challenges (e.g., IPRs or PGRs). Attorneys should flag these patents for deep-dive invalidity searches before initiating an acquisition, a merger, or responding to a threat letter.

### 4. `Invalid` (Critical Vulnerability)
* **Definition:** A patent with statistical markers severely deviating from enforceable norms, indicating it is highly susceptible to invalidation under sections like 35 U.S.C. § 101, § 102, or § 103 if challenged in court or at the PTAB.
* **Strategic Use Case:** A green light for FTO. If a competitor attempts to assert a patent in this category, counsel can aggressively counter with an Inter Partes Review (IPR) or declaratory judgment, knowing the statistical probability of the patent surviving administrative review is exceptionally low.

## How It Works

1. **Model Training**: The core engine (`analyse.py`) trains a classification model using historical prosecution data.
2. **Interactive Assessment**: After establishing baseline parameters, the system enters an interactive assessment mode. 
   * The user inputs a target **Patent Number** (e.g., `US8335472B2`).
   * The system automatically queries Google Patents to fetch the asset's bibliographic data.
   * Prosecution features are extracted and fed into the trained linear classifier.
   * The model outputs a definitive risk classification along with a probabilistic breakdown across all four categories to aid in calculated risk-taking.

## Technical Setup & Execution

Technical teams or paralegals can run this prototype locally using the following steps:

**1. Install Prerequisites**
Ensure Python is installed, then install the required analytical libraries:
```bash
pip install scikit-learn numpy
```

**2. Execute the Classifier**
Navigate to the project directory and run the script:
```bash
cd "c:\Users\Sajeed Ansari\Desktop\AI\Linear-classifier"
python analyse.py
```

**3. Input Patent Data**
Follow the terminal prompts to input a target patent number for immediate probabilistic evaluation.

## Note on Data Architecture
*Disclaimer:* Due to Google Patents' client-side rendering and scraping restrictions, extracting precise real-time prosecution metrics (like office action counts) directly from HTML requires a dedicated commercial API (e.g., USPTO API, LexisNexis, or PatSnap). For the purposes of this standalone prototype, the dynamic extraction is deterministically simulated based on the patent number to accurately demonstrate the model's analytical capability without requiring commercial API keys.
