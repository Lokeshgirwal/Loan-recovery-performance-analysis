# Loan Recovery Performance Analysis 
## Presentation Outline (For PowerPoint / Slides)

---

### Slide 1: Title Slide
*   **Headline:** Loan Recovery Performance Analysis
*   **Sub-Headline:** Identifying Defaulters and Optimizing Banking Risk Portfolios
*   **Presenter:** [Your Name]
*   **Visual:** Bank/Data Analytics graphic.

---

### Slide 2: Project Objectives (Step 1)
*   **The Problem:** High loan default rates affect banking profitability and operational efficiency.
*   **Objective:** Analyze borrower demographics and loan profiles to predict default behaviors.
*   **Key Questions:** 
    *   Who defaults?
    *   What loans are most risky?

---

### Slide 3: Scope & Dataset (Steps 2-4)
*   **Dataset:** 10,000 synthetically generated loan records mimicking live production environments.
*   **Key Features:** `Customer_Age`, `Income`, `Loan_Amount`, `Loan_Type`, `Interest_Rate`.
*   **Environment:** Python (Pandas, Plotly, Streamlit, Scikit-Learn)

---

### Slide 4: Data Quality & Preprocessing (Steps 5-6)
*   **Challenges Identified:** Missing Age demographics & duplicated registry rows.
*   **Cleansing Action:**
    *   Imputed median values for age.
    *   Deduplicated entire frames ensuring pure analytical sets.
*   **Engineering:** Created logical `Recovery_Status` and Segmentations based on raw financial numbers.

---

### Slide 5: Exploratory Insights (Step 7)
*   **Bullet 1:** Secured Loans (Home/Auto) exhibit >90% recovery rates.
*   **Bullet 2:** Unsecured Loans (Personal) show double the default rates.
*   **Bullet 3:** High Income does not immune a borrower if Debt-to-Income ratios exceed critical thresholds.
*   *Provide a screenshot or link of one Bar Chart from Streamlit.*

---

### Slide 6: Predictive Modeling (Steps 8-10)
*   **Algorithm:** Logistic Regression (Classification).
*   **Features Used:** Demographics (Age, Income) + Loan Particulars (Amount, Type).
*   **Performance:** Model detects potential defaults efficiently; weighted classes balanced out raw recovery skew.

---

### Slide 7: The Interactive Dashboard (Step 9)
*   **Live Metrics:** Developed a Streamlit Dashboard capturing dynamic KPIs.
*   **Visual Highlights:** Real-time Risk segmentation donut charts and temporal recovery line graphs. 
*   *Insert Screenshot of the overview dashboard page.*

---

### Slide 8: Business Conclusions (Step 12)
*   **Action 1:** Revise interest rates for high-risk segments specifically in personal loans.
*   **Action 2:** Utilize Machine Learning checkpoints during standard loan underwriting.
---
