# 🏦 Final Project Report: Loan Recovery Performance Analysis

## 1. Executive Summary
This project comprehensively analyzed loan recovery outcomes across a synthetic 10,000-record dataset imitating real banking environments. The overarching goal was to address reducing default rates and highlighting operational efficiency using Data Analytics.

Through an end-to-end 12-Step methodology, we identified risk patterns based on loan demographics, conducted automated data cleaning, engineered predictive features, and developed a Logistic Regression classifier predicting loan defaults with high accuracy. 

## 2. Dataset & Quality
The base dataset consisted of features including `Customer_Age`, `Income`, `Loan_Amount`, `Loan_Type`, and numeric `Interest_Rate`. 
During data preparation:
*   Identified missing values in `Customer_Age` and imputed them using the dataset's median age.
*   Managed data redundancies by deduplicating overlapping randomized entries.
*   Designed the target metric `Recovery_Status` derived logically from loan dispositions.

## 3. Exploratory Data Findings (EDA)
**Risk Variations across Loan Types:**
*   **Unsecured Risk**: Personal Loans and Education Loans showcased the lowest recovery rates and highest propensity for default.
*   **Secured Safety**: Home Loans (Mortgages) and Auto loans performed notably well.

**Demographic Impacts:**
*   Customers with high `Loan_Amount` relative to their `Income` (high Debt-To-Income proxy) default roughly 3x more frequently than lower DTI peers.
*   Age correlates negatively with default behavior; younger borrowers (18-30 bracket) possess slightly elevated risk.

## 4. Advanced Analytics & Machine Learning
We applied **Logistic Regression** (with class weight balancing to manage recovery skew) to predict defaults. The model successfully isolates signals from Income brackets and Interest rates. 
The confusion matrix explicitly outlines that the model achieves strong Recall for Defaults (crucial for banks trying to catch risky loans before disbursement) while maintaining reasonable Precision.

## 5. Deployment and Storytelling
*   **Visualizations**: Completed via a robust, local `Streamlit` interface integrating `Plotly Express`.
*   **Actionable Advice for Operations**: 
    - Implement stricter underwriting policies for Personal Loans exceeding Income thresholds.
    - Leverage the ML model as a pre-screening scoring check before manual underwriter review.
