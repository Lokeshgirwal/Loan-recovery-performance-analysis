import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_loan_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    print(f"Generating {n_samples} loan records...")

    # Basic distributions
    loan_types = ['Home Loan', 'Personal Loan', 'Auto Loan', 'Education Loan', 'Business Loan']
    
    # Specific attributes mapped to loan types for realism
    # (mean_amount, default_probability, base_interest)
    type_profiles = {
        'Home Loan': (3_500_000, 0.05, 8.5),
        'Auto Loan': (800_000, 0.08, 10.0),
        'Business Loan': (4_000_000, 0.18, 12.5),
        'Education Loan': (600_000, 0.15, 11.0),
        'Personal Loan': (400_000, 0.25, 15.0),
    }

    # Generate basic customer profiles
    customer_ages = np.random.randint(22, 65, size=n_samples)
    
    # Income correlated slightly with age (older -> higher income up to a point)
    base_incomes = np.random.lognormal(mean=13.5, sigma=0.6, size=n_samples) 
    incomes = np.clip(base_incomes, 2_000_00, 5_000_000).astype(int)

    # Issue dates spanning the last 12 months for trends
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    # Pre-allocate arrays
    v_loan_types = np.random.choice(loan_types, size=n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.15])
    v_loan_amounts = []
    v_interest_rates = []
    v_loan_status = []
    v_issue_dates = []

    for i in range(n_samples):
        l_type = v_loan_types[i]
        profile = type_profiles[l_type]
        
        # Loan Amount
        amt = np.random.normal(loc=profile[0], scale=profile[0]*0.2)
        amt = max(50_000, round(amt, -3)) # round to nearest 1k
        v_loan_amounts.append(int(amt))
        
        # Interest Rate (affected by income, higher income = slightly lower rate)
        income_factor = max(0, (2_000_000 - incomes[i]) / 1_000_000 * 0.5)
        rate = profile[2] + income_factor + np.random.uniform(-1, 1)
        v_interest_rates.append(round(rate, 2))
        
        # Default modeling
        # Higher probability if (Loan Amount / Income) is high (DTI proxy)
        dti_risk = min(1.0, (amt / max(incomes[i], 1)) * 0.1)
        # Age risk (younger implies slightly higher default)
        age_risk = 0.05 if customer_ages[i] < 30 else 0
        
        total_default_prob = profile[1] + dti_risk * 0.3 + age_risk
        total_default_prob = min(0.9, total_default_prob)
        
        is_default = np.random.rand() < total_default_prob
        v_loan_status.append('Charged Off' if is_default else 'Fully Paid')

        # Random Issue Date in the last year
        random_days = np.random.randint(0, 365)
        i_date = start_date + timedelta(days=random_days)
        v_issue_dates.append(i_date.strftime('%Y-%m-%d'))
        
    df = pd.DataFrame({
        'Loan_ID': [f'LN{str(i+1).zfill(5)}' for i in range(n_samples)],
        'Customer_Age': customer_ages,
        'Income': incomes,
        'Loan_Type': v_loan_types,
        'Loan_Amount': v_loan_amounts,
        'Interest_Rate': v_interest_rates,
        'Issue_Date': v_issue_dates,
        'Loan_Status': v_loan_status
    })

    # Step 3 requirement: Derived Status and Amounts
    df['Recovery_Status'] = df['Loan_Status'].map({
        'Fully Paid': 'Recovered',
        'Charged Off': 'Default'
    })
    
    # Calculate amounts
    def calc_recovery(row):
        amt = row['Loan_Amount']
        if row['Recovery_Status'] == 'Recovered':
            return amt * np.random.uniform(0.95, 1.0) # slightly under or full
        else:
            return amt * np.random.uniform(0.1, 0.4) # defaults recover 10-40%

    df['Recovery_Amount'] = df.apply(calc_recovery, axis=1).round(2)
    df['Outstanding_Amount'] = (df['Loan_Amount'] - df['Recovery_Amount']).clip(lower=0).round(2)
    df['Recovery_Rate'] = (df['Recovery_Amount'] / df['Loan_Amount'] * 100).round(2)

    # Define Risk Segment based directly on Income and Amount (as per user heatmap spec)
    # Simple explicit logic for resume readiness
    def calculate_risk(row):
        inc_segment = "Low" if row['Income'] < 600000 else ("Medium" if row['Income'] < 1500000 else "High")
        amt_segment = "High" if row['Loan_Amount'] > 2000000 else "Low"
        
        if inc_segment == "Low" and amt_segment == "High":
            return "High Risk"
        elif inc_segment == "High" and amt_segment == "Low":
            return "Low Risk"
        else:
            return "Medium Risk"

    df['Risk_Segment'] = df.apply(calculate_risk, axis=1)

    # Injecting "Messy Data" for Data Cleaning Step (Step 6)
    # 1. Add null values randomly to 'Customer_Age'
    null_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[null_indices, 'Customer_Age'] = np.nan
    
    # 2. Add some random duplicates
    duplicates = df.sample(int(n_samples * 0.02), random_state=1)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle slightly so duplicates aren't just at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Data generation complete!")
    print(f"Total Rows: {len(df)}")
    print(f"Default Rate: {(df['Recovery_Status'] == 'Default').mean() * 100:.1f}%")
    
    output_path = 'data/loan_recovery_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")

if __name__ == '__main__':
    generate_loan_data(10000) # Increased to 10k for better visualization density
