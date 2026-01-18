"""
Data Generator for Revenue & Churn Optimization Engine
Creates a realistic dataset of 5,000 customers for remittance/financial services industry
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=5000, random_seed=42):
    """
    Generate realistic customer data for churn prediction
    
    Args:
        n_customers: Number of customers to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with customer features and churn label
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Initialize lists for features
    customer_ids = []
    avg_transaction_values = []
    fee_percentages = []
    tenure_months = []
    complaints_count = []
    transaction_frequency = []
    customer_segments = []
    payment_methods = []
    monthly_volumes = []
    support_tickets = []
    loyalty_points = []
    churned = []
    
    # Generate data for each customer
    for i in range(n_customers):
        customer_id = f"CUST_{str(i+1).zfill(6)}"
        customer_ids.append(customer_id)
        
        # Average transaction value (in USD) - log-normal distribution
        # Remittance customers typically send $200-$2000 per transaction
        avg_transaction_val = np.random.lognormal(mean=5.5, sigma=0.8)
        avg_transaction_val = np.clip(avg_transaction_val, 50, 5000)
        avg_transaction_values.append(round(avg_transaction_val, 2))
        
        # Fee percentage - higher for lower value transactions (inverse relationship)
        # Typical range: 2% to 8%
        base_fee = np.random.uniform(2.0, 8.0)
        # Discount for high-value customers
        if avg_transaction_val > 1000:
            base_fee *= np.random.uniform(0.7, 0.9)
        fee_percentage = round(base_fee, 2)
        fee_percentages.append(fee_percentage)
        
        # Tenure in months - beta distribution (most customers are newer)
        tenure = int(np.random.beta(2, 5) * 60)  # 0-60 months
        if np.random.random() < 0.1:  # 10% are long-term customers
            tenure = np.random.randint(24, 60)
        tenure_months.append(tenure)
        
        # Transaction frequency per month
        freq = np.random.poisson(3)
        freq = max(1, min(freq, 20))  # Between 1 and 20 per month
        transaction_frequency.append(freq)
        
        # Monthly volume (avg_transaction_val * frequency)
        monthly_volume = avg_transaction_val * freq
        monthly_volumes.append(round(monthly_volume, 2))
        
        # Customer segment (based on monthly volume)
        if monthly_volume > 5000:
            segment = "Premium"
        elif monthly_volume > 2000:
            segment = "Gold"
        elif monthly_volume > 1000:
            segment = "Silver"
        else:
            segment = "Standard"
        customer_segments.append(segment)
        
        # Payment method
        payment_method = np.random.choice(
            ["Bank Transfer", "Credit Card", "Debit Card", "Digital Wallet", "Cash"],
            p=[0.35, 0.25, 0.20, 0.15, 0.05]
        )
        payment_methods.append(payment_method)
        
        # Complaints count - correlated with tenure and transaction issues
        # More complaints for longer-tenured customers who've had issues
        base_complaints = np.random.poisson(0.5)
        if tenure > 12:
            # Older customers more likely to have complaints
            base_complaints += np.random.poisson(1)
        complaints_count.append(int(base_complaints))
        
        # Support tickets (different from complaints - technical issues, etc.)
        support_tickets_count = np.random.poisson(0.3)
        if complaints_count[-1] > 2:
            support_tickets_count += np.random.poisson(2)
        support_tickets.append(int(support_tickets_count))
        
        # Loyalty points (accrued over tenure)
        points = int(tenure * 10 * np.random.uniform(0.8, 1.2))
        loyalty_points.append(points)
        
        # CHURN LABEL - complex logic based on multiple factors
        churn_probability = 0.0
        
        # High complaints increase churn probability
        churn_probability += complaints_count[-1] * 0.15
        
        # Very high fees increase churn (price sensitivity)
        if fee_percentage > 6.0:
            churn_probability += 0.20
        
        # Low transaction frequency (infrequent users more likely to churn)
        if transaction_frequency[-1] < 2:
            churn_probability += 0.15
        
        # Low tenure (new customers more likely to churn if dissatisfied)
        if tenure < 3:
            churn_probability += 0.25
        elif tenure > 24:
            # Long-term customers less likely to churn
            churn_probability -= 0.10
        
        # Low average transaction value (price-sensitive segment)
        if avg_transaction_val < 200:
            churn_probability += 0.10
        
        # High support tickets indicate issues
        churn_probability += support_tickets[-1] * 0.08
        
        # Add some randomness
        churn_probability += np.random.uniform(-0.1, 0.1)
        
        # Normalize and determine churn
        churn_probability = max(0.0, min(1.0, churn_probability))
        churned_label = 1 if np.random.random() < churn_probability else 0
        churned.append(churned_label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'avg_transaction_val': avg_transaction_values,
        'fee_percentage': fee_percentages,
        'tenure_months': tenure_months,
        'complaints_count': complaints_count,
        'transaction_frequency': transaction_frequency,
        'customer_segment': customer_segments,
        'payment_method': payment_methods,
        'monthly_volume': monthly_volumes,
        'support_tickets': support_tickets,
        'loyalty_points': loyalty_points,
        'churned': churned
    })
    
    return df

def save_data(df, filepath='../data/customer_data.csv'):
    """
    Save generated data to CSV file
    
    Args:
        df: DataFrame to save
        filepath: Path to save the CSV file
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")

if __name__ == "__main__":
    print("Generating customer dataset...")
    df = generate_customer_data(n_customers=5000, random_seed=42)
    
    print("\nDataset Statistics:")
    print(df.describe())
    print(f"\nChurn distribution:")
    print(df['churned'].value_counts())
    print(f"\nChurn rate: {df['churned'].mean():.2%}")
    
    # Save to data directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'customer_data.csv')
    save_data(df, filepath=data_path)
    
    print("\nData generation completed successfully!")
