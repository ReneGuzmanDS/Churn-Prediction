# Revenue & Churn Optimization Engine

A production-level machine learning project focused on **Predictive Modeling for Customer Retention and Revenue Optimization**.

## 🎯 Project Overview

This project leverages advanced machine learning (Random Forest) to predict customer churn and optimize revenue through data-driven pricing and retention strategies. The engine combines predictive analytics with economic principles of **price elasticity** and **return on investment (ROI)** to maximize customer lifetime value while minimizing churn risk.

### Business Value Proposition

**Core Problem**: Customer churn represents a significant revenue loss. Traditional reactive retention strategies are costly and inefficient.

**Solution**: This predictive engine enables proactive identification of at-risk customers and data-driven optimization of pricing structures (fee percentages) based on customer price elasticity, resulting in:
- **Reduced customer churn** through early intervention
- **Revenue optimization** via elasticity-based pricing
- **Improved ROI** on retention campaigns by targeting high-value at-risk customers

---

## 📊 Economics Framework: Elasticity & ROI Analysis

### Price Elasticity of Demand

**Elasticity Definition**: The sensitivity of customer demand (transaction frequency/volume) to changes in fee percentage.

```
Elasticity (E) = (Δ% Quantity) / (Δ% Price)
```

**Economic Implications**:
- **Elastic Customers (|E| > 1)**: Price-sensitive segment. Small fee increases lead to significant transaction volume reduction or churn.
- **Inelastic Customers (|E| < 1)**: Price-insensitive segment. Fee increases have minimal impact on transaction volume.
- **Optimal Pricing Strategy**: Set higher fees for inelastic customers, maintain competitive pricing for elastic customers.

### Revenue Impact Model

**Current Revenue**:
```
Monthly Revenue = Monthly Volume × Fee Percentage
Annual Revenue (AR) = Monthly Revenue × 12
```

**Churn Risk-Adjusted Revenue**:
```
Expected Annual Revenue = AR × (1 - Churn Probability)
Potential Revenue Loss = AR × Churn Probability
```

**ROI Calculation for Retention Campaign**:
```
Campaign Cost = Intervention Cost per Customer
Revenue Preserved = Potential Revenue Loss
ROI = (Revenue Preserved - Campaign Cost) / Campaign Cost × 100%
```

### Business Decision Framework

| Churn Probability | Annual Revenue | Campaign Cost | Revenue at Risk | ROI | Action |
|------------------|----------------|---------------|-----------------|-----|--------|
| < 30% (Low Risk) | $X | $Y | $Z | Low | Monitor only |
| 30-60% (Medium Risk) | $X | $Y | $Z | High | Targeted retention |
| > 60% (High Risk) | $X | $Y | $Z | Critical | Immediate intervention |

**Decision Rule**: Prioritize customers with high **Revenue at Risk** (High Annual Revenue × High Churn Probability) and positive ROI for retention campaigns.

---

## 🏗️ Project Structure

```
Churn Engine/
│
├── data/                      # Data storage
│   └── customer_data.csv      # Generated customer dataset
│
├── src/                       # Source code
│   ├── data_generator.py      # Script to generate realistic customer data
│   └── train.py               # Model training script with evaluation metrics
│
├── app/                       # Streamlit application
│   └── main.py                # Interactive What-If analysis dashboard
│
├── models/                    # Trained models
│   ├── churn_model.pkl        # Trained Random Forest model
│   └── churn_model_encoders.pkl  # Label encoders for categorical features
│
├── tests/                     # Unit tests (to be implemented)
│
└── README.md                  # This file
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate customer data**:
   ```bash
   python src/data_generator.py
   ```
   This creates a dataset of 5,000 customers with realistic features in `data/customer_data.csv`.

4. **Train the model**:
   ```bash
   python src/train.py
   ```
   This will:
   - Load the generated data
   - Perform train/test split (80/20)
   - Train a Random Forest classifier
   - Output accuracy, precision, recall, and F1-score
   - Save the model to `models/churn_model.pkl`

5. **Launch the Streamlit app**:
   ```bash
   streamlit run app/main.py
   ```
   The app will open in your browser with an interactive dashboard for What-If analysis.

---

## 📈 Features

### Data Generation (`src/data_generator.py`)

Generates realistic customer data with the following features:

- **avg_transaction_val**: Average transaction value in USD (correlated with churn risk)
- **fee_percentage**: Transaction fee as percentage (2-8%, affects churn)
- **tenure_months**: Customer tenure (new customers more likely to churn)
- **complaints_count**: Number of customer complaints (high complaints = high churn risk)
- **transaction_frequency**: Transactions per month (engagement metric)
- **customer_segment**: Tier classification (Standard, Silver, Gold, Premium)
- **payment_method**: Preferred payment method (categorical)
- **monthly_volume**: Calculated monthly transaction volume
- **support_tickets**: Technical support tickets count
- **loyalty_points**: Accrued loyalty points
- **churned**: Binary target variable (0 = Not Churned, 1 = Churned)

**Data Quality**: The generator creates realistic relationships between features (e.g., high fees increase churn probability, long tenure reduces churn risk).

### Model Training (`src/train.py`)

**Model**: Random Forest Classifier with optimized hyperparameters

**Evaluation Metrics**:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

**Output**: 
- Trained model saved as `.pkl` file
- Label encoders for categorical features
- Feature importance rankings
- Detailed classification report

### Streamlit Dashboard (`app/main.py`)

**Features**:
1. **What-If Analysis Sidebar**: Adjust customer characteristics in real-time
2. **Churn Probability Display**: Visual gauge showing churn risk level
3. **Top 5 Feature Importance**: Bar chart of features driving the prediction
4. **Business Insights**: AI-generated recommendations based on customer profile
5. **Economic Impact Metrics**: Monthly and annual revenue projections

**Use Cases**:
- **Pricing Optimization**: Test different fee percentages to find optimal pricing
- **Retention Prioritization**: Identify high-value customers at risk
- **Campaign ROI**: Estimate revenue impact of retention interventions

---

## 💡 Use Cases & Applications

### 1. Proactive Churn Prevention

**Scenario**: Identify customers with >50% churn probability and implement targeted retention campaigns.

**Economic Benefit**: Reduce customer acquisition costs by retaining existing customers (typically 5-25x cheaper than acquisition).

### 2. Dynamic Pricing Optimization

**Scenario**: Adjust fee percentages based on customer price elasticity.

**Example**:
- **Inelastic Customer** (Premium segment, low churn probability): Increase fee by 1% → Minimal churn risk, +20% revenue
- **Elastic Customer** (Standard segment, high churn probability): Reduce fee by 0.5% → Significant churn reduction, net revenue gain

**Economic Benefit**: Maximize revenue through personalized pricing strategies.

### 3. Customer Segmentation for Marketing

**Scenario**: Segment customers by churn risk and revenue potential.

**Segmentation Matrix**:
- **High Value, High Risk**: Premium retention campaigns
- **High Value, Low Risk**: Maintain current service level
- **Low Value, High Risk**: Standard retention efforts or allow churn
- **Low Value, Low Risk**: Cost-effective retention

**Economic Benefit**: Optimize marketing spend ROI by targeting the right segments.

### 4. Revenue Forecasting

**Scenario**: Forecast annual revenue by adjusting churn probabilities for different customer cohorts.

**Economic Benefit**: Improve financial planning and budget allocation for retention initiatives.

---

## 🔬 Model Performance

The Random Forest model is trained on 5,000 customers with the following typical performance metrics:

- **Accuracy**: ~85-90%
- **Precision**: ~80-85%
- **Recall**: ~75-80%
- **F1-Score**: ~77-82%

*Note: Actual performance may vary based on data distribution and model hyperparameters.*

### Feature Importance

The model typically ranks these as top predictive features:
1. **Complaints Count**: Strong indicator of dissatisfaction
2. **Fee Percentage**: Price sensitivity driver
3. **Tenure Months**: Loyalty indicator
4. **Transaction Frequency**: Engagement metric
5. **Average Transaction Value**: Customer value segment

---

## 🎓 Key Learnings & Economic Insights

### 1. Price Elasticity Varies by Segment

- **Premium customers** (high transaction value) are typically less price-sensitive → opportunity for fee optimization
- **Standard customers** (low transaction value) are highly price-sensitive → competitive pricing is critical

### 2. Churn Risk is Multi-Factorial

- **Not just about price**: Complaints, tenure, and engagement are equally important
- **Complaints are early warning signals**: Addressing complaints proactively can prevent churn

### 3. ROI Maximization Requires Prioritization

- Focus retention efforts on **high-value, high-risk** customers
- Low-value, high-risk customers may have negative ROI for premium retention campaigns

### 4. Customer Lifetime Value (CLV) Optimization

```
CLV = (Monthly Revenue × Gross Margin %) × (1 / Churn Rate) - Customer Acquisition Cost
```

By reducing churn probability, we directly increase CLV, creating long-term value for the business.

---

## 🔧 Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: Random Forest (scikit-learn)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit, Plotly
- **Model Persistence**: Pickle

---

## 📝 Future Enhancements

- [ ] A/B testing framework for pricing experiments
- [ ] Real-time model retraining pipeline
- [ ] Integration with CRM systems
- [ ] Customer lifetime value (CLV) prediction
- [ ] Elasticity estimation models
- [ ] Cost-benefit analysis for retention campaigns
- [ ] Automated alerting system for high-risk customers

---

## 📚 References

- **Price Elasticity**: [Economics of Customer Retention](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)
- **Churn Prediction**: Best practices in customer retention modeling
- **ROI Analysis**: Cost-benefit frameworks for retention initiatives

---

## 👤 Author

**Rene Guzman** - Revenue & Churn Optimization Engine

---


This project is a portfolio demonstration of predictive modeling and economic analysis capabilities.

---

