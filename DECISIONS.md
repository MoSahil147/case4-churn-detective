# Decisions Log — Case 4: Churn Detective

## AI Assistance Disclosure

I used Claude (Anthropic) throughout this project for structuring the notebook, writing the Streamlit dashboard, debugging the HuggingFace Spaces deployment and drafting this file. I can explain every line of code in my own words. The analytical decisions like which model to use, why the threshold matters and what the segments mean for the CMO are all my own. No internet templates or starter code were used.

---

## Assumptions I made

1. The dataset is treated as real telecom data even though it's synthetic because the CMO brief asks for actionable recommendations not a toy exercise.
2. `churned = 1` means the customer left within a 30-day window because the brief never defines the churn window and 30 days is the most common industry definition.
3. All customers are postpaid because the brief explicitly scopes to postpaid churn so I didn't model prepaid behaviour separately.
4. Retention offer costs ~$100 USD and saves ~$670 USD in LTV because the brief gave no business numbers. I grounded it in the data: $69.84 mean monthly charges x 24 months x 40% margin gets you to ~$670. The offer cost is roughly one month bill credit which is standard US telecom practice. The CMO's finance team should validate both numbers before acting on the revenue projections.
5. Customer behaviour next quarter follows similar patterns to historical data because without this assumption the model's predictions aren't actionable.
6. The primary success metric is churn rate reduction in the targeted cohort versus a holdout control and not offer redemption rate because a customer clicking a discount link is not success. A customer who stays is.

---

## Trade-offs

| Choice | Alternative | Why I picked this |
|---|---|---|
| XGBoost | Logistic Regression | XGBoost captures non-linear interactions (high charges + month-to-month + low tenure all compound together) and scored higher on ROC-AUC (0.7226 vs ~0.68). I kept Logistic Regression in the notebook as a visible baseline. |
| XGBoost | Random Forest | XGBoost trains faster and produces exact SHAP values via TreeExplainer which is critical for the per-customer explanations the CMO actually needs. |
| Label encoding | One-hot encoding | XGBoost handles ordinal integers natively. One-hot would add around 15 sparse columns with no performance benefit for tree models. |
| KMeans k=3 | k=2 or k=4 | The elbow method on inertia flattens after k=3. More importantly k=3 maps to three distinct root causes which are contract type, payment friction and service frustration and that gives three clearly different retention plays. |
| No SMOTE | SMOTE oversampling | At 36.2% churn the class imbalance is mild. SMOTE only makes sense when one class is under around 10-15% and adding it here would just introduce noise. |
| Cost-aware threshold tuning | Default 0.5 threshold | The costs are asymmetric ($100 offer vs $670 LTV saved) so the optimal decision boundary is lower than 0.5. Flagging more customers and accepting more false positives actually increases expected revenue saved. |
| HuggingFace Spaces | Streamlit Community Cloud | Both are free and HTTPS. HF Spaces supports Docker which gives full control over the runtime and lets me copy the `data/` and `outputs/` directories exactly where the app expects them. |

---

## What I de-scoped and why

- **Causal inference** because correlation is not causation and building a proper causal model like propensity score matching was out of scope for the time I had. I flagged this clearly in the notebook limitations section.
- **Real-time feature engineering** because total_charges is derived at prediction time as tenure x monthly_charges. A production system would pull live billing data from a warehouse but the brief asks for a prototype not a production pipeline.
- **Hyperparameter tuning** because XGBoost on sensible defaults already hits 0.7226 ROC-AUC. A grid search or Optuna run would likely push it to 0.74-0.75 but wasn't worth the time when segmentation and the dashboard were the higher-value deliverables.
- **Prepaid customer modelling** because the brief explicitly scopes to postpaid so I didn't build a separate model even though the dataset might mix both.

---

## What I'd do differently with another day

- Run a proper hyperparameter search on XGBoost and test a LightGBM baseline. I'd expect a 2-3 point AUC gain with not much extra effort.
- Build a proper A/B test design document because the 60-day measurement plan in the notebook is conceptual and a real rollout needs randomisation logic, sample size calculations and a guardrail metric to catch cannibalization.
- Sharpen the segmentation because KMeans on four features is just a starting point. With more time I'd try UMAP for dimensionality reduction before clustering and test whether 4-5 segments gives more actionable retention plays.
- Add a data drift monitor because the model degrades silently when churn drivers shift after something like a pricing change. A simple PSI check on monthly_charges and contract_type distributions would catch this before predictions go stale.
