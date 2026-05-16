# Case 4: Churn Detective

**Live demo:** https://huggingface.co/spaces/sahil147/churn-detective  
**Repo:** https://github.com/MoSahil147/case4-churn-detective  

## See it in action

**Demo video:** https://youtu.be/9B7frSfVOyE?si=047qDYOWkKD_jZSb

![App walkthrough](demo.gif)

## What this is

A mid-sized telecom is losing postpaid customers at 2.3% per month against a 1.5% industry benchmark and the CMO needs to know who is about to leave and why before signing off on a retention budget. This project builds an XGBoost churn predictor, explains every prediction with SHAP and groups at-risk customers into three actionable segments so the retention team knows exactly what offer to send to whom.

## How to run locally

1. `git clone https://github.com/sahil147/case4-churn-detective`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Open http://localhost:8501

## Stack

- **XGBoost** for churn prediction because it outperforms logistic regression on this dataset and produces exact SHAP values via TreeExplainer without any extra approximation
- **SHAP** to explain model predictions at the individual customer level so the CMO can see why a specific person is flagged and not just that they are flagged
- **KMeans (k=3)** to cluster predicted churners into segments because three clusters maps cleanly onto three distinct root causes and three different retention plays
- **Streamlit** for the dashboard because it lets you turn a notebook into an interactive app in a single Python file with no frontend code
- **HuggingFace Spaces** for deployment because it supports Docker out of the box and is free with no cold start issues that would matter for a demo

## What's NOT done

- Hyperparameter tuning on XGBoost. The model runs on sensible defaults and hits 0.7226 ROC-AUC which is solid for this dataset but a proper grid search would likely push it to 0.74 or 0.75.
- A/B test design document. The 60-day measurement plan in the notebook is conceptual. A real rollout would need sample size calculations and a randomisation plan before anyone runs the campaign.
- Causal inference. The model tells you who is likely to churn and SHAP tells you which features are driving it but high support calls is a symptom not a cause. Fixing the network is the intervention not flagging the customer.
- Prepaid customer modelling. The brief scopes to postpaid so this was never started.

## In production I would also add

- A data drift monitor that runs a PSI check on the key features every week so you know when the model is going stale before the predictions start degrading silently
- A proper experiment framework with a randomised holdout control so you can measure whether the retention campaign actually reduced churn rather than just counting voucher redemptions
- Live feature ingestion from a billing data warehouse so the predictor runs on fresh data rather than a static CSV snapshot
- Model retraining on a rolling 6-month window because customer behaviour shifts after pricing changes and the current model would not catch that
- Role-based access on the dashboard so a frontline retention agent sees the prediction and the recommended offer but not the raw SHAP values which are not meaningful to them
