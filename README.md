# FlexFactor — AI-Based Constrained Payment Routing Optimization

This repository contains the final-project implementation and notebook for a dynamic payment-routing optimization system.

## Architecture

```text
Google Drive data
      │
      ▼
data loading / schema standardization
      │
      ▼
static + interaction + temporal-memory features
      │
      ▼
A1 / A2 / A3 probability models
      │
      ▼
candidate-route probabilities
      │
      ├──────────► offline LP oracle
      │
      ▼
online greedy / pressure policy
      │
      ▼
route assignment + capacity audit
```

The notebook is the **main project narrative**. The `src/` files contain the reusable implementation so the notebook stays readable.

## Repository structure

```text
FlexFactor_Final_Project_GitHub/
├── FlexFactor_Final_Project.ipynb
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── optimization.py
├── artifacts/
│   ├── reference_predictive_metrics.csv
│   ├── reference_policy_results.csv
│   └── README.md
├── report/
│   └── REPORT_STRUCTURE.md
└── presentation/
    └── PRESENTATION_STRUCTURE.md
```

## Google Drive setup

Create this folder in Google Drive:

```text
MyDrive/
└── FlexFactor_Final_Project/
    ├── data/
    │   └── transactions.parquet
    └── artifacts/
```

If the source file is CSV instead, that is fine; update `DATA_PATH` in the notebook/config.

Default Colab path:

```text
/content/drive/MyDrive/FlexFactor_Final_Project/data/transactions.parquet
```

## GitHub upload — first time

1. Unzip the project package.
2. Go to GitHub → **New repository**.
3. Suggested name: `flexfactor-routing-final-project`.
4. Prefer **Private** unless you explicitly want it public.
5. Create the repository without adding another README.
6. Click **Add file → Upload files**.
7. Drag the **contents** of this folder into GitHub.
8. Commit the upload.
9. Click **Code → HTTPS** and copy the repository URL.
10. Paste that URL into the first code cell of `FlexFactor_Final_Project.ipynb` as `REPO_URL`.

The raw dataset does not need to be uploaded to GitHub. Keep it in Drive.

## Run in Google Colab — recommended workflow

1. Go to `https://colab.research.google.com`.
2. Choose the **GitHub** tab.
3. Paste/open your repository and select `FlexFactor_Final_Project.ipynb`.
4. Edit the first cell:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/flexfactor-routing-final-project.git"
```

5. Run the bootstrap cell. It clones the full repo into the Colab runtime so `src/` imports work.
6. Run the install cell.
7. Run the Drive-mount cell and authorize Google Drive.
8. Confirm `DATA_PATH` points to the real file in Drive.
9. Set `REBUILD_FROM_RAW = True` when you want to execute the large-data sections.
10. Run the notebook top-to-bottom.

## Why both a notebook and `src/` files?

There is only **one implementation**:

- `src/` = reusable implementation details.
- notebook = calls those functions, explains why they exist, and shows results.

The notebook stays readable while the grader can inspect the actual algorithms in separate modules.

## Data schema

The code standardizes common raw names into:

```text
timestamp
success
attempt
transaction_id
route
amount
merchant_id
issuer_name
card_network
card_type
card_level
bank_category
mcc
processor
provider
sponsor_bank
response_code
response_description
```

Common FlexFactor-style aliases such as `Order_MerchantId`, `BinCheck_IssuerName`, and `PaymentProvider_Processor` are already included in `src/data.py`.

If any column is not recognized, add an override in the notebook:

```python
COLUMN_OVERRIDES = {
    "timestamp": "YOUR_RAW_TIMESTAMP_COLUMN",
    "success": "YOUR_RAW_SUCCESS_COLUMN",
}
```

## Two execution modes

### Fast / presentation mode

```python
REBUILD_FROM_RAW = False
```

Uses the saved small reference result tables to reproduce the headline outputs quickly.

### Full rebuild mode

```python
REBUILD_FROM_RAW = True
```

Loads the Drive dataset and runs the real data-processing sections.

Heavy temporal-feature construction is separately gated because it can take substantial time. After building it once, save the resulting Parquet file to Drive and load that checkpoint thereafter.

## Methodological note

The offline LP is a **full-hindsight model-based oracle**.

“100% LP opportunity” means 100% of the maximum incremental objective found by the LP under the frozen probability model and stated route-capacity constraints.

It does **not** mean 100% real-world improvement.

Alternative-route outcomes are counterfactual, so realized causal policy uplift requires prospective online / A-B validation.
