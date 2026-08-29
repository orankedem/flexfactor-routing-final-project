from pathlib import Path

# EDIT THESE TWO PATHS IN COLAB IF NEEDED.
DRIVE_PROJECT_ROOT = Path('/content/drive/MyDrive/FlexFactor_Final_Project')
DATA_PATH = DRIVE_PROJECT_ROOT / 'data' / 'transactions.parquet'
DRIVE_ARTIFACT_ROOT = DRIVE_PROJECT_ROOT / 'artifacts'

RANDOM_STATE = 42
CAPACITY_SHIFT = 0.30
PRIOR_STRENGTH = 50.0
HALF_LIVES_DAYS = [3, 14, 60, 180]

COLUMN_OVERRIDES = {}

HISTORY_GROUPS = {
    'global': [],
    'route': ['route'],
    'merchant': ['merchant_id'],
    'issuer': ['issuer_name'],
    'bankcat': ['bank_category'],
    'amount_exact': ['amount_exact'],
    'merchant_amount': ['merchant_id', 'amount_exact'],
    'merchant_route': ['merchant_id', 'route'],
    'issuer_route': ['issuer_name', 'route'],
    'bankcat_route': ['bank_category', 'route'],
    'cardtype_route': ['card_type', 'route'],
    'cardlevel_route': ['card_level', 'route'],
    'mcc_route': ['mcc', 'route'],
    'issuer_cardtype': ['issuer_name', 'card_type'],
    'issuer_processor': ['issuer_name', 'processor'],
    'issuer_sponsor': ['issuer_name', 'sponsor_bank'],
    'bankcat_processor': ['bank_category', 'processor'],
    'bankcat_sponsor': ['bank_category', 'sponsor_bank'],
    'merchant_processor': ['merchant_id', 'processor'],
    'merchant_sponsor': ['merchant_id', 'sponsor_bank'],
}

STATIC_CROSS_SPECS = {
    'bankcat_route': ['bank_category', 'route'],
    'cardnetwork_route': ['card_network', 'route'],
    'cardtype_route': ['card_type', 'route'],
    'cardlevel_route': ['card_level', 'route'],
    'mcc_route': ['mcc', 'route'],
}
