import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "Quant_Intern", "Assignemnt")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

ASSIGNMENT_MCAP = os.path.join(DATA_DIR, "assignment_mcap.csv")
ASSIGNMENT_SECTOR = os.path.join(DATA_DIR, "assignment_sector_data.csv")
ASSIGNMENT_RULES = os.path.join(DATA_DIR, "assignment_investment_rules.csv")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
PREPROCESS_PATH = os.path.join(ARTIFACTS_DIR, "preprocess.joblib")
TARGET_COLUMN = "label"

# Choose a simple target: whether a stock appears in filtered portfolio next year
# If not available, we derive a proxy target via rank threshold
RANK_THRESHOLD = 500
