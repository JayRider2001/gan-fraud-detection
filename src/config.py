"""Paths and hyperparameters. Change values here, not in the training scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# ULB / Kaggle credit-card set, mirrored by TensorFlow for the Keras imbalanced-classification example.
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
RAW_CSV = DATA_DIR / "creditcard.csv"

SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15  # test is the remainder (0.15)

# --- WGAN-GP ---
NOISE_DIM = 32
G_HIDDEN = (128, 256)
C_HIDDEN = (256, 128)
BATCH_SIZE = 64
G_STEPS = 4000
N_CRITIC = 5
GP_LAMBDA = 10.0
GAN_LR = 1e-4
GAN_BETA_1 = 0.0
GAN_BETA_2 = 0.9

# How many extra fraud rows to print after G is trained.
# Same target count is used for SMOTE so the comparison is fair.
SYNTH_MULTIPLIER = 8

# --- XGBoost (identical for all three detectors) ---
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    n_jobs=-1,
    random_state=SEED,
    eval_metric="aucpr",
    tree_method="hist",
)

PROCESSED = ARTIFACTS / "processed.npz"
SCALER_PATH = ARTIFACTS / "scaler.joblib"
META_PATH = ARTIFACTS / "meta.json"
GENERATOR_PATH = ARTIFACTS / "generator.keras"
CRITIC_PATH = ARTIFACTS / "critic.keras"
GAN_HISTORY = ARTIFACTS / "gan_history.json"
SYNTH_PATH = ARTIFACTS / "synthetic_fraud.npy"
METRICS_PATH = ARTIFACTS / "metrics.json"
THRESHOLD_PATH = ARTIFACTS / "thresholds.json"
MODELS = {
    "weighted": ARTIFACTS / "xgb_weighted.json",
    "smote": ARTIFACTS / "xgb_smote.json",
    "gan": ARTIFACTS / "xgb_gan.json",
}
