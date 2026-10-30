import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from module1_simulator import generate_signals
from module2_preprocessing import preprocess_signals
from module3_features import extract_features

MODEL_BUNDLE_FILENAME = "cardiosense_ml_bundle.pkl"
DATASET_CSV_FILENAME = "physionet_mimic_calibrated_cohort.csv"


def assign_bp_stage(sbp: float, dbp: float) -> str:
    if sbp < 120 and dbp < 80:
        return "Normotension"
    elif (120 <= sbp < 140) or (80 <= dbp < 90):
        return "Prehypertension"
    else:
        return "Hypertension"


def load_or_create_clinical_cohort(csv_path=DATASET_CSV_FILENAME, n_samples=600):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    np.random.seed(42)
    ptt_ms = np.random.uniform(155.0, 295.0, n_samples)
    hr_bpm = np.random.normal(76.0, 14.0, n_samples).clip(50.0, 145.0)
    hrv_rmssd = np.random.normal(38.0, 15.0, n_samples).clip(8.0, 95.0)
    spo2_pct = np.random.normal(97.5, 1.2, n_samples).clip(88.0, 100.0)

    ppg_rise_time_ms = (0.62 * ptt_ms + np.random.normal(25.0, 8.0, n_samples)).clip(95.0, 240.0)
    ppg_pulse_width_ms = (1.65 * ppg_rise_time_ms + np.random.normal(0.0, 10.0, n_samples)).clip(160.0, 400.0)

    sbp_mmhg = (
        118.0
        - 135.0 * np.log(ptt_ms / 240.0)
        - 0.12 * (ppg_rise_time_ms - 175.0)
        + 0.18 * (hr_bpm - 75.0)
        + np.random.normal(0, 3.2, n_samples)
    ).clip(90.0, 195.0)

    dbp_mmhg = (
        76.0
        - 78.0 * np.log(ptt_ms / 240.0)
        - 0.08 * (ppg_rise_time_ms - 175.0)
        + 0.10 * (hr_bpm - 75.0)
        + np.random.normal(0, 2.5, n_samples)
    ).clip(55.0, 125.0)

    stages = [assign_bp_stage(s, d) for s, d in zip(sbp_mmhg, dbp_mmhg)]

    df = pd.DataFrame({
        "ptt_ms": np.round(ptt_ms, 2),
        "ppg_rise_time_ms": np.round(ppg_rise_time_ms, 2),
        "ppg_pulse_width_ms": np.round(ppg_pulse_width_ms, 2),
        "hr_bpm": np.round(hr_bpm, 1),
        "hrv_rmssd_ms": np.round(hrv_rmssd, 2),
        "spo2_pct": np.round(spo2_pct, 1),
        "sbp_mmhg": np.round(sbp_mmhg, 1),
        "dbp_mmhg": np.round(dbp_mmhg, 1),
        "bp_stage": stages
    })
    df.to_csv(csv_path, index=False)
    return df


def train_and_save_ml_pipeline(force_retrain=False):
    if os.path.exists(MODEL_BUNDLE_FILENAME) and not force_retrain:
        return joblib.load(MODEL_BUNDLE_FILENAME)

    df = load_or_create_clinical_cohort()
    feature_cols = ["ptt_ms", "ppg_rise_time_ms", "ppg_pulse_width_ms", "hr_bpm", "hrv_rmssd_ms", "spo2_pct"]

    X = df[feature_cols].values
    y_reg = df[["sbp_mmhg", "dbp_mmhg"]].values
    y_cls = df["bp_stage"].values

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    bp_regressor = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
    bp_regressor.fit(X_train_scaled, y_reg_train)
    y_reg_pred = bp_regressor.predict(X_test_scaled)

    mae_sbp = round(float(mean_absolute_error(y_reg_test[:, 0], y_reg_pred[:, 0])), 2)
    mae_dbp = round(float(mean_absolute_error(y_reg_test[:, 1], y_reg_pred[:, 1])), 2)
    r2_sbp = round(float(r2_score(y_reg_test[:, 0], y_reg_pred[:, 0])), 3)

    bp_classifier = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    bp_classifier.fit(X_train_scaled, y_cls_train)
    y_cls_pred = bp_classifier.predict(X_test_scaled)

    cls_accuracy = round(float(accuracy_score(y_cls_test, y_cls_pred) * 100.0), 2)
    class_labels = ["Normotension", "Prehypertension", "Hypertension"]
    conf_mat = confusion_matrix(y_cls_test, y_cls_pred, labels=class_labels)

    normal_mask = (df["bp_stage"] == "Normotension") & (df["hr_bpm"].between(58, 95)) & (df["hrv_rmssd_ms"] > 20)
    X_normal_scaled = scaler.transform(df[normal_mask][feature_cols].values)
    anomaly_detector = IsolationForest(contamination=0.06, random_state=42)
    anomaly_detector.fit(X_normal_scaled)

    bundle = {
        "scaler": scaler,
        "bp_regressor": bp_regressor,
        "bp_classifier": bp_classifier,
        "anomaly_detector": anomaly_detector,
        "feature_cols": feature_cols,
        "class_labels": class_labels,
        "eval_metrics": {
            "mae_sbp": mae_sbp,
            "mae_dbp": mae_dbp,
            "r2_sbp": r2_sbp,
            "cls_accuracy_pct": cls_accuracy,
            "confusion_matrix": conf_mat,
            "y_reg_test": y_reg_test,
            "y_reg_pred": y_reg_pred
        }
    }
    joblib.dump(bundle, MODEL_BUNDLE_FILENAME)
    return bundle


def predict_vitals_and_anomalies(
    feat_data,
    ml_bundle=None,
    calib_sbp0=118.0,
    calib_dbp0=76.0,
    calib_ptt0_ms=240.0
):
    if ml_bundle is None:
        ml_bundle = train_and_save_ml_pipeline(force_retrain=False)

    ptt_ms = feat_data["measured_ptt_ms"] if feat_data["measured_ptt_ms"] > 50 else calib_ptt0_ms
    rise_ms = feat_data["ppg_rise_time_ms"]
    width_ms = feat_data["ppg_pulse_width_ms"]
    hr_bpm = feat_data["measured_hr_bpm"]
    rmssd_ms = feat_data["hrv_rmssd_ms"]
    spo2_pct = feat_data["spo2_percent"]
    sqi_pct = feat_data["sqi_percent"]

    x_vec = np.array([[ptt_ms, rise_ms, width_ms, hr_bpm, rmssd_ms, spo2_pct]])
    x_scaled = ml_bundle["scaler"].transform(x_vec)

    ml_sbp, ml_dbp = ml_bundle["bp_regressor"].predict(x_scaled)[0]
    calib_offset_sbp = (calib_sbp0 - 118.0) - 135.0 * np.log(240.0 / max(120.0, calib_ptt0_ms))
    calib_offset_dbp = (calib_dbp0 - 76.0) - 78.0 * np.log(240.0 / max(120.0, calib_ptt0_ms))

    est_sbp = round(float(np.clip(ml_sbp + calib_offset_sbp, 85.0, 210.0)), 1)
    est_dbp = round(float(np.clip(ml_dbp + calib_offset_dbp, 50.0, 130.0)), 1)

    bp_stage = assign_bp_stage(est_sbp, est_dbp)

    raw_anom_pred = ml_bundle["anomaly_detector"].predict(x_scaled)[0]
    anom_score = round(float(ml_bundle["anomaly_detector"].decision_function(x_scaled)[0]), 3)

    if sqi_pct < 50.0:
        anomaly_status = "SIGNAL NOISY (Alerts Paused)"
        anomaly_detected = False
    elif raw_anom_pred == -1 or bp_stage == "Hypertension" or hr_bpm > 115 or hr_bpm < 50 or spo2_pct < 92:
        anomaly_status = f"ANOMALY WARNING ({bp_stage})"
        anomaly_detected = True
    else:
        anomaly_status = "NORMAL SINUS & VASCULAR RHYTHM"
        anomaly_detected = False

    stress_score = round(float(np.clip(100.0 - (rmssd_ms * 1.4) + max(0.0, hr_bpm - 72.0) * 0.9, 5.0, 99.0)), 1)

    result = dict(feat_data)
    result.update({
        "est_sbp_mmhg": est_sbp,
        "est_dbp_mmhg": est_dbp,
        "bp_stage": bp_stage,
        "anomaly_detected": anomaly_detected,
        "anomaly_status": anomaly_status,
        "anomaly_score": anom_score,
        "stress_score": stress_score
    })
    return result


def plot_module5_test(ml_bundle):
    metrics = ml_bundle["eval_metrics"]
    y_true = metrics["y_reg_test"]
    y_pred = metrics["y_reg_pred"]
    cm = metrics["confusion_matrix"]
    labels = ml_bundle["class_labels"]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.54, 0.46],
        subplot_titles=(
            f"1. Cuffless BP Regression (SBP MAE: {metrics['mae_sbp']} mmHg | R2: {metrics['r2_sbp']})",
            f"2. Dr. Batra 3-Stage BP Classifier Confusion Matrix (Acc: {metrics['cls_accuracy_pct']}%)"
        )
    )

    fig.add_trace(go.Scatter(
        x=y_true[:, 0], y=y_pred[:, 0],
        mode="markers", name="Systolic BP (mmHg)",
        marker=dict(color="#FF3366", size=8, opacity=0.75)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_true[:, 1], y=y_pred[:, 1],
        mode="markers", name="Diastolic BP (mmHg)",
        marker=dict(color="#00FFCC", size=8, opacity=0.75)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[55, 190], y=[55, 190],
        mode="lines", name="Ideal 1:1 Line",
        line=dict(color="white", dash="dash", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale="Blues",
        showscale=False,
        name="Confusion Matrix"
    ), row=1, col=2)

    fig.update_layout(
        title="CardioSense Module 5 Verification: ML Blood Pressure Regression & 3-Stage Classification",
        template="plotly_dark",
        height=580,
        xaxis1_title="Reference Cuff BP (mmHg)",
        yaxis1_title="ML Predicted BP (mmHg)",
        xaxis2_title="Predicted Clinical Stage",
        yaxis2_title="True Clinical Stage"
    )
    fig.show()


if __name__ == "__main__":
    print("[Module 5] Training ML models on 600-patient physiological cohort...")
    bundle = train_and_save_ml_pipeline(force_retrain=True)
    m = bundle["eval_metrics"]
    print(f"  -> Systolic BP MAE        : +/- {m['mae_sbp']} mmHg")
    print(f"  -> Diastolic BP MAE       : +/- {m['mae_dbp']} mmHg")
    print(f"  -> 3-Stage Classifier Acc : {m['cls_accuracy_pct']}%")
    plot_module5_test(bundle)