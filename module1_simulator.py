"""
CardioSense — Module 1: Synthetic Physiological Signal Generator
File: module1_simulator.py
Purpose: Generates synchronized 500 Hz ECG and 250 Hz PPG signals with controllable
         Pulse Transit Time (PTT), Respiration Rate, and realistic sensor noise.
"""

import numpy as np
import neurokit2 as nk
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_signals(
    duration_sec=10,
    heart_rate_bpm=75,
    ptt_ms=240,
    breathing_rate_bpm=18,
    noise_level=1.0
):
    # 1. Define hardware sampling rates (Snapshots per second)
    fs_ecg = 500  # 500 samples/sec = 1 sample every 2 milliseconds
    fs_ppg = 250  # 250 samples/sec = 1 sample every 4 milliseconds

    t_ecg = np.arange(0, duration_sec, 1.0 / fs_ecg)
    t_ppg = np.arange(0, duration_sec, 1.0 / fs_ppg)

    # 2. Generate clean ECG and PPG on the same 500 Hz master clock
    ecg_clean = nk.ecg_simulate(
        duration=duration_sec,
        sampling_rate=fs_ecg,
        heart_rate=heart_rate_bpm,
        noise=0.0,
        random_state=42
    )

    ppg_clean_500 = nk.ppg_simulate(
        duration=duration_sec,
        sampling_rate=fs_ecg,
        heart_rate=heart_rate_bpm,
        ibi_randomness=0.02,
        frequency_modulation=0.0,
        random_state=42
    )

    # 3. Shift the PPG wave forward in time by exact PTT milliseconds
    shift_samples = int((ptt_ms / 1000.0) * fs_ecg)
    ppg_delayed_500 = np.roll(ppg_clean_500, shift_samples)

    # Downsample PPG from 500 Hz to 250 Hz (take every 2nd sample)
    ppg_clean = ppg_delayed_500[::2]

    # 4. Inject realistic 50 Hz AC wall hum + 0.3 Hz breathing wave + optical jitter
    np.random.seed(42)
    resp_freq_hz = breathing_rate_bpm / 60.0

    hum_50hz = (0.08 * noise_level) * np.sin(2 * np.pi * 50.0 * t_ecg)
    wander_ecg = (0.15 * noise_level) * np.sin(2 * np.pi * resp_freq_hz * t_ecg)
    fuzz_ecg = np.random.normal(0, 0.03 * noise_level, len(t_ecg))
    ecg_raw = ecg_clean + hum_50hz + wander_ecg + fuzz_ecg

    resp_ppg = (0.28 * max(0.3, noise_level)) * np.sin(2 * np.pi * resp_freq_hz * t_ppg)
    jitter_ppg = np.random.normal(0, 0.035 * noise_level, len(t_ppg))
    ppg_raw = ppg_clean + resp_ppg + jitter_ppg

    # Simulate Red & Infrared LED channels for SpO2 calculation later
    ppg_ir_raw = 50000 + (ppg_raw * 1500)
    ppg_red_raw = 45000 + (ppg_raw * 950)

    return {
        "t_ecg": t_ecg,
        "ecg_clean": ecg_clean,
        "ecg_raw": ecg_raw,
        "fs_ecg": fs_ecg,
        "t_ppg": t_ppg,
        "ppg_clean": ppg_clean,
        "ppg_raw": ppg_raw,
        "ppg_ir_raw": ppg_ir_raw,
        "ppg_red_raw": ppg_red_raw,
        "fs_ppg": fs_ppg,
        "true_ptt_ms": ptt_ms,
        "true_hr_bpm": heart_rate_bpm,
        "true_resp_bpm": breathing_rate_bpm,
        "noise_level": noise_level
    }


def plot_module1_test(data):
    """Opens an interactive Plotly chart in your browser to verify Module 1."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"1. ECG Sensor (500 Hz) | Input HR: {data['true_hr_bpm']} BPM | Noise Level: {data['noise_level']}",
            f"2. PPG Finger Sensor (250 Hz) | Shifted by True PTT: {data['true_ptt_ms']} ms | Breathing: {data['true_resp_bpm']} BPM"
        )
    )

    fig.add_trace(go.Scatter(x=data["t_ecg"], y=data["ecg_raw"], name="Raw Noisy ECG",
                             line=dict(color="rgba(255, 99, 71, 0.45)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["t_ecg"], y=data["ecg_clean"], name="Clean ECG Reference",
                             line=dict(color="#00FFCC", width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=data["t_ppg"], y=data["ppg_raw"], name="Raw PPG (With Breathing Wave)",
                             line=dict(color="rgba(255, 165, 0, 0.5)", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data["t_ppg"], y=data["ppg_clean"], name="Clean Delayed PPG",
                             line=dict(color="#FF3366", width=2.5)), row=2, col=1)

    fig.update_layout(
        title="CardioSense Module 1 Verification: Synthetic ECG + PPG Generator",
        template="plotly_dark",
        height=680,
        hovermode="x unified",
        xaxis2_title="Time (Seconds)",
        yaxis1_title="ECG (mV)",
        yaxis2_title="PPG (Normalized)"
    )
    fig.update_xaxes(range=[0, 4.0])
    fig.show()


if __name__ == "__main__":
    print("[Module 1] Generating 10 seconds of test signals...")
    test_data = generate_signals(
        duration_sec=10,
        heart_rate_bpm=75,
        ptt_ms=240,
        breathing_rate_bpm=18,
        noise_level=1.0
    )
    print(f"[Module 1] Success! Generated {len(test_data['ecg_raw'])} ECG samples and {len(test_data['ppg_raw'])} PPG samples.")
    plot_module1_test(test_data)