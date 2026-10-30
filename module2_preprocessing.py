"""
CardioSense — Module 2: Signal Denoising, Respiration Extraction & Spectrogram
File: module2_preprocessing.py
Purpose:
  1. Removes 50 Hz Indian AC power-line hum using an IIR Notch Filter.
  2. Cleans ECG (0.5-40 Hz) and PPG (0.5-8 Hz) using Zero-Phase Filtering (filtfilt)
     so peak timestamps suffer ZERO millisecond phase delay.
  3. Implements Dr. Pankaj Batra's multi-vital & time-frequency approach:
     extracts the Respiration Wave (0.15-0.45 Hz), calculates Breaths/Min,
     computes Signal Quality Index (SQI), and generates a 2D Spectrogram.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch, spectrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Module 1 so Module 2 can pull the noisy signals directly!
from module1_simulator import generate_signals


def apply_notch_filter(signal_data, fs, freq=50.0, quality_factor=30.0):
    """Deletes 50 Hz AC power-line interference with zero time lag."""
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal_data)


def apply_bandpass_filter(signal_data, fs, lowcut, highcut, order=3):
    """Keeps only frequencies between lowcut and highcut (Hz) with zero phase shift."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal_data)


def preprocess_signals(sim_data, enable_filters=True):
    """
    Inputs: Raw signal dictionary from Module 1
    Outputs: Dictionary updated with cleaned ECG/PPG, extracted Respiration wave,
             Signal Quality Index (SQI), and 2D Spectrogram arrays.
    """
    fs_ecg = sim_data["fs_ecg"]
    fs_ppg = sim_data["fs_ppg"]
    ecg_raw = sim_data["ecg_raw"]
    ppg_raw = sim_data["ppg_raw"]

    if enable_filters:
        # 1. Clean ECG: Delete 50 Hz hum first, then keep 0.5 to 40 Hz cardiac band
        ecg_notched = apply_notch_filter(ecg_raw, fs=fs_ecg, freq=50.0)
        ecg_filtered = apply_bandpass_filter(ecg_notched, fs=fs_ecg, lowcut=0.5, highcut=40.0, order=3)

        # 2. Clean PPG: Keep 0.5 to 8.0 Hz cardiac wave (removes slow breathing drift & high-freq fuzz)
        ppg_filtered = apply_bandpass_filter(ppg_raw, fs=fs_ppg, lowcut=0.5, highcut=8.0, order=3)
    else:
        ecg_filtered = ecg_raw.copy()
        ppg_filtered = ppg_raw.copy()

    # 3. Extract Respiration Wave from PPG (0.15 to 0.45 Hz = 9 to 27 Breaths/min)
    resp_wave = apply_bandpass_filter(ppg_raw, fs=fs_ppg, lowcut=0.15, highcut=0.45, order=2)

    # Calculate Breaths Per Minute using Fourier Power Spectral Density (Welch's method)
    freqs_resp, psd_resp = welch(resp_wave, fs=fs_ppg, nperseg=min(len(resp_wave), fs_ppg * 8))
    valid_resp_mask = (freqs_resp >= 0.15) & (freqs_resp <= 0.45)
    if np.any(valid_resp_mask):
        peak_resp_freq = freqs_resp[valid_resp_mask][np.argmax(psd_resp[valid_resp_mask])]
        estimated_resp_bpm = round(peak_resp_freq * 60.0, 1)
    else:
        estimated_resp_bpm = 0.0

    # 4. Calculate Signal Quality Index (SQI) from 0% to 100%
    # Checks how much clean heartbeat energy exists vs. high-frequency motion/optical noise
    freqs_ppg, psd_ppg = welch(ppg_raw, fs=fs_ppg, nperseg=min(len(ppg_raw), fs_ppg * 4))
    cardiac_band = (freqs_ppg >= 0.8) & (freqs_ppg <= 3.5)
    noise_band = (freqs_ppg > 8.0)
    cardiac_power = np.sum(psd_ppg[cardiac_band])
    high_freq_noise = np.sum(psd_ppg[noise_band]) + 1e-9
    sqi_percent = round(float(np.clip((cardiac_power / (cardiac_power + high_freq_noise * 5)) * 100, 0, 100)), 1)

    # 5. Compute Dr. Batra-Style 2D Time-Frequency Spectrogram of PPG
    spec_freqs, spec_times, spec_power = spectrogram(
        ppg_raw,
        fs=fs_ppg,
        nperseg=int(fs_ppg * 2.0),
        noverlap=int(fs_ppg * 1.75),
        nfft=512
    )
    freq_limit_mask = spec_freqs <= 4.0
    spec_freqs = spec_freqs[freq_limit_mask]
    spec_power_db = 10 * np.log10(spec_power[freq_limit_mask, :] + 1e-10)

    result = dict(sim_data)
    result.update({
        "ecg_filtered": ecg_filtered,
        "ppg_filtered": ppg_filtered,
        "resp_wave": resp_wave,
        "estimated_resp_bpm": estimated_resp_bpm,
        "sqi_percent": sqi_percent,
        "spec_freqs": spec_freqs,
        "spec_times": spec_times,
        "spec_power_db": spec_power_db
    })
    return result


def plot_module2_test(data):
    """Opens a 4-panel Plotly dashboard proving our filters and Spectrogram work."""
    fig = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
        subplot_titles=(
            "1. ECG Denoising: 50Hz Notch + Zero-Phase Filter (0.5-40Hz)",
            "2. PPG Denoising: Stripping Breathing Wobble & Fuzz (0.5-8Hz)",
            f"3. Extracted Respiration Wave | Measured: {data['estimated_resp_bpm']} BPM (True: {data['true_resp_bpm']} BPM)",
            f"4. Time-Frequency Spectrogram | SQI: {data['sqi_percent']}%"
        )
    )

    # Panel 1: ECG Raw vs Filtered
    fig.add_trace(go.Scatter(x=data["t_ecg"], y=data["ecg_raw"], name="Raw Noisy ECG",
                             line=dict(color="rgba(255,99,71,0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["t_ecg"], y=data["ecg_filtered"], name="Filtered Clean ECG",
                             line=dict(color="#00FFCC", width=2)), row=1, col=1)

    # Panel 2: PPG Raw vs Filtered
    fig.add_trace(go.Scatter(x=data["t_ppg"], y=data["ppg_raw"], name="Raw Wobbling PPG",
                             line=dict(color="rgba(255,165,0,0.45)", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=data["t_ppg"], y=data["ppg_filtered"], name="Filtered Cardiac PPG",
                             line=dict(color="#FF3366", width=2.2)), row=1, col=2)

    # Panel 3: Extracted Respiration Wave
    fig.add_trace(go.Scatter(x=data["t_ppg"], y=data["resp_wave"], name="Extracted Breathing Wave",
                             line=dict(color="#33CCFF", width=2.5)), row=2, col=1)

    # Panel 4: 2D Spectrogram Heatmap
    fig.add_trace(go.Heatmap(
        x=data["spec_times"],
        y=data["spec_freqs"],
        z=data["spec_power_db"],
        colorscale="Inferno",
        colorbar=dict(title="dB", len=0.45, y=0.2),
        name="Spectrogram"
    ), row=2, col=2)

    fig.update_layout(
        title="CardioSense Module 2 Verification: Zero-Phase Filtering & Multi-Vital Spectrogram",
        template="plotly_dark",
        height=760
    )
    fig.update_xaxes(range=[0, 4.0], title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(range=[0, 4.0], title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)

    fig.show()


if __name__ == "__main__":
    print("[Module 2] Pulling noisy signals from Module 1...")
    raw_data = generate_signals(duration_sec=12, heart_rate_bpm=75, ptt_ms=240, breathing_rate_bpm=18, noise_level=1.0)
    print("[Module 2] Running 50Hz Notch, Zero-Phase Bandpass, & Spectrogram extraction...")
    clean_data = preprocess_signals(raw_data, enable_filters=True)
    print(f"[Module 2] Success! SQI: {clean_data['sqi_percent']}% | Extracted Breathing Rate: {clean_data['estimated_resp_bpm']} BPM")
    plot_module2_test(clean_data)