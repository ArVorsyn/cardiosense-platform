"""
CardioSense — Module 6: Master Interactive Real-Time Plotly/Dash Control Center
File: module6_dashboard.py
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from module1_simulator import generate_signals
from module2_preprocessing import preprocess_signals
from module3_features import extract_features
from module4_storage_packets import simulate_binary_stream_and_save_hdf5
from module5_ml_models import train_and_save_ml_pipeline, predict_vitals_and_anomalies

print("[Module 6] Loading trained ML pipeline from Module 5...")
ML_BUNDLE = train_and_save_ml_pipeline(force_retrain=False)

app = dash.Dash(__name__, title="CardioSense Research Platform")

BG_DARK = "#0B0F19"
CARD_BG = "#111827"
BORDER_COL = "#1F2937"
TEXT_MAIN = "#F9FAFB"
TEXT_MUTED = "#9CA3AF"
ACCENT_CYAN = "#00FFCC"
ACCENT_PINK = "#FF3366"


def make_kpi_card(title, value_id, unit, subtitle_id, border_color=ACCENT_CYAN):
    return html.Div(
        style={
            "backgroundColor": CARD_BG,
            "border": f"1px solid {BORDER_COL}",
            "borderLeft": f"4px solid {border_color}",
            "borderRadius": "8px",
            "padding": "10px 14px",
            "flex": "1",
            "minWidth": "150px",
        },
        children=[
            html.Div(title, style={"color": TEXT_MUTED, "fontSize": "11px", "fontWeight": "600", "textTransform": "uppercase"}),
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": "6px", "marginTop": "4px"},
                children=[
                    html.Span("--", id=value_id, style={"color": TEXT_MAIN, "fontSize": "22px", "fontWeight": "700"}),
                    html.Span(unit, style={"color": TEXT_MUTED, "fontSize": "12px"}),
                ],
            ),
            html.Div("--", id=subtitle_id, style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "3px"}),
        ],
    )


app.layout = html.Div(
    style={"backgroundColor": BG_DARK, "color": TEXT_MAIN, "minHeight": "100vh", "fontFamily": "Segoe UI, sans-serif", "padding": "14px"},
    children=[
        # TOP HEADER BAR
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "backgroundColor": CARD_BG,
                "padding": "12px 18px",
                "borderRadius": "8px",
                "border": f"1px solid {BORDER_COL}",
                "marginBottom": "14px",
            },
            children=[
                html.Div([
                    html.H2("CardioSense: Multi-Sensor ECG + PPG Cuffless BP & Anomaly Platform",
                            style={"margin": "0", "fontSize": "20px", "color": ACCENT_CYAN}),
                    html.Div("5-Layer Pipeline: Sensing (500/250Hz) -> Zero-Phase Filtering & Spectrogram -> PTT Stopwatch -> Binary/HDF5 -> ML Inference",
                             style={"color": TEXT_MUTED, "fontSize": "12px", "marginTop": "2px"}),
                ]),
                html.Div(id="top-alert-banner", style={
                    "padding": "8px 16px",
                    "borderRadius": "6px",
                    "fontWeight": "700",
                    "fontSize": "13px",
                    "backgroundColor": "#065F46",
                    "color": "#A7F3D0",
                }),
            ],
        ),

        # MAIN BODY: LEFT SIDEBAR CONTROLS + RIGHT MONITOR AREA
        html.Div(
            style={"display": "flex", "gap": "14px", "alignItems": "flex-start"},
            children=[
                # LEFT SIDEBAR
                html.Div(
                    style={
                        "width": "310px",
                        "backgroundColor": CARD_BG,
                        "padding": "14px",
                        "borderRadius": "8px",
                        "border": f"1px solid {BORDER_COL}",
                        "flexShrink": "0",
                    },
                    children=[
                        html.H4("1. Quick Clinical Presets", style={"margin": "0 0 8px 0", "fontSize": "14px", "color": ACCENT_CYAN}),
                        dcc.Dropdown(
                            id="preset-dropdown",
                            options=[
                                {"label": "Normal Resting Subject (PTT 240ms)", "value": "normal"},
                                {"label": "Prehypertension / Exam Stress (PTT 200ms)", "value": "prehyp"},
                                {"label": "Hypertension / Stiff Arteries (PTT 165ms)", "value": "hyp"},
                                {"label": "Motion Artifact / Noisy Finger (SQI Gate)", "value": "noisy"},
                            ],
                            value="normal",
                            clearable=False,
                            style={"color": "#000000", "fontSize": "13px", "marginBottom": "14px"},
                        ),

                        html.Hr(style={"borderColor": BORDER_COL}),
                        html.H4("2. Physiology Inputs (Module 1)", style={"margin": "10px 0 8px 0", "fontSize": "14px", "color": ACCENT_CYAN}),

                        html.Label("True Arterial PTT Delay (ms):", style={"fontSize": "12px", "color": TEXT_MUTED}),
                        dcc.Slider(id="slider-ptt", min=155, max=290, step=5, value=240,
                                   marks={160: "160", 220: "220", 280: "280"},
                                   tooltip={"placement": "bottom", "always_visible": True}),

                        html.Label("Heart Rate (BPM):", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "10px", "display": "block"}),
                        dcc.Slider(id="slider-hr", min=52, max=140, step=2, value=75,
                                   marks={55: "55", 75: "75", 100: "100", 135: "135"},
                                   tooltip={"placement": "bottom", "always_visible": True}),

                        html.Label("Respiration Rate (Breaths/min):", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "10px", "display": "block"}),
                        dcc.Slider(id="slider-resp", min=10, max=26, step=1, value=18,
                                   marks={10: "10", 18: "18", 26: "26"},
                                   tooltip={"placement": "bottom", "always_visible": True}),

                        html.Hr(style={"borderColor": BORDER_COL, "marginTop": "14px"}),
                        html.H4("3. Noise & Filter Controls (Module 2)", style={"margin": "10px 0 8px 0", "fontSize": "14px", "color": ACCENT_CYAN}),

                        html.Label("50Hz Hum & Motion Noise Level:", style={"fontSize": "12px", "color": TEXT_MUTED}),
                        dcc.Slider(id="slider-noise", min=0.0, max=3.0, step=0.2, value=1.0,
                                   marks={0: "Clean", 1: "Normal", 2.5: "Heavy"},
                                   tooltip={"placement": "bottom", "always_visible": True}),

                        dcc.Checklist(
                            id="toggle-filters",
                            options=[{"label": " Enable 50Hz Notch + Zero-Phase Filters", "value": "ON"}],
                            value=["ON"],
                            style={"fontSize": "12px", "marginTop": "10px", "color": TEXT_MAIN},
                        ),

                        html.Hr(style={"borderColor": BORDER_COL, "marginTop": "14px"}),
                        html.H4("4. Cuff Calibration Anchor (Module 5)", style={"margin": "10px 0 8px 0", "fontSize": "14px", "color": ACCENT_CYAN}),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                            children=[
                                html.Div([
                                    html.Label("Ref SBP", style={"fontSize": "11px", "color": TEXT_MUTED}),
                                    dcc.Input(id="input-sbp0", type="number", value=118, style={"width": "100%", "padding": "4px", "backgroundColor": "#1F2937", "color": "white", "border": "1px solid #374151"}),
                                ], style={"flex": "1"}),
                                html.Div([
                                    html.Label("Ref DBP", style={"fontSize": "11px", "color": TEXT_MUTED}),
                                    dcc.Input(id="input-dbp0", type="number", value=76, style={"width": "100%", "padding": "4px", "backgroundColor": "#1F2937", "color": "white", "border": "1px solid #374151"}),
                                ], style={"flex": "1"}),
                                html.Div([
                                    html.Label("Ref PTT", style={"fontSize": "11px", "color": TEXT_MUTED}),
                                    dcc.Input(id="input-ptt0", type="number", value=240, style={"width": "100%", "padding": "4px", "backgroundColor": "#1F2937", "color": "white", "border": "1px solid #374151"}),
                                ], style={"flex": "1"}),
                            ],
                        ),

                        html.Button(
                            "Log Snapshot to HDF5 (.h5)",
                            id="btn-save-hdf5",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginTop": "6px",
                                "backgroundColor": "#2563EB",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "6px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="hdf5-status-text", style={"fontSize": "11px", "color": "#34D399", "marginTop": "6px"}),
                    ],
                ),

                # RIGHT MAIN AREA
                html.Div(
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                            children=[
                                make_kpi_card("Est. Blood Pressure", "kpi-bp", "mmHg", "kpi-bp-stage", border_color=ACCENT_PINK),
                                make_kpi_card("Pulse Transit Time", "kpi-ptt", "ms", "kpi-rise-time", border_color="#A855F7"),
                                make_kpi_card("Heart Rate (ECG)", "kpi-hr", "BPM", "kpi-hrv", border_color=ACCENT_CYAN),
                                make_kpi_card("Blood Oxygen (SpO2)", "kpi-spo2", "%", "kpi-sqi", border_color="#3B82F6"),
                                make_kpi_card("Respiration Rate", "kpi-resp", "Br/min", "kpi-stress", border_color="#10B981"),
                                make_kpi_card("Binary Telemetry", "kpi-pkt", "KB/s", "kpi-crc", border_color="#F59E0B"),
                            ],
                        ),
                        html.Div(
                            style={"backgroundColor": CARD_BG, "borderRadius": "8px", "border": f"1px solid {BORDER_COL}", "padding": "6px"},
                            children=[
                                dcc.Graph(id="master-plotly-graph", style={"height": "690px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("slider-ptt", "value"),
    Output("slider-hr", "value"),
    Output("slider-resp", "value"),
    Output("slider-noise", "value"),
    Input("preset-dropdown", "value"),
)
def apply_preset(preset_name):
    if preset_name == "normal":
        return 240, 75, 18, 0.8
    elif preset_name == "prehyp":
        return 200, 92, 21, 1.0
    elif preset_name == "hyp":
        return 165, 108, 23, 1.0
    elif preset_name == "noisy":
        return 235, 80, 18, 2.8
    return 240, 75, 18, 1.0


@app.callback(
    Output("kpi-bp", "children"),
    Output("kpi-bp-stage", "children"),
    Output("kpi-ptt", "children"),
    Output("kpi-rise-time", "children"),
    Output("kpi-hr", "children"),
    Output("kpi-hrv", "children"),
    Output("kpi-spo2", "children"),
    Output("kpi-sqi", "children"),
    Output("kpi-resp", "children"),
    Output("kpi-stress", "children"),
    Output("kpi-pkt", "children"),
    Output("kpi-crc", "children"),
    Output("top-alert-banner", "children"),
    Output("top-alert-banner", "style"),
    Output("hdf5-status-text", "children"),
    Output("master-plotly-graph", "figure"),
    Input("slider-ptt", "value"),
    Input("slider-hr", "value"),
    Input("slider-resp", "value"),
    Input("slider-noise", "value"),
    Input("toggle-filters", "value"),
    Input("input-sbp0", "value"),
    Input("input-dbp0", "value"),
    Input("input-ptt0", "value"),
    Input("btn-save-hdf5", "n_clicks"),
)
def update_entire_dashboard(ptt_ms, hr_bpm, resp_bpm, noise_lvl, filter_chk, sbp0, dbp0, ptt0, n_clicks):
    enable_filters = "ON" in (filter_chk or [])
    sbp0 = float(sbp0 or 118.0)
    dbp0 = float(dbp0 or 76.0)
    ptt0 = float(ptt0 or 240.0)

    sim = generate_signals(
        duration_sec=10,
        heart_rate_bpm=hr_bpm,
        ptt_ms=ptt_ms,
        breathing_rate_bpm=resp_bpm,
        noise_level=noise_lvl
    )
    clean = preprocess_signals(sim, enable_filters=enable_filters)
    feat = extract_features(clean)
    pkt_stats = simulate_binary_stream_and_save_hdf5(feat, hdf5_filename="cardiosense_session.h5", inject_corrupted_packets=2)
    pred = predict_vitals_and_anomalies(
        feat,
        ml_bundle=ML_BUNDLE,
        calib_sbp0=sbp0,
        calib_dbp0=dbp0,
        calib_ptt0_ms=ptt0
    )

    bp_str = f"{pred['est_sbp_mmhg']:.0f}/{pred['est_dbp_mmhg']:.0f}"
    bp_stage_str = f"Stage: {pred['bp_stage']}"
    ptt_str = f"{pred['measured_ptt_ms']:.1f}"
    rise_str = f"PPG Rise: {pred['ppg_rise_time_ms']:.0f} ms"
    hr_str = f"{pred['measured_hr_bpm']:.1f}"
    hrv_str = f"HRV RMSSD: {pred['hrv_rmssd_ms']:.1f} ms"
    spo2_str = f"{pred['spo2_percent']:.1f}"
    sqi_str = f"SQI: {pred['sqi_percent']:.0f}%"
    resp_str = f"{pred['estimated_resp_bpm']:.1f}"
    stress_str = f"Stress: {pred['stress_score']:.0f}/100"
    pkt_str = f"{pkt_stats['data_rate_kbps']}"
    crc_str = f"CRC-8 OK | {pkt_stats['hdf5_size_kb']}KB"

    if pred["sqi_percent"] < 50.0:
        banner_style = {"padding": "8px 16px", "borderRadius": "6px", "fontWeight": "700", "fontSize": "13px", "backgroundColor": "#78350F", "color": "#FDE68A"}
    elif pred["anomaly_detected"]:
        banner_style = {"padding": "8px 16px", "borderRadius": "6px", "fontWeight": "700", "fontSize": "13px", "backgroundColor": "#7F1D1D", "color": "#FECACA"}
    else:
        banner_style = {"padding": "8px 16px", "borderRadius": "6px", "fontWeight": "700", "fontSize": "13px", "backgroundColor": "#065F46", "color": "#A7F3D0"}

    hdf5_msg = f"Active HDF5: cardiosense_session.h5 ({pkt_stats['hdf5_size_kb']} KB)"

    fig = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
        subplot_titles=(
            f"1. ECG & PPG Peak Locking | Filters: {'ON' if enable_filters else 'OFF'}",
            f"2. Beat-to-Beat PTT Trend & Est. BP ({pred['bp_stage']})",
            f"3. Raw PPG vs Extracted Respiration Wave ({pred['estimated_resp_bpm']} Br/min)",
            f"4. Dr. Batra Spectrogram | SQI: {pred['sqi_percent']}%"
        )
    )

    ecg_norm = pred["ecg_filtered"] / (np.max(np.abs(pred["ecg_filtered"])) + 1e-9)
    ppg_norm = pred["ppg_filtered"] / (np.max(np.abs(pred["ppg_filtered"])) + 1e-9)

    fig.add_trace(go.Scatter(x=pred["t_ecg"], y=ecg_norm, name="ECG", line=dict(color=ACCENT_CYAN, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["t_ppg"], y=ppg_norm, name="PPG", line=dict(color=ACCENT_PINK, width=2.0)), row=1, col=1)

    r_idx = pred["r_peaks_idx"]
    p_idx = pred["ppg_peaks_idx"]
    fig.add_trace(go.Scatter(x=pred["r_peaks_time"], y=ecg_norm[r_idx], mode="markers", name="ECG R-Peak",
                             marker=dict(color="#22C55E", size=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["ppg_peaks_time"], y=ppg_norm[p_idx], mode="markers", name="PPG Peak",
                             marker=dict(color="#FACC15", size=9, symbol="diamond")), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pred["paired_ecg_times"],
        y=pred["ptt_array_ms"],
        mode="lines+markers+text",
        name="Beat PTT",
        text=[f"{v:.0f}ms" for v in pred["ptt_array_ms"]],
        textposition="top center",
        line=dict(color="#A855F7", width=2.5),
        marker=dict(size=8, color="#FACC15")
    ), row=1, col=2)

    fig.add_trace(go.Scatter(x=pred["t_ppg"], y=pred["ppg_raw"], name="Raw PPG",
                             line=dict(color="rgba(255,165,0,0.4)", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred["t_ppg"], y=pred["resp_wave"] * 2.5, name="Respiration",
                             line=dict(color="#38BDF8", width=2.5)), row=2, col=1)

    fig.add_trace(go.Heatmap(
        x=pred["spec_times"],
        y=pred["spec_freqs"],
        z=pred["spec_power_db"],
        colorscale="Inferno",
        showscale=False,
        name="Spectrogram"
    ), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor="#0B0F19",
        margin=dict(l=40, r=20, t=40, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(range=[0, 4.5], title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(range=[0, 10.0], title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(range=[100, 380], title_text="PTT (ms)", row=1, col=2)
    fig.update_xaxes(range=[0, 10.0], title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)

    return (
        bp_str, bp_stage_str, ptt_str, rise_str, hr_str, hrv_str,
        spo2_str, sqi_str, resp_str, stress_str, pkt_str, crc_str,
        pred["anomaly_status"], banner_style, hdf5_msg, fig
    )


if __name__ == "__main__":
    print("\n==================================================================")
    print("  CardioSense Live Dashboard Ready!")
    print("  Open this link in your browser: http://127.0.0.1:8050")
    print("==================================================================\n")
    app.run(debug=False, port=8050)