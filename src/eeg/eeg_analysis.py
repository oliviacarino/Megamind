"""
megamind/eeg_analysis.py
─────────────────────────────────────────────────────────────
Scale-independent alpha power and ERD detection.
Works with both synthetic (gain ~10,000x) and real adapted
data (gain ~2,663x) without changing any thresholds.

Key insight: use bandpass RMS instead of FFT magnitude.
FFT magnitude scales with absolute amplitude — so a signal
at gain 2663x gives much smaller FFT values than one at 10000x.
Bandpass RMS is a voltage — it scales correctly with the signal
and has physical meaning (V_rms in the alpha band).

ERD threshold is auto-calibrated from the first N seconds
of each session rather than hardcoded.
"""

import numpy as np
from scipy import signal as sp_signal

SAMPLE_RATE = 250
ALPHA_LOW   = 8.0
ALPHA_HIGH  = 12.0
FILTER_ORDER = 4


def alpha_rms(samples: list[float], sr: int = SAMPLE_RATE) -> float:
    """
    RMS voltage in the alpha band (8-12 Hz).

    Uses a 4th-order Butterworth bandpass filter then computes
    RMS of the filtered signal. This is scale-independent —
    the result is in the same units (volts) regardless of gain.

    Returns 0.0 if fewer than 1 second of samples provided.
    """
    if len(samples) < sr:
        return 0.0
    arr = np.array(samples, dtype=np.float32)
    sos = sp_signal.butter(FILTER_ORDER, [ALPHA_LOW, ALPHA_HIGH],
                           btype='band', fs=sr, output='sos')
    filtered = sp_signal.sosfiltfilt(sos, arr)
    return float(np.sqrt(np.mean(filtered ** 2)))


def calibrate_threshold(baseline_samples: list[float],
                        erd_fraction: float = 0.70,
                        sr: int = SAMPLE_RATE) -> float:
    """
    Calculate ERD detection threshold from a baseline window.

    ERD is defined as alpha power dropping below a fraction of
    the resting baseline. Default: 70% of baseline (30% drop).
    Literature range: 40-60% drop (Pfurtscheller & Neuper 1997).

    Parameters
    ----------
    baseline_samples : voltage buffer from a resting-state window
                       (10+ seconds recommended)
    erd_fraction     : threshold = baseline_rms × erd_fraction
                       0.70 = trigger when power drops 30% below rest

    Returns
    -------
    threshold voltage (float) — pass to classify_state()
    """
    base = alpha_rms(baseline_samples, sr)
    return base * erd_fraction


def classify_state(samples: list[float],
                   threshold: float,
                   sr: int = SAMPLE_RATE) -> dict:
    """
    Classify cognitive state from EEG buffer.

    Parameters
    ----------
    samples   : rolling voltage buffer (5s = 1250 samples recommended)
    threshold : ERD trigger level in volts, from calibrate_threshold()

    Returns
    -------
    {
        "alpha_rms": float,   # RMS voltage in alpha band
        "erd":       bool,    # True = alpha suppressed = motor imagery
        "state":     str,     # "motor_imagery" | "relaxed"
        "threshold": float,   # threshold used
    }
    """
    power = alpha_rms(samples, sr)
    erd   = power < threshold
    return {
        "alpha_rms": round(power, 6),
        "erd":       erd,
        "state":     "motor_imagery" if erd else "relaxed",
        "threshold": round(threshold, 6),
    }


# Quick self-test
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from eeg_simulator import EEGSimulator, EEGState, SAMPLE_RATE as SR

    print("Testing on simulator data...")
    sim = EEGSimulator(state=EEGState.RELAXED)

    # Collect 10s baseline
    baseline = [sim.next_sample()[0] for _ in range(SR * 10)]
    threshold = calibrate_threshold(baseline)
    print(f"Baseline RMS: {alpha_rms(baseline):.5f}V")
    print(f"Threshold:    {threshold:.5f}V  (70% of baseline)")

    # Test relaxed state
    sim.set_state(EEGState.RELAXED)
    buf = [sim.next_sample()[0] for _ in range(SR * 5)]
    r = classify_state(buf, threshold)
    print(f"\nRelaxed:   α={r['alpha_rms']:.5f}V  erd={r['erd']}  state={r['state']}")

    # Test motor imagery state
    sim.set_state(EEGState.IMAGINING)
    buf2 = [sim.next_sample()[0] for _ in range(SR * 5)]
    r2 = classify_state(buf2, threshold)
    print(f"Imagery:   α={r2['alpha_rms']:.5f}V  erd={r2['erd']}  state={r2['state']}")

    assert r['erd']  == False, "Relaxed state should NOT trigger ERD"
    assert r2['erd'] == True,  "Motor imagery should trigger ERD"
    print("\nPASS ✓")
