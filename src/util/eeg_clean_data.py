"""
megamind/eeg_raw_data.py
─────────────────────────────────────────────────────────────
Adapts a PhysioNet EEGMMIDB .edf file to match the exact output
of the ryanlopezzzz/EEG circuit.

Transformation steps:
  1. Read EDF with pyedflib (lenient reader, works on all PhysioNet files)
     Falls back to MNE if pyedflib unavailable.
  2. Select O2-Fp2 bipolar channel (matches our electrode placement)
  3. Resample 160 → 250 SPS (polyphase, anti-aliased, exact 25/16 ratio)
  4. Scale µV → volts: V = (µV × 1e-6 × gain) + 1.65V bias
  5. Re-inject circuit artifacts: PLI residual, skin drift, ADC quantization
  6. Clamp to 0–3.3V ADC rail

Quickstart:
    # Adapt a file (run once, caches .npy alongside .edf)
    python eeg_raw_data.py S001R04.edf --plot

    # Use in pipeline
    python eeg_pipeline.py --source raw_data --edf S001R04.edf
    # or faster after first run:
    python eeg_pipeline.py --source raw_data --npy S001R04_adapted.npy

Motor imagery runs: R04, R08, R12  (imagined left/right hand)
Dataset: https://physionet.org/content/eegmmidb/1.0.0/
"""

import os
import time
import math
import random
import threading
from collections import deque

import numpy as np
from scipy import signal as sp_signal

# ── Constants ────────────────────────────────────────────────
SAMPLE_RATE  = 250
BUFFER_SIZE  = SAMPLE_RATE * 5
EEGMMIDB_SR  = 160           # native dataset sample rate

CIRCUIT_GAIN = 10_000        # total amp chain gain
ADC_BIAS     = 1.65          # V — 3.3V supply at midpoint
ADC_MIN      = 0.0
ADC_MAX      = 3.3
ADS1015_LSB  = 8.192 / 4096  # 2 mV per step (12-bit ±4.096V)

PLI_AMP      = 0.02          # V — 60 Hz residual after notch
DRIFT_AMP    = 0.05          # V — galvanic skin drift
DRIFT_FREQ   = 0.03          # Hz


# ═══════════════════════════════════════════════════════════════
# EDF reader — pyedflib first, MNE fallback
# ═══════════════════════════════════════════════════════════════

def _read_edf(edf_path: str) -> tuple[dict, int, int]:
    """
    Read an EDF file and return:
        (ch_dict, n_samples, sample_rate)

    ch_dict maps channel name → np.ndarray in µV.

    Uses pyedflib as primary reader because it handles the older
    EDF format used by PhysioNet without throwing 'Bad EDF file'.
    Falls back to MNE if pyedflib is not installed.
    """
    try:
        return _read_edf_pyedflib(edf_path)
    except ImportError:
        print("[adapter] pyedflib not found, trying MNE...")
        return _read_edf_mne(edf_path)


def _read_edf_pyedflib(edf_path: str) -> tuple[dict, int, int]:
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pip install pyedflib --break-system-packages")

    f        = pyedflib.EdfReader(edf_path)
    n_ch     = f.signals_in_file
    labels   = f.getSignalLabels()
    sr_arr   = f.getSampleFrequencies()
    dims     = [f.getPhysicalDimension(i) for i in range(n_ch)]

    # Verify all channels share the same sample rate
    sr = int(sr_arr[0])
    if len(set(sr_arr)) > 1:
        print(f"[adapter] Warning: mixed sample rates {set(sr_arr)}, using {sr}")

    ch_dict = {}
    n_samples = 0
    for i in range(n_ch):
        raw = f.readSignal(i)               # returns physical values
        # Normalise label: strip dots/spaces (PhysioNet uses "Oz." etc.)
        label = labels[i].strip().rstrip('.').strip()
        # Convert to µV if needed (some files store in V or mV)
        dim = dims[i].strip().upper()
        if dim in ("V",):
            raw = raw * 1e6
        elif dim in ("MV", "MILLIVOLT"):
            raw = raw * 1e3
        # else already µV
        ch_dict[label] = raw.astype(np.float32)
        n_samples = max(n_samples, len(raw))

    f._close()
    return ch_dict, n_samples, sr


def _read_edf_mne(edf_path: str) -> tuple[dict, int, int]:
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        raise ImportError(
            "Neither pyedflib nor mne is available.\n"
            "  pip install pyedflib --break-system-packages"
        )

    # Try several MNE loading strategies for compatibility
    raw = None
    errors = []
    strategies = [
        dict(preload=True, verbose=False),
        dict(preload=True, verbose=False, infer_types=True),
        dict(preload=True, verbose=False, encoding="latin1"),
    ]
    for kwargs in strategies:
        try:
            raw = mne.io.read_raw_edf(edf_path, **kwargs)
            break
        except Exception as e:
            errors.append(str(e))

    if raw is None:
        raise RuntimeError(
            "MNE could not read this EDF file.\n"
            "Install pyedflib for reliable reading:\n"
            "  pip install pyedflib --break-system-packages\n"
            f"MNE errors: {errors}"
        )

    try:
        mne.datasets.eegbci.standardize(raw)
    except Exception:
        pass  # not critical, just normalizes channel name dots

    data, _  = raw[:]
    sr       = int(raw.info["sfreq"])
    ch_names = raw.ch_names
    # MNE stores in V → convert to µV
    ch_dict  = {name.strip().rstrip('.'): data[i] * 1e6
                for i, name in enumerate(ch_names)}
    return ch_dict, data.shape[1], sr


# ═══════════════════════════════════════════════════════════════
# Signal processing helpers
# ═══════════════════════════════════════════════════════════════

def _find_channel(ch_dict: dict, candidates: list) -> tuple[str, np.ndarray]:
    """Return (name, array) for first matching channel."""
    for name in candidates:
        if name in ch_dict:
            return name, ch_dict[name]
    # Case-insensitive fuzzy match
    for cand in candidates:
        for ch in ch_dict:
            if cand.lower() == ch.lower():
                return ch, ch_dict[ch]
    # Partial match fallback
    for cand in candidates:
        for ch in ch_dict:
            if cand.lower() in ch.lower():
                print(f"[adapter] Using '{ch}' as fallback for '{cand}'")
                return ch, ch_dict[ch]
    raise ValueError(
        f"None of {candidates} found.\n"
        f"Available channels: {list(ch_dict.keys())}"
    )


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample with anti-alias FIR filter."""
    from math import gcd
    g = gcd(src_sr, dst_sr)
    return sp_signal.resample_poly(data, dst_sr // g, src_sr // g)


# ═══════════════════════════════════════════════════════════════
# Main adaptation function
# ═══════════════════════════════════════════════════════════════

def adapt_edf(edf_path: str,
              output_npy: str = None,
              channel_mode: str = "O2-Fp2",
              gain: float = CIRCUIT_GAIN) -> np.ndarray:
    """
    Full pipeline: EDF file → circuit-compatible voltage array.

    Parameters
    ----------
    edf_path     : path to a .edf file from EEGMMIDB
    output_npy   : save result to this .npy path (recommended for reuse)
    channel_mode : "O2-Fp2" bipolar (default, matches our electrodes) or "Oz"
    gain         : circuit gain. Reduce if >1% of signal clips.

    Returns
    -------
    np.ndarray float32, shape (n_samples,), values in [0.0, 3.3] V
    """

    # ── 1. Read EDF ───────────────────────────────────────────
    print(f"[adapter] Reading {os.path.basename(edf_path)}...")
    ch_dict, n_raw, actual_sr = _read_edf(edf_path)
    print(f"[adapter] {len(ch_dict)} channels · "
          f"{n_raw} samples · {n_raw/actual_sr:.1f}s @ {actual_sr} SPS")
    print(f"[adapter] Channels found: {list(ch_dict.keys())[:8]}{'...' if len(ch_dict) > 8 else ''}")

    # ── 2. Channel selection ──────────────────────────────────
    if channel_mode == "O2-Fp2":
        o2_name,  o2  = _find_channel(ch_dict, ["O2", "o2"])
        fp2_name, fp2 = _find_channel(ch_dict, ["Fp2", "FP2", "fp2", "Fpz"])
        mono = o2 - fp2
        print(f"[adapter] Bipolar: {o2_name} − {fp2_name} (matches our electrode placement)")
    else:
        oz_name, mono = _find_channel(ch_dict, ["Oz", "O2", "O1"])
        print(f"[adapter] Single channel: {oz_name}")

    # ── 3. Resample to circuit sample rate ────────────────────
    if actual_sr != SAMPLE_RATE:
        mono = _resample(mono, actual_sr, SAMPLE_RATE)
        print(f"[adapter] Resampled {actual_sr}→{SAMPLE_RATE} SPS "
              f"({n_raw}→{len(mono)} samples)")
    else:
        print(f"[adapter] Sample rate already {SAMPLE_RATE} SPS — no resample needed")

    # ── 4. Scale µV → circuit voltage ────────────────────────
    volts    = (mono * 1e-6 * gain) + ADC_BIAS
    clip_pct = float(np.mean((volts < ADC_MIN) | (volts > ADC_MAX)) * 100)
    v_min, v_max = float(volts.min()), float(volts.max())

    if clip_pct > 1.0:
        # Auto-suggest a gain that keeps signal within rail with 10% headroom
        headroom = max(abs(v_min - ADC_BIAS), abs(v_max - ADC_BIAS))
        suggested = int(gain * (ADC_BIAS * 0.9) / headroom)
        print(f"[adapter] ⚠  {clip_pct:.1f}% clipping at gain={gain:.0f}. "
              f"Try --gain {suggested}")
    else:
        print(f"[adapter] Amplitude OK · [{v_min:.3f}, {v_max:.3f}]V · "
              f"{clip_pct:.2f}% clipping")

    # ── 5. Re-inject circuit artifacts ───────────────────────
    n      = len(volts)
    t_arr  = np.arange(n) / SAMPLE_RATE
    drift_ph = random.uniform(0, 2 * math.pi)

    volts += PLI_AMP   * np.sin(2 * np.pi * 60.0       * t_arr)
    volts += DRIFT_AMP * np.sin(2 * np.pi * DRIFT_FREQ  * t_arr + drift_ph)
    volts  = np.round(volts / ADS1015_LSB) * ADS1015_LSB  # 12-bit quantization
    volts  = np.clip(volts, ADC_MIN, ADC_MAX).astype(np.float32)

    print(f"[adapter] Artifacts: PLI {PLI_AMP*1000:.0f}mV · "
          f"drift {DRIFT_AMP*1000:.0f}mV · quant {ADS1015_LSB*1000:.0f}mV LSB")

    # ── 6. Save ───────────────────────────────────────────────
    if output_npy:
        np.save(output_npy, volts)
        sz = os.path.getsize(output_npy) / 1024
        print(f"[adapter] Saved → {output_npy} ({sz:.0f} KB)")

    return volts


# ═══════════════════════════════════════════════════════════════
# Reader class — same interface as EEGSimulator
# ═══════════════════════════════════════════════════════════════

class RawEEGReader:
    """
    Streams adapted EEGMMIDB data through the same interface as
    EEGSimulator so eeg_pipeline.py works with zero changes.

    Pass either an .edf path (adapted on first use, then cached as
    .npy alongside the file) or a pre-adapted .npy for instant startup.

        reader.next_sample() → (float voltage, "raw_data")
        reader.stream()      → generator
        reader.get_buffer()  → list[float]
        reader.SOURCE        → "raw_data"
    """

    SOURCE = "raw_data"

    def __init__(self, edf_path: str = None, npy_path: str = None,
                 loop: bool = True, gain: float = CIRCUIT_GAIN,
                 channel_mode: str = "O2-Fp2"):

        if npy_path and os.path.exists(npy_path):
            self._samples = np.load(npy_path)
            print(f"[RawEEG] Loaded {npy_path} "
                  f"({len(self._samples)/SAMPLE_RATE:.1f}s)")

        elif edf_path:
            # Auto-detect cache path alongside the .edf
            cache = (edf_path
                     .replace(".edf", "_adapted.npy")
                     .replace(".EDF", "_adapted.npy"))
            if os.path.exists(cache):
                self._samples = np.load(cache)
                print(f"[RawEEG] Loaded cache {cache} "
                      f"({len(self._samples)/SAMPLE_RATE:.1f}s)")
            else:
                self._samples = adapt_edf(
                    edf_path, output_npy=cache,
                    channel_mode=channel_mode, gain=gain
                )
        else:
            raise ValueError("Provide edf_path or npy_path")

        self._idx    = 0
        self._loop   = loop
        self._buffer = deque(maxlen=BUFFER_SIZE)
        self._lock   = threading.Lock()

    def next_sample(self) -> tuple[float, str]:
        with self._lock:
            v = float(self._samples[self._idx])
            self._idx += 1
            if self._idx >= len(self._samples):
                self._idx = 0 if self._loop else len(self._samples) - 1
                if self._loop:
                    print("[RawEEG] Looping")
            self._buffer.append(v)
        return v, self.SOURCE

    def get_buffer(self) -> list[float]:
        with self._lock:
            return list(self._buffer)

    def stream(self, realtime=True):
        interval = 1.0 / SAMPLE_RATE
        while True:
            t0 = time.perf_counter()
            yield self.next_sample()
            if realtime:
                s = interval - (time.perf_counter() - t0)
                if s > 0:
                    time.sleep(s)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Adapt a PhysioNet EEGMMIDB .edf file to circuit-compatible voltage"
    )
    parser.add_argument("edf",
        help="Path to .edf file (e.g. S001R04.edf)")
    parser.add_argument("--out", default=None,
        help="Output .npy path (default: <name>_adapted.npy)")
    parser.add_argument("--channel", choices=["O2-Fp2", "Oz"], default="O2-Fp2",
        help="Channel: O2-Fp2 bipolar (default, matches electrodes) or Oz single")
    parser.add_argument("--gain", type=float, default=CIRCUIT_GAIN,
        help=f"Circuit gain (default {CIRCUIT_GAIN}). Reduce if signal clips.")
    parser.add_argument("--plot", action="store_true",
        help="Save waveform + power spectrum preview PNG")
    parser.add_argument("--info", action="store_true",
        help="Print channel list and exit (useful for debugging)")
    args = parser.parse_args()

    # --info mode: just print channels
    if args.info:
        ch_dict, n, sr = _read_edf(args.edf)
        print(f"\nFile: {args.edf}")
        print(f"Sample rate: {sr} SPS")
        print(f"Duration: {n/sr:.1f}s ({n} samples)")
        print(f"Channels ({len(ch_dict)}):")
        for i, name in enumerate(ch_dict.keys()):
            arr = ch_dict[name]
            print(f"  [{i:2d}] {name:<8}  "
                  f"min={arr.min():7.2f}  max={arr.max():7.2f}  "
                  f"mean={arr.mean():7.2f}  µV")
        raise SystemExit(0)

    out = args.out or (args.edf
                       .replace(".edf", "_adapted.npy")
                       .replace(".EDF", "_adapted.npy"))

    volts = adapt_edf(args.edf, output_npy=out,
                      channel_mode=args.channel, gain=args.gain)

    print(f"\n{'='*56}")
    print(f"  Input:   {args.edf}")
    print(f"  Output:  {out}")
    print(f"  Samples: {len(volts):,} @ {SAMPLE_RATE} SPS = {len(volts)/SAMPLE_RATE:.1f}s")
    print(f"  Range:   [{volts.min():.4f}V, {volts.max():.4f}V]")
    print(f"  Mean:    {volts.mean():.4f}V  (ideal centre: {ADC_BIAS}V)")
    print(f"{'='*56}")
    print(f"\nNext steps:")
    print(f"  python eeg_pipeline.py --source raw_data --edf {args.edf}")
    print(f"  # or faster (skips re-adaptation):")
    print(f"  python eeg_pipeline.py --source raw_data --npy {out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            preview = volts[:SAMPLE_RATE * 10]   # 10-second preview
            t = np.arange(len(preview)) / SAMPLE_RATE

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6),
                                            facecolor="#080c14")
            title = f"Adapted EEG — {os.path.basename(args.edf)} — first 10s"
            fig.suptitle(title, color="#00e5ff", fontsize=11,
                         fontfamily="monospace")

            # Waveform
            ax1.set_facecolor("#080c14")
            ax1.plot(t, preview, color="#00e5ff", lw=0.6, alpha=0.9)
            ax1.axhline(ADC_BIAS, color="#4a6080", lw=0.6, linestyle="--",
                        label=f"ADC centre ({ADC_BIAS}V)")
            ax1.axhline(ADC_MIN, color="#ff6b35", lw=0.5, linestyle=":")
            ax1.axhline(ADC_MAX, color="#ff6b35", lw=0.5, linestyle=":",
                        label="ADC rail (0 / 3.3V)")
            ax1.set_ylim(-0.1, 3.4)
            ax1.set_ylabel("Voltage (V)", color="#c8d8f0", fontsize=9)
            ax1.set_xlabel("Time (s)", color="#c8d8f0", fontsize=9)
            ax1.legend(fontsize=8, labelcolor="#c8d8f0",
                       facecolor="#0d1422", edgecolor="#1a2540")
            ax1.tick_params(colors="#4a6080")
            for sp in ax1.spines.values():
                sp.set_edgecolor("#1a2540")

            # Power spectrum
            ax2.set_facecolor("#080c14")
            f_arr, psd = sp_signal.welch(
                preview - preview.mean(), fs=SAMPLE_RATE, nperseg=SAMPLE_RATE
            )
            ax2.semilogy(f_arr, psd, color="#7c3aed", lw=1.2)
            ax2.axvspan(8, 12, alpha=0.18, color="#00e5ff",
                        label="Alpha band (8–12 Hz)")
            ax2.axvline(60, color="#ff6b35", lw=0.8, linestyle="--",
                        alpha=0.6, label="60 Hz PLI")
            ax2.set_xlim(0, 80)
            ax2.set_ylabel("PSD (V²/Hz)", color="#c8d8f0", fontsize=9)
            ax2.set_xlabel("Frequency (Hz)", color="#c8d8f0", fontsize=9)
            ax2.legend(fontsize=8, labelcolor="#c8d8f0",
                       facecolor="#0d1422", edgecolor="#1a2540")
            ax2.tick_params(colors="#4a6080")
            for sp in ax2.spines.values():
                sp.set_edgecolor("#1a2540")

            plt.tight_layout()
            preview_path = out.replace(".npy", "_preview.png")
            plt.savefig(preview_path, dpi=150, facecolor="#080c14")
            print(f"\n  Preview → {preview_path}")

        except ImportError:
            print("  (pip install matplotlib for --plot)")
