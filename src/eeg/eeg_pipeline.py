"""
megamind/eeg_pipeline.py
─────────────────────────────────────────────────────────────
Unified EEG pipeline — real hardware OR synthetic simulation,
selected by a single flag.

Usage:
    python eeg_pipeline.py --source sim    # synthetic (default, laptop)
    python eeg_pipeline.py --source raw_data --edf S001R04.edf  # adapted PhysioNet EDF

The --source flag is the ONLY difference between demo and
production mode. All downstream processing is identical.

Hardware dependencies (Raspberry Pi only):
    pip install adafruit-circuitpython-ads1x15 --break-system-packages
"""

import argparse
import time
import json
import threading
from collections import deque

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

from eeg_simulator import (
    EEGSimulator, EEGState, alpha_power, classify_state,
    SAMPLE_RATE, BUFFER_SIZE
)

# ── Optional: real hardware (Raspberry Pi + ADS1015) ───────────
try:
    import board, busio
    import adafruit_ads1x15.ads1015 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
# Real hardware reader
# ════════════════════════════════════════════════════════════════

class HardwareEEGReader:
    """
    Reads voltage from ADS1015 ADC over I2C.
    Wiring matches ryanlopezzzz/EEG:
      ADS1015 P0 → circuit output (alpha-filtered signal)
      I2C SDA/SCL on Raspberry Pi GPIO 2/3
    """

    SOURCE = "raw_data"

    def __init__(self, sample_rate=SAMPLE_RATE):
        if not HW_AVAILABLE:
            raise RuntimeError(
                "Hardware libraries missing.\n"
                "  pip install adafruit-circuitpython-ads1x15 --break-system-packages\n"
                "Or run with --source sim"
            )
        self.sample_rate = sample_rate
        i2c         = busio.I2C(board.SCL, board.SDA)
        ads         = ADS.ADS1015(i2c)
        ads.gain    = 1             # ±4.096 V range
        self._chan  = AnalogIn(ads, ADS.P0)
        self._buf   = deque(maxlen=BUFFER_SIZE)
        print(f"[HW] ADS1015 connected · {sample_rate} SPS")

    def next_sample(self):
        v = self._chan.voltage
        self._buf.append(v)
        return v, "raw_data"

    def get_buffer(self):
        return list(self._buf)

    def stream(self, realtime=True):
        interval = 1.0 / self.sample_rate
        while True:
            t0 = time.perf_counter()
            yield self.next_sample()
            if realtime:
                s = interval - (time.perf_counter() - t0)
                if s > 0:
                    time.sleep(s)


# ════════════════════════════════════════════════════════════════
# Pipeline — wraps reader, runs analysis, notifies subscribers
# ════════════════════════════════════════════════════════════════

class MegamindPipeline:
    """
    Reads EEG samples, computes alpha power every 200 ms,
    and calls registered subscribers with a JSON-serializable frame.

    Frame format:
    {
      "source":      "raw_data" | "sim",
      "voltage":     1.72,          # latest ADC reading (V)
      "alpha_power": 18.4,          # RMS power in 8–12 Hz band
      "erd":         false,         # True = alpha suppressed = motor imagery
      "state":       "relaxed",     # "relaxed" | "motor_imagery"
      "t":           4.2            # seconds since pipeline start
    }
    """

    ANALYSIS_HZ = 5   # state updates per second

    def __init__(self, reader, verbose=True):
        self.reader    = reader
        self.verbose   = verbose
        self._buf      = deque(maxlen=BUFFER_SIZE)
        self._cbs      = []
        self._running  = False

    def subscribe(self, fn):
        self._cbs.append(fn)

    def _emit(self, frame):
        for fn in self._cbs:
            try:
                fn(frame)
            except Exception as e:
                print(f"[pipeline] subscriber error: {e}")

    def run(self):
        """Blocking. Wrap in a thread for parallel use."""
        self._running   = True
        source          = self.reader.SOURCE
        interval        = 1.0 / self.ANALYSIS_HZ
        last_analysis   = time.time()
        t0              = time.time()

        print(f"[pipeline] source={source}  {SAMPLE_RATE} SPS  analysis@{self.ANALYSIS_HZ}Hz")

        for voltage, _ in self.reader.stream(realtime=True):
            if not self._running:
                break
            self._buf.append(voltage)

            now = time.time()
            if (now - last_analysis) >= interval:
                last_analysis = now
                result = classify_state(list(self._buf))
                frame  = {
                    "source":  source,
                    "voltage": round(voltage, 4),
                    "t":       round(now - t0, 2),
                    **result,
                }
                self._emit(frame)
                if self.verbose:
                    bar = "█" * int(result["alpha_power"] / 2)
                    erd = " ← ERD" if result["erd"] else ""
                    print(
                        f"  [{source.upper()}] "
                        f"t={frame['t']:6.1f}s | "
                        f"state={result['state']:<14} | "
                        f"α={result['alpha_power']:5.1f} {bar}{erd}"
                    )

    def stop(self):
        self._running = False


# ════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════

def build_reader(source: str, edf_path: str = None, npy_path: str = None):
    if source == "raw_data":
        return HardwareEEGReader()
    elif source == "raw_data":
        from eeg_raw_data import RawEEGReader
        if not edf_path and not npy_path:
            raise ValueError(
                "--source raw_data requires --edf <path> or --npy <path>\n"
                "  Example: python eeg_pipeline.py --source raw_data --edf S001R04.edf"
            )
        print("[RAW_DATA] Using adapted PhysioNet EEGMMIDB recording")
        return RawEEGReader(edf_path=edf_path, npy_path=npy_path)
    else:
        sim = EEGSimulator(state=EEGState.RELAXED)
        print("[SIM] Synthetic EEG — parameterized from PhysioNet EEGMMIDB statistics")
        return sim


def main():
    parser = argparse.ArgumentParser(description="Megamind EEG pipeline")
    parser.add_argument(
        "--source",
        choices=["raw_data", "sim"],
        default="sim",
        help=(
            "'sim'       = synthetic data parameterized from EEGMMIDB stats (default)\n"
            "'raw_data'      = live ADS1015 hardware via I2C (Raspberry Pi)\n"
            "'raw_data' = adapted PhysioNet EEGMMIDB .edf file (requires --edf or --npy)"
        )
    )
    parser.add_argument(
        "--edf", default=None,
        help="Path to EEGMMIDB .edf file (used with --source raw_data)"
    )
    parser.add_argument(
        "--npy", default=None,
        help="Path to pre-adapted .npy cache (used with --source raw_data, fastest)"
    )
    args = parser.parse_args()

    reader   = build_reader(args.source, edf_path=args.edf, npy_path=args.npy)
    pipeline = MegamindPipeline(reader, verbose=True)

    # Example subscriber: print JSON frames
    def print_frame(frame):
        pass  # verbose=True in pipeline already prints; add your own logic here

    pipeline.subscribe(print_frame)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
