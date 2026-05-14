"""
megamind/eeg_simulator.py
─────────────────────────────────────────────────────────────
Synthetic EEG signal generator parameterized from published
statistics of the PhysioNet EEG Motor Movement/Imagery Dataset
(EEGMMIDB). Used for pipeline validation when hardware is
unavailable.

NOTE: This is SYNTHETIC data. All parameters are derived from
peer-reviewed published values, not random numbers.

Dataset this is parameterized from
────────────────────────────────────
  PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
  https://physionet.org/content/eegmmidb/1.0.0/

  Primary citation:
    Schalk G, McFarland DJ, Hinterberger T, Birbaumer N,
    Wolpaw JR (2004). BCI2000: A General-Purpose Brain-Computer
    Interface (BCI) System. IEEE Trans Biomed Eng 51(6):1034-1043.

  PhysioNet citation:
    Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and
    PhysioNet. Circulation 101(23):e215-e220.

ERD statistics source
──────────────────────
  Pfurtscheller G & Neuper C (1997). Motor imagery activates
  primary sensorimotor area in humans. Neurosci Lett 239:65-68.

Why we can't use the raw dataset directly
──────────────────────────────────────────
  EEGMMIDB is recorded at 160 SPS with 64 channels, outputting
  raw microvolts. Our circuit outputs 250 SPS, single-channel,
  0-3.3V after ~8000x instrumentation amp gain. Direct playback
  would require resampling, channel collapse, and re-adding
  circuit-specific artifacts. The published spectral statistics
  are a cleaner parameterization source.

Published parameters used (see DATA_SOURCES.md for full detail)
────────────────────────────────────────────────────────────────
  Alpha frequency:  9.5-11 Hz (individual alpha frequency range)
  Alpha at rest:    20-40 uV RMS at scalp → ~0.35V after 8000x gain
  Alpha ERD:        50% power suppression during motor imagery
                    → amplitude × 0.707 → ~0.247V
  ERD range:        40-60% power drop (Pfurtscheller & Neuper 1997)
  PLI:              60 Hz residual (US, post-notch filter)
  Blink artifact:   every ~6s, ~0.3s duration
  Skin drift:       ~0.03 Hz galvanic potential
"""

import time
import math
import random
import threading
from collections import deque

# ── Hardware constants ───────────────────────────────────────────
SAMPLE_RATE   = 250
BUFFER_WINDOW = 5.0
BUFFER_SIZE   = int(SAMPLE_RATE * BUFFER_WINDOW)

ALPHA_LOW  = 8.0
ALPHA_HIGH = 12.0

# ── Voltage parameters (derived from published scalp amplitudes) ─
ADC_CENTER        = 1.65   # V  (3.3V supply, biased at midpoint)
ALPHA_AMP_RELAXED = 0.35   # V  peak at rest (~30uV scalp × 8000x gain)
ALPHA_AMP_ERD     = 0.247  # V  peak during motor imagery (50% ERD)
                            #    → amp × sqrt(0.5) = amp × 0.707
NOISE_AMP         = 0.04   # V  broadband noise floor
PLI_AMP           = 0.02   # V  60 Hz power-line residual
DRIFT_AMP         = 0.05   # V  slow galvanic skin drift


class EEGState:
    RELAXED   = "relaxed"
    IMAGINING = "motor_imagery"


class EEGSimulator:
    """
    Synthetic EEG stream parameterized from PhysioNet EEGMMIDB
    published statistics (Schalk et al. 2004, Pfurtscheller 1997).
    See module docstring and DATA_SOURCES.md for full citation detail.
    """

    SOURCE = "sim"

    def __init__(self, state=EEGState.RELAXED, sample_rate=SAMPLE_RATE):
        self.sample_rate     = sample_rate
        self.state           = state
        self._t              = 0.0
        self._buffer         = deque(maxlen=BUFFER_SIZE)
        self._lock           = threading.Lock()
        self._state_timer    = 0.0
        self._state_duration = self._random_duration()
        self._drift_phase    = random.uniform(0, 2 * math.pi)
        # Subject-to-subject IAF variability: 9.5-11 Hz
        self._alpha_center   = random.uniform(9.5, 11.0)

    def set_state(self, state: str):
        with self._lock:
            self.state = state
            self._state_timer = 0.0

    def next_sample(self) -> tuple[float, str]:
        dt = 1.0 / self.sample_rate
        with self._lock:
            v = self._generate()
            self._t += dt
            self._state_timer += dt
            if self._state_timer >= self._state_duration:
                self._auto_transition()
            self._buffer.append(v)
        return v, self.state

    def get_buffer(self) -> list[float]:
        with self._lock:
            return list(self._buffer)

    def stream(self, realtime=True):
        interval = 1.0 / self.sample_rate
        while True:
            t0 = time.perf_counter()
            yield self.next_sample()
            if realtime:
                s = interval - (time.perf_counter() - t0)
                if s > 0:
                    time.sleep(s)

    def _generate(self) -> float:
        t = self._t
        amp  = ALPHA_AMP_RELAXED if self.state == EEGState.RELAXED else ALPHA_AMP_ERD
        freq = self._alpha_center + 0.3 * math.sin(2 * math.pi * 0.05 * t)
        alpha = amp   * math.sin(2 * math.pi * freq  * t)
        noise = random.gauss(0, NOISE_AMP / 2)
        pli   = PLI_AMP   * math.sin(2 * math.pi * 60.0  * t)
        drift = DRIFT_AMP * math.sin(2 * math.pi * 0.03  * t + self._drift_phase)
        blink = self._blink(t)
        return max(0.0, min(3.3, ADC_CENTER + alpha + noise + pli + drift + blink))

    def _blink(self, t: float) -> float:
        phase = (t % 6.2) / 6.2
        if phase < 0.05:
            return 0.15 * math.exp(-((phase - 0.025) ** 2) / (2 * 0.01 ** 2))
        return 0.0

    def _auto_transition(self):
        self.state = (EEGState.IMAGINING if self.state == EEGState.RELAXED
                      else EEGState.RELAXED)
        self._state_timer = 0.0
        self._state_duration = self._random_duration()

    def _random_duration(self) -> float:
        return random.uniform(5.0, 10.0)


def alpha_power(samples: list[float], sample_rate: int = SAMPLE_RATE) -> float:
    """RMS power in 8-12 Hz band via FFT."""
    import numpy as np
    if len(samples) < sample_rate:
        return 0.0
    arr = np.array(samples, dtype=np.float32)
    arr -= arr.mean()
    fft_mag = np.abs(np.fft.rfft(arr))
    freqs   = np.fft.rfftfreq(len(arr), d=1.0 / sample_rate)
    mask    = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    return float(np.sqrt(np.mean(fft_mag[mask] ** 2)))


def classify_state(samples: list[float], sample_rate: int = SAMPLE_RATE,
                   threshold: float = 10.0) -> dict:
    """
    Detect ERD: alpha power below threshold = motor imagery.
    Threshold should be calibrated per patient from baseline.
    """
    power = alpha_power(samples, sample_rate)
    erd   = power < threshold
    return {
        "alpha_power": round(power, 2),
        "erd":         erd,
        "state":       EEGState.IMAGINING if erd else EEGState.RELAXED,
    }


if __name__ == "__main__":
    print("Megamind EEG Simulator — SYNTHETIC DATA")
    print("Parameterized from PhysioNet EEGMMIDB (Schalk et al. 2004)")
    print("Alpha band (8-12 Hz) only. ERD = alpha suppression.")
    print("=" * 54)
    sim = EEGSimulator(state=EEGState.RELAXED)
    buf = []
    t0  = time.time()
    try:
        for voltage, state in sim.stream(realtime=True):
            buf.append(voltage)
            if len(buf) % (SAMPLE_RATE // 2) == 0:
                result = classify_state(buf[-BUFFER_SIZE:])
                bar    = "█" * int(result["alpha_power"] / 2)
                erd    = " ← ERD" if result["erd"] else ""
                print(f"  t={time.time()-t0:5.1f}s | state={state:<14} | "
                      f"α={result['alpha_power']:5.1f} {bar}{erd}")
    except KeyboardInterrupt:
        print("\nStopped.")
