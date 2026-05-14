import json
import math
import random
import socket
import threading
import time
from collections import deque

import tkinter as tk


HOST = "127.0.0.1"
PORT = 9999
FS = 250
BUFFER_SECONDS = 4
MAX_SAMPLES = FS * BUFFER_SECONDS

state = {
    "mode": "neutral",
    "clients": [],
    "running": True,
    "predicted": "neutral",
    "score": 0.0,
    "left_alpha_power": 0.0,
    "right_alpha_power": 0.0,
}

samples = deque(maxlen=MAX_SAMPLES)
samples_lock = threading.Lock()

# alpha = 8-13 Hz; we simulate strongest around 10 Hz
ALPHA_FREQ = 10.0


def make_channel(t, alpha_amp, noise_amp=4.0):
    alpha = alpha_amp * math.sin(2 * math.pi * ALPHA_FREQ * t)

    # small extra brain-ish rhythms/noise
    theta = 6 * math.sin(2 * math.pi * 6 * t)
    beta = 4 * math.sin(2 * math.pi * 18 * t)
    noise = random.gauss(0, noise_amp)

    return alpha + theta + beta + noise


def eeg_pair(t, mode):
    normal_alpha = 40
    suppressed_alpha = 12

    if mode == "left":
        # attending left suppresses alpha more on right-side channel
        left_ch = make_channel(t, normal_alpha)
        right_ch = make_channel(t, suppressed_alpha)
    elif mode == "right":
        # attending right suppresses alpha more on left-side channel
        left_ch = make_channel(t, suppressed_alpha)
        right_ch = make_channel(t, normal_alpha)
    else:
        left_ch = make_channel(t, 28)
        right_ch = make_channel(t, 28)

    return left_ch, right_ch


def classify(left_alpha_power, right_alpha_power):
    total = left_alpha_power + right_alpha_power + 1e-9

    # positive = left attention, negative = right attention
    score = (left_alpha_power - right_alpha_power) / total

    if score > 0.25:
        return "left", score
    if score < -0.25:
        return "right", score
    return "neutral", score


def server_thread():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen()
    srv.settimeout(0.25)

    print(f"Alpha suppression simulator streaming on tcp://{HOST}:{PORT}")

    while state["running"]:
        try:
            conn, _ = srv.accept()
            conn.setblocking(False)
            state["clients"].append(conn)
            print("Client connected")
        except socket.timeout:
            continue
        except OSError:
            break

    srv.close()


def stream_thread():
    sample_index = 0
    next_time = time.perf_counter()

    # crude rolling alpha-power estimate
    left_alpha_energy = 0.0
    right_alpha_energy = 0.0

    while state["running"]:
        now = time.perf_counter()

        if now >= next_time:
            t = sample_index / FS
            mode = state["mode"]

            left_uv, right_uv = eeg_pair(t, mode)

            # estimate alpha by multiplying against 10 Hz reference
            ref = math.sin(2 * math.pi * ALPHA_FREQ * t)
            left_alpha_energy = 0.98 * left_alpha_energy + 0.02 * abs(left_uv * ref)
            right_alpha_energy = 0.98 * right_alpha_energy + 0.02 * abs(right_uv * ref)

            predicted, score = classify(left_alpha_energy, right_alpha_energy)

            state["predicted"] = predicted
            state["score"] = score
            state["left_alpha_power"] = left_alpha_energy
            state["right_alpha_power"] = right_alpha_energy

            packet = {
                "timestamp": time.time(),
                "sample": sample_index,
                "fs": FS,
                "selected_mode": mode,
                "predicted_direction": predicted,
                "attention_score": score,
                "left_uv": left_uv,
                "right_uv": right_uv,
                "left_alpha_power": left_alpha_energy,
                "right_alpha_power": right_alpha_energy,
            }

            with samples_lock:
                samples.append((left_uv, right_uv))

            line = json.dumps(packet) + "\n"

            dead = []
            for c in state["clients"]:
                try:
                    c.sendall(line.encode("utf-8"))
                except Exception:
                    dead.append(c)

            for c in dead:
                try:
                    c.close()
                except Exception:
                    pass
                state["clients"].remove(c)

            sample_index += 1
            next_time += 1 / FS
        else:
            time.sleep(0.001)


def set_mode(mode):
    state["mode"] = mode
    label.config(text=f"Selected: {mode.upper()}")


def make_points(values, width, mid_y, scale):
    if len(values) < 2:
        return []

    x_step = width / (MAX_SAMPLES - 1)
    start_offset = MAX_SAMPLES - len(values)
    points = []

    for i, value in enumerate(values):
        x = (start_offset + i) * x_step
        y = mid_y - value * scale
        points.extend((x, y))

    return points


def draw_waveform():
    canvas.delete("all")

    width = canvas.winfo_width()
    height = canvas.winfo_height()
    mid_y = height / 2

    canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
    canvas.create_line(0, mid_y, width, mid_y, fill="#4b5563")

    for i in range(1, 5):
        y = height * i / 5
        canvas.create_line(0, y, width, y, fill="#1f2937")

    with samples_lock:
        visible = list(samples)

    if len(visible) > 1:
        left_values = [pair[0] for pair in visible]
        right_values = [pair[1] for pair in visible]
        max_abs = max(max(abs(v) for v in left_values + right_values), 1)
        scale = (height * 0.38) / max_abs

        left_points = make_points(left_values, width, mid_y, scale)
        right_points = make_points(right_values, width, mid_y, scale)
        canvas.create_line(left_points, fill="#22d3ee", width=2, smooth=True)
        canvas.create_line(right_points, fill="#f97316", width=2, smooth=True)

    selected = state["mode"].upper()
    predicted = state["predicted"].upper()
    score = state["score"]
    left_power = state["left_alpha_power"]
    right_power = state["right_alpha_power"]

    canvas.create_text(
        12,
        12,
        anchor="nw",
        fill="#e5e7eb",
        font=("Arial", 12),
        text=f"{BUFFER_SECONDS}s live waveform | left cyan | right orange | {FS} Hz",
    )
    canvas.create_text(
        12,
        34,
        anchor="nw",
        fill="#e5e7eb",
        font=("Arial", 12),
        text=(
            f"selected {selected} | predicted {predicted} | "
            f"score {score:+.2f} | alpha L {left_power:.1f} R {right_power:.1f}"
        ),
    )

    prediction_label.config(
        text=(
            f"Predicted: {predicted}    "
            f"Score: {score:+.2f}    "
            f"Alpha power L/R: {left_power:.1f} / {right_power:.1f}"
        )
    )

    if state["running"]:
        root.after(33, draw_waveform)


def on_close():
    state["running"] = False

    for c in state["clients"]:
        try:
            c.close()
        except Exception:
            pass

    root.destroy()


root = tk.Tk()
root.title("Alpha Suppression EEG Simulator")
root.geometry("850x600")

label = tk.Label(root, text="Selected: NEUTRAL", font=("Arial", 20))
label.pack(pady=(12, 6))

prediction_label = tk.Label(
    root,
    text="Predicted: NEUTRAL    Score: +0.00    Alpha power L/R: 0.0 / 0.0",
    font=("Arial", 13),
)
prediction_label.pack(pady=(0, 8))

canvas = tk.Canvas(root, height=330, highlightthickness=0)
canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

frame = tk.Frame(root)
frame.pack(fill="x", padx=10, pady=(0, 10))

buttons = [
    ("Attend LEFT", "left", 0, 0),
    ("NEUTRAL", "neutral", 0, 1),
    ("Attend RIGHT", "right", 0, 2),
]

for text, mode, r, c in buttons:
    b = tk.Button(
        frame,
        text=text,
        font=("Arial", 16),
        command=lambda m=mode: set_mode(m),
    )
    b.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)

frame.rowconfigure(0, weight=1)
for i in range(3):
    frame.columnconfigure(i, weight=1)

root.protocol("WM_DELETE_WINDOW", on_close)

threading.Thread(target=server_thread, daemon=True).start()
threading.Thread(target=stream_thread, daemon=True).start()
root.after(33, draw_waveform)

root.mainloop()
