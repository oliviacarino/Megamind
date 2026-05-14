import json
import math
import socket
import threading
import time
from collections import deque

import tkinter as tk

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


HOST = "127.0.0.1"
PORT = 9999
FS = 250

MIDPOINT_V = 1.65
ALPHA_FREQ = 10.0

WINDOW_SIZE = FS * 4

state = {
    "running": True,
    "clients": [],
    "left_uv": 0.0,
    "right_uv": 0.0,
    "left_alpha_power": 0.0,
    "right_alpha_power": 0.0,
    "predicted_direction": "neutral",
    "attention_score": 0.0,
    "connected_clients": 0,
}

left_buffer = deque(maxlen=WINDOW_SIZE)
right_buffer = deque(maxlen=WINDOW_SIZE)


def classify(left_alpha, right_alpha):
    total = left_alpha + right_alpha + 1e-9
    score = (left_alpha - right_alpha) / total

    if score > 0.25:
        return "left", score
    elif score < -0.25:
        return "right", score
    else:
        return "neutral", score


def server_thread():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen()

    print(f"Broadcasting EEG stream on tcp://{HOST}:{PORT}")

    while state["running"]:
        conn, _ = srv.accept()
        conn.setblocking(False)
        state["clients"].append(conn)
        state["connected_clients"] = len(state["clients"])


def broadcast(packet):
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

    state["connected_clients"] = len(state["clients"])


def adc_thread():
    i2c = busio.I2C(board.SCL, board.SDA)

    ads = ADS.ADS1115(i2c)
    ads.gain = 1
    ads.data_rate = 250

    left_adc = AnalogIn(ads, ADS.P0)
    right_adc = AnalogIn(ads, ADS.P1)

    sample_index = 0
    next_time = time.perf_counter()

    left_alpha_energy = 0.0
    right_alpha_energy = 0.0

    while state["running"]:
        now = time.perf_counter()

        if now >= next_time:
            t = sample_index / FS

            left_v = left_adc.voltage
            right_v = right_adc.voltage

            # Convert voltage centered around 1.65V to approximate microvolts-ish.
            # This is not calibrated gain compensation. It is mainly for visualization.
            left_uv = (left_v - MIDPOINT_V) * 1_000_000
            right_uv = (right_v - MIDPOINT_V) * 1_000_000

            left_buffer.append(left_uv)
            right_buffer.append(right_uv)

            # Simple alpha estimate using 10 Hz reference.
            ref = math.sin(2 * math.pi * ALPHA_FREQ * t)
            left_alpha_energy = 0.995 * left_alpha_energy + 0.005 * abs(left_uv * ref)
            right_alpha_energy = 0.995 * right_alpha_energy + 0.005 * abs(right_uv * ref)

            predicted, score = classify(left_alpha_energy, right_alpha_energy)

            state["left_uv"] = left_uv
            state["right_uv"] = right_uv
            state["left_alpha_power"] = left_alpha_energy
            state["right_alpha_power"] = right_alpha_energy
            state["predicted_direction"] = predicted
            state["attention_score"] = score

            packet = {
                "timestamp": time.time(),
                "sample": sample_index,
                "fs": FS,
                "source": "ads1115",
                "selected_mode": "real",
                "predicted_direction": predicted,
                "attention_score": score,
                "left_uv": left_uv,
                "right_uv": right_uv,
                "left_alpha_power": left_alpha_energy,
                "right_alpha_power": right_alpha_energy,
            }

            broadcast(packet)

            sample_index += 1
            next_time += 1 / FS
        else:
            time.sleep(0.001)


root = tk.Tk()
root.title("Real ADS1115 EEG Source")
root.geometry("900x500")
root.configure(bg="black")

title = tk.Label(
    root,
    text="ADS1115 EEG Source",
    font=("Arial", 22),
    fg="white",
    bg="black",
)
title.pack(pady=8)

status = tk.Label(
    root,
    text="Starting...",
    font=("Arial", 14),
    fg="white",
    bg="black",
)
status.pack()

canvas = tk.Canvas(root, width=850, height=320, bg="#111111", highlightthickness=0)
canvas.pack(pady=15)

info = tk.Label(
    root,
    text="",
    font=("Arial", 14),
    fg="white",
    bg="black",
)
info.pack()


def draw_waveform():
    canvas.delete("all")

    w = 850
    h = 320
    mid_left = h * 0.33
    mid_right = h * 0.66

    canvas.create_text(60, 20, text="A0 Left", fill="white", font=("Arial", 12))
    canvas.create_text(65, h * 0.52, text="A1 Right", fill="white", font=("Arial", 12))

    canvas.create_line(0, mid_left, w, mid_left, fill="#333333")
    canvas.create_line(0, mid_right, w, mid_right, fill="#333333")

    def draw_buffer(buf, mid, tag):
        if len(buf) < 2:
            return

        data = list(buf)[-500:]
        max_abs = max(max(abs(x) for x in data), 1)

        points = []
        for i, val in enumerate(data):
            x = i * w / max(len(data) - 1, 1)
            y = mid - (val / max_abs) * 80
            points.extend([x, y])

        canvas.create_line(points, fill=tag, width=2)

    draw_buffer(left_buffer, mid_left, "#00ff99")
    draw_buffer(right_buffer, mid_right, "#66aaff")

    direction = state["predicted_direction"]
    score = state["attention_score"]

    status.configure(
        text=f"Broadcasting on {HOST}:{PORT} | Clients: {state['connected_clients']}"
    )

    info.configure(
        text=(
            f"Direction: {direction.upper()} | "
            f"Score: {score:.2f} | "
            f"L alpha: {state['left_alpha_power']:.1f} | "
            f"R alpha: {state['right_alpha_power']:.1f}"
        )
    )

    root.after(33, draw_waveform)


def on_close():
    state["running"] = False
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

threading.Thread(target=server_thread, daemon=True).start()
threading.Thread(target=adc_thread, daemon=True).start()

draw_waveform()
root.mainloop()
