import json
import math
import os
import queue
import random
import socket
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk


HOST = "127.0.0.1"
PORT = 9999
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
IMAGE_SIZE = (420, 420)
CONFIDENCE_THRESHOLD = 0.30
HOLD_TIME_SECONDS = 0.45
ROUND_COUNTDOWN_SECONDS = 3

packet_queue = queue.Queue()

state = {
    "direction": "neutral",
    "score": 0.0,
    "connected": False,
    "current_side": None,
    "current_image": None,
    "locked": False,
    "running": True,
    "hold_started_at": None,
    "round_ready_at": None,
}


def find_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ]


def load_resized_image(path, max_size=IMAGE_SIZE):
    img = Image.open(path).convert("RGBA")
    img.thumbnail(max_size)
    return ImageTk.PhotoImage(img)


def load_images(paths):
    loaded = []

    for path in paths:
        try:
            loaded.append((path, load_resized_image(path)))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not loaded:
        raise RuntimeError("No loadable images found in this folder.")

    return loaded


def eeg_receiver():
    while state["running"]:
        sock = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((HOST, PORT))
            sock.settimeout(1)
            packet_queue.put({"connected": True})

            buffer = ""

            while state["running"]:
                data = sock.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    try:
                        packet = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    packet_queue.put(
                        {
                            "connected": True,
                            "direction": packet.get("predicted_direction", "neutral"),
                            "score": float(packet.get("attention_score", 0.0)),
                        }
                    )

        except Exception:
            packet_queue.put({"connected": False, "direction": "neutral", "score": 0.0})
            time.sleep(1)
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


def drain_packets():
    latest = None

    while True:
        try:
            latest = packet_queue.get_nowait()
        except queue.Empty:
            break

    if not latest:
        return

    state["connected"] = latest.get("connected", state["connected"])
    state["direction"] = latest.get("direction", state["direction"])
    state["score"] = latest.get("score", state["score"])


def set_panel_active(panel, active=False, success=False):
    if success:
        panel.configure(bg="#064e3b", highlightbackground="#34d399")
    elif active:
        panel.configure(bg="#172554", highlightbackground="#60a5fa")
    else:
        panel.configure(bg="#18181b", highlightbackground="#3f3f46")


def clear_panel_images():
    left_image_label.configure(image="")
    right_image_label.configure(image="")
    left_image_label.image = None
    right_image_label.image = None


def next_round():
    state["locked"] = False
    state["hold_started_at"] = None
    state["round_ready_at"] = time.perf_counter() + ROUND_COUNTDOWN_SECONDS
    state["current_side"] = random.choice(["left", "right"])
    state["current_image"], tk_img = random.choice(images)

    clear_panel_images()

    if state["current_side"] == "left":
        left_image_label.configure(image=tk_img)
        left_image_label.image = tk_img
    else:
        right_image_label.configure(image=tk_img)
        right_image_label.image = tk_img

    left_title.configure(text="TARGET" if state["current_side"] == "left" else "LEFT")
    right_title.configure(text="TARGET" if state["current_side"] == "right" else "RIGHT")
    progress_bar.configure(width=0)
    set_panel_active(left_panel)
    set_panel_active(right_panel)
    countdown_label.configure(text=str(ROUND_COUNTDOWN_SECONDS))


def mark_success():
    global score_count

    if state["locked"]:
        return

    state["locked"] = True
    score_count += 1
    score_label.configure(text=f"{score_count}")

    if state["current_side"] == "left":
        set_panel_active(left_panel, success=True)
    else:
        set_panel_active(right_panel, success=True)

    progress_bar.configure(width=progress_track.winfo_width())
    root.after(650, next_round)


def update_direction_indicator(direction, confidence):
    is_connected = state["connected"]
    status_dot.configure(bg="#22c55e" if is_connected else "#ef4444")
    connection_label.configure(text="Connected" if is_connected else "Disconnected")
    direction_value.configure(text=direction.upper())
    confidence_value.configure(text=f"{confidence:.2f}")

    if direction == "left":
        direction_value.configure(fg="#38bdf8")
    elif direction == "right":
        direction_value.configure(fg="#fb923c")
    else:
        direction_value.configure(fg="#e4e4e7")


def update_hold_progress(now, direction, confidence):
    ready_at = state["round_ready_at"]

    if ready_at is not None and now < ready_at:
        remaining = max(0, math.ceil(ready_at - now))
        countdown_label.configure(text=str(remaining))
        state["hold_started_at"] = None
        progress_bar.configure(width=0)
        set_panel_active(left_panel)
        set_panel_active(right_panel)
        return

    countdown_label.configure(text="")

    matching = (
        state["connected"]
        and not state["locked"]
        and direction == state["current_side"]
        and confidence >= CONFIDENCE_THRESHOLD
    )

    if not matching:
        state["hold_started_at"] = None
        progress_bar.configure(width=0)
        set_panel_active(left_panel, active=direction == "left")
        set_panel_active(right_panel, active=direction == "right")
        return

    if state["hold_started_at"] is None:
        state["hold_started_at"] = now

    elapsed = now - state["hold_started_at"]
    progress = min(elapsed / HOLD_TIME_SECONDS, 1.0)
    progress_bar.configure(width=int(progress_track.winfo_width() * progress))

    set_panel_active(left_panel, active=state["current_side"] == "left")
    set_panel_active(right_panel, active=state["current_side"] == "right")

    if progress >= 1.0:
        mark_success()


def update_game():
    drain_packets()

    direction = state["direction"]
    confidence = abs(state["score"])
    now = time.perf_counter()

    update_direction_indicator(direction, confidence)
    update_hold_progress(now, direction, confidence)

    root.after(33, update_game)


def on_close():
    state["running"] = False
    root.destroy()


image_paths = find_images()
if not image_paths:
    raise RuntimeError("No images found in this folder.")

root = tk.Tk()
root.title("Alpha Focus Game")
root.geometry("1100x700")
root.minsize(850, 560)
root.configure(bg="#09090b")

images = load_images(image_paths)
score_count = 0

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

header = tk.Frame(root, bg="#09090b")
header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 10))
header.grid_columnconfigure(0, weight=1)

title_label = tk.Label(
    header,
    text="Alpha Focus Game",
    font=("Arial", 28, "bold"),
    fg="#fafafa",
    bg="#09090b",
)
title_label.grid(row=0, column=0, sticky="w")

score_box = tk.Frame(header, bg="#18181b", highlightthickness=1, highlightbackground="#3f3f46")
score_box.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))

score_caption = tk.Label(
    score_box,
    text="SCORE",
    font=("Arial", 10, "bold"),
    fg="#a1a1aa",
    bg="#18181b",
)
score_caption.pack(padx=22, pady=(10, 0))

score_label = tk.Label(
    score_box,
    text="0",
    font=("Arial", 30, "bold"),
    fg="#fafafa",
    bg="#18181b",
)
score_label.pack(padx=22, pady=(0, 10))

sub_label = tk.Label(
    header,
    text="Hold attention on the side with the image until the meter fills.",
    font=("Arial", 14),
    fg="#a1a1aa",
    bg="#09090b",
)
sub_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

status = tk.Frame(root, bg="#111113", highlightthickness=1, highlightbackground="#27272a")
status.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 14))
status.grid_columnconfigure(6, weight=1)

status_dot = tk.Label(status, width=2, bg="#ef4444")
status_dot.grid(row=0, column=0, sticky="ns", padx=(12, 8), pady=12)

connection_label = tk.Label(
    status,
    text="Waiting for simulator",
    font=("Arial", 13, "bold"),
    fg="#fafafa",
    bg="#111113",
)
connection_label.grid(row=0, column=1, sticky="w", pady=12)

tk.Label(status, text="Direction", font=("Arial", 11), fg="#a1a1aa", bg="#111113").grid(
    row=0, column=2, sticky="e", padx=(28, 8)
)
direction_value = tk.Label(
    status,
    text="NEUTRAL",
    font=("Arial", 13, "bold"),
    fg="#e4e4e7",
    bg="#111113",
)
direction_value.grid(row=0, column=3, sticky="w")

tk.Label(status, text="Confidence", font=("Arial", 11), fg="#a1a1aa", bg="#111113").grid(
    row=0, column=4, sticky="e", padx=(28, 8)
)
confidence_value = tk.Label(
    status,
    text="0.00",
    font=("Arial", 13, "bold"),
    fg="#fafafa",
    bg="#111113",
)
confidence_value.grid(row=0, column=5, sticky="w")

arena = tk.Frame(root, bg="#09090b")
arena.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 16))
arena.grid_columnconfigure(0, weight=1, uniform="panels")
arena.grid_columnconfigure(1, weight=1, uniform="panels")
arena.grid_rowconfigure(0, weight=1)

left_panel = tk.Frame(arena, bg="#18181b", highlightthickness=4, highlightbackground="#3f3f46")
right_panel = tk.Frame(arena, bg="#18181b", highlightthickness=4, highlightbackground="#3f3f46")
left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

for panel in (left_panel, right_panel):
    panel.grid_rowconfigure(1, weight=1)
    panel.grid_columnconfigure(0, weight=1)

left_title = tk.Label(
    left_panel,
    text="LEFT",
    font=("Arial", 15, "bold"),
    fg="#e4e4e7",
    bg="#18181b",
)
right_title = tk.Label(
    right_panel,
    text="RIGHT",
    font=("Arial", 15, "bold"),
    fg="#e4e4e7",
    bg="#18181b",
)
left_title.grid(row=0, column=0, sticky="ew", pady=(14, 8))
right_title.grid(row=0, column=0, sticky="ew", pady=(14, 8))

left_image_label = tk.Label(left_panel, bg="#18181b")
right_image_label = tk.Label(right_panel, bg="#18181b")
left_image_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
right_image_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))

countdown_label = tk.Label(
    arena,
    text="",
    font=("Arial", 72, "bold"),
    fg="#fafafa",
    bg="#09090b",
)
countdown_label.place(relx=0.5, rely=0.5, anchor="center")

footer = tk.Frame(root, bg="#09090b")
footer.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 20))
footer.grid_columnconfigure(0, weight=1)

progress_track = tk.Frame(footer, height=16, bg="#27272a")
progress_track.grid(row=0, column=0, sticky="ew")
progress_track.grid_propagate(False)

progress_bar = tk.Frame(progress_track, width=0, height=16, bg="#22c55e")
progress_bar.place(x=0, y=0)

hint_label = tk.Label(
    footer,
    text=f"Required confidence: {CONFIDENCE_THRESHOLD:.2f}   Hold: {HOLD_TIME_SECONDS:.2f}s",
    font=("Arial", 11),
    fg="#71717a",
    bg="#09090b",
)
hint_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

root.protocol("WM_DELETE_WINDOW", on_close)

threading.Thread(target=eeg_receiver, daemon=True).start()
next_round()
update_game()

root.mainloop()
