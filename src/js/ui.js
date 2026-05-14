// ─────────────────────────────────────────────────────────────
// Megamind UI — ui.js
//
// SIM mode  — synthetic EEG, hand open/close, start/stop
// LIVE mode — connects to AlphaWaveInputDriver.py via TCP
//             (host 127.0.0.1:9999), runs the Alpha Focus Game
//             (logic ported from AlphaWave_Focus_Activity.py)
//
// API key is read from .env in the same directory at startup.
// .env format:
//   GEMINI_API_KEY=AIza...
// ─────────────────────────────────────────────────────────────

// ═════════════════════════════════════════════════════════════
// CONFIG — loaded from .env at startup (see boot section)
// ═════════════════════════════════════════════════════════════
let GEMINI_API_KEY = "";

// Live EEG source (AlphaWaveInputDriver.py default)
const TCP_HOST = "127.0.0.1";
const TCP_PORT = 9999;

// Focus game constants (matches AlphaWave_Focus_Activity.py)
const CONFIDENCE_THRESHOLD = 0.30;
const HOLD_TIME_SECONDS = 0.45;
const ROUND_COUNTDOWN_SECS = 3;


// ═════════════════════════════════════════════════════════════
// EEG / signal constants
// ═════════════════════════════════════════════════════════════
const SR = 250;
const ALPHA_MAX = 0.25;
const WAVE_N = 750;    // visible window (3s)
const WAVE_MAX = 7500;   // full history kept (30s) for scrubbing
const RELAXED = "relaxed";
const IMAGINING = "motor_imagery";


// ═════════════════════════════════════════════════════════════
// App state
// ═════════════════════════════════════════════════════════════
let MODE = "sim";
let simRunning = false;
let simPaused = false;
let simInterval = null;

// Sim signal state
let latestFrame = { voltage: 1.65, alpha_power: 0, erd: false, state: RELAXED };
const wavePts = [];
const erdPts = [];
let handProg = 1;  // 1 = open (relaxed), 0 = closed (motor imagery)
let lastState = "";
let sessionSec = 0;
let renderLoop = null;
let npyPlayer = null;   // DemoPlayer when .npy file is loaded; null = use EEGSim

// Pause-scroll state
let scrollOffset = 0;     // samples from right end (read from scrollbar position)

// Live / TCP state
let tcpSocket = null;  // placeholder — browser can't open raw TCP
let liveConnected = false;
let liveFrame = { direction: "neutral", score: 0, left_alpha: 0, right_alpha: 0 };

// Focus game state (from AlphaWave_Focus_Activity.py)
let gameScore = 0;
let currentSide = null;   // "left" | "right"
let gameLocked = false;
let holdStartedAt = null;
let roundReadyAt = null;

// LLM throttle — single global wall-clock timestamp
// Minimum 45s between ANY call to avoid Gemini free tier 429s
const LLM_MIN_INTERVAL = 30;  // seconds between ERD/relax observations
let lastLLMCall = -LLM_MIN_INTERVAL;
let ptBusy = false;

const eegCanvas = document.getElementById("eegCanvas");
const eegCtx = eegCanvas.getContext("2d");


// ═════════════════════════════════════════════════════════════
// JS EEG Simulator (alpha-only, mirrors eeg_simulator.py)
// ═════════════════════════════════════════════════════════════
class EEGSim {
  constructor() {
    this.state = RELAXED; this.t = 0; this.stateTimer = 0;
    this.stateDur = this._rand(); this.driftPh = Math.random() * 6.28; this.buf = [];
  }
  next() {
    const amp = this.state === RELAXED ? 0.35 : 0.06;
    const freq = 10 + 0.3 * Math.sin(6.28 * 0.05 * this.t);
    const v = Math.max(0, Math.min(3.3,
      1.65
      + amp * Math.sin(6.28 * freq * this.t)
      + (Math.random() - 0.5) * 0.04
      + 0.02 * Math.sin(6.28 * 60 * this.t)
      + 0.05 * Math.sin(6.28 * 0.03 * this.t + this.driftPh)
      + this._blink()
    ));
    this.buf.push(v); if (this.buf.length > SR * 5) this.buf.shift();
    this.t += 1 / SR; this.stateTimer += 1 / SR;
    if (this.stateTimer >= this.stateDur) {
      this.state = this.state === RELAXED ? IMAGINING : RELAXED;
      this.stateTimer = 0; this.stateDur = this._rand();
    }
    return v;
  }
  alphaPower() {
    if (this.buf.length < 100) return 0;
    const m = this.buf.reduce((a, b) => a + b, 0) / this.buf.length;
    const rms = Math.sqrt(this.buf.reduce((a, b) => a + (b - m) * (b - m), 0) / this.buf.length);
    return this.state === RELAXED ? rms * 1.1 : rms * 0.22;
  }
  _blink() { const p = (this.t % 6.2) / 6.2; return p < 0.05 ? 0.15 * Math.exp(-Math.pow(p - 0.025, 2) / 0.0002) : 0; }
  _rand() { return 5 + Math.random() * 5; }
}
const sim = new EEGSim();


// ═════════════════════════════════════════════════════════════
// .npy parser + DemoPlayer (streams real EDF-adapted data)
// ═════════════════════════════════════════════════════════════
function parseNpy(buffer) {
  const u8 = new Uint8Array(buffer);
  if (u8[0] !== 0x93 || String.fromCharCode(u8[1], u8[2], u8[3], u8[4], u8[5]) !== 'NUMPY')
    throw new Error("Not a valid .npy file");
  const hlen = u8[8] | (u8[9] << 8);
  const header = String.fromCharCode(...u8.slice(10, 10 + hlen));
  const dtype = (header.match(/'descr':\s*'([^']+)'/) || [, '<f4'])[1];
  const data = buffer.slice(10 + hlen);
  if (dtype.includes('f4') || dtype === 'float32') return new Float32Array(data);
  if (dtype.includes('f8') || dtype === 'float64') return new Float32Array(new Float64Array(data));
  throw new Error(`Unsupported dtype: ${dtype}`);
}

class DemoPlayer {
  constructor(samples) {
    this.samples = samples;
    this.idx = 0;
    this.buf = [];        // rolling 4s buffer
    this.baseline = null;
    this.sampleCount = 0;         // total samples seen (not capped)
    this.alphaPow = 0;
    this.lastCalc = 0;
    this.erd = false;
  }

  next() {
    const v = this.samples[this.idx];
    this.idx = (this.idx + 1) % this.samples.length;
    if (this.idx === 0) { this.baseline = null; this.sampleCount = 0; }

    this.buf.push(v);
    if (this.buf.length > SR * 4) this.buf.shift();   // 4s rolling window
    this.sampleCount++;

    // Recompute alpha every 250ms
    if (this.buf.length >= SR && (this.sampleCount - this.lastCalc) >= SR / 4) {
      this.lastCalc = this.sampleCount;
      const win = this.buf.slice(-SR);   // last 1s
      this.alphaPow = this._alphaBandRMS(win);

      // Calibrate baseline after 4s (buf is full) — use mean of current buffer powers
      if (!this.baseline && this.sampleCount >= SR * 4) {
        this.baseline = this.alphaPow;
        console.log(`[megamind] baseline=${this.baseline.toFixed(6)}V  thr=${(this.baseline * 0.70).toFixed(6)}V`);
      }

      const thr = this.baseline ? this.baseline * 0.70 : null;
      this.erd = thr !== null && this.alphaPow < thr;
      console.log(`[megamind] ap=${this.alphaPow.toFixed(6)}  thr=${thr?.toFixed(6) ?? 'calibrating'}  erd=${this.erd}`);
    }

    return {
      voltage: v,
      alpha_power: this.alphaPow,
      erd: this.erd,
      state: this.erd ? IMAGINING : RELAXED
    };
  }

  _alphaBandRMS(win) {
    // Goertzel on 1-second window — matches Python eeg_analysis.py
    const n = win.length;
    const mean = win.reduce((a, b) => a + b, 0) / n;

    let power = 0, count = 0;
    // Bins for 8–12 Hz with 1s window at 250 SPS → bins 8 to 12
    for (let k = 8; k <= 12; k++) {
      const w = 2 * Math.PI * k / n;
      const coeff = 2 * Math.cos(w);
      let s1 = 0, s2 = 0;
      for (let i = 0; i < n; i++) {
        const s0 = (win[i] - mean) + coeff * s1 - s2;
        s2 = s1; s1 = s0;
      }
      const re = s1 - s2 * Math.cos(w);
      const im = s2 * Math.sin(w);
      power += (re * re + im * im) / (n * n);
      count++;
    }
    return Math.sqrt(power / count);
  }

  get duration() { return this.samples.length / SR; }
  get currentTime() { return this.idx / SR; }
}

function loadNpy(event) {
  const file = event.target.files[0];
  if (!file) return;
  const status = document.getElementById("simStatus");
  status.textContent = `Loading ${file.name}…`;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const samples = parseNpy(e.target.result);
      npyPlayer = new DemoPlayer(samples);
      const dur = Math.floor(npyPlayer.duration);
      const mm = Math.floor(dur / 60), ss = String(dur % 60).padStart(2, '0');
      document.getElementById("npyLabel").textContent = `✓ ${file.name}`;
      status.textContent = `${file.name} loaded (${mm}:${ss}) — press Start`;
      status.className = "sim-status running";
      console.log(`[megamind] Loaded ${file.name}: ${samples.length} samples, ${dur}s`);
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
      status.className = "sim-status";
      npyPlayer = null;
    }
  };
  reader.readAsArrayBuffer(file);
}
function setMode(mode) {
  const prev = MODE;
  MODE = mode;
  if (prev !== mode) clearFeed();
  document.querySelectorAll(".mode-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === mode));

  document.getElementById("simSection").style.display = mode === "sim" ? "flex" : "none";
  document.getElementById("liveSection").style.display = mode === "live" ? "flex" : "none";

  document.getElementById("ptModelLabel").textContent = "Model: Gemini 2.5 Flash-Lite (free)";
}


// ═════════════════════════════════════════════════════════════
// SIM — start / stop
// ═════════════════════════════════════════════════════════════
function startSim() {
  if (simRunning) return;
  simRunning = true;
  sessionSec = 0;
  lastState = "";
  wavePts.length = 0; erdPts.length = 0;
  lastLLMCall = -LLM_MIN_INTERVAL;
  erdEventCount = 0;
  relaxEventCount = 0;
  lastObsTime = 0;
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;
  document.getElementById("pauseBtn").disabled = false;

  const usingFile = !!npyPlayer;
  document.getElementById("simStatus").textContent = usingFile
    ? "Streaming EDF data…"
    : "Session running (synthetic data)…";
  document.getElementById("simStatus").className = "sim-status running";


  simInterval = setInterval(() => {
    if (!simRunning || simPaused) return;
    const batch = Math.round(SR / 50);
    if (usingFile) {
      // Stream from loaded .npy file
      for (let i = 0; i < batch; i++) {
        const f = npyPlayer.next();
        latestFrame = f;
        pushWave(f.voltage, f.erd);
      }
      sessionSec += batch / SR;
    } else {
      // Fallback: synthetic simulator
      for (let i = 0; i < batch; i++) pushWave(sim.next(), null);
      const ap = sim.alphaPower();
      const erd = ap < 0.04;
      latestFrame = {
        voltage: sim.buf.at(-1) || 1.65, alpha_power: ap, erd,
        state: erd ? IMAGINING : RELAXED
      };
      for (let i = 0; i < batch && i < erdPts.length; i++) erdPts[erdPts.length - 1 - i] = erd;
      sessionSec += batch / SR;
    }
  }, 1000 / 50);

  renderLoop = setInterval(() => {
    if (!simRunning || simPaused) return;
    drawEEG(latestFrame);
    updateMeter(latestFrame);
    updateStateLabel(latestFrame);
    updateHand(latestFrame);
    if (latestFrame.state !== lastState) {
      onStateChange(latestFrame.state);
      lastState = latestFrame.state;
    }
  }, 1000 / 50);

  // Single opening observation — sets lastLLMCall so onStateChange is blocked
  // until the full LLM_MIN_INTERVAL has elapsed after this call
  addObs("Session started — monitoring alpha band (8–12 Hz).", "sys");
  ptBusy = true;
  // Use callGemini directly so we don't stamp lastLLMCall — first ERD can fire right away
  (async () => {
    const t = addTyping();
    const reply = await callGemini(
      "Motor imagery session starting. Alpha drops = hand opens, alpha high = hand closed. " +
      "One sentence telling the doctor what to watch for. No greeting."
    );
    t.remove();
    if (reply) addObs(reply, "sys");
    ptBusy = false;
  })();
}

function pauseSim() {
  simPaused = !simPaused;
  eegCanvas.style.cursor = "default";
  const btn = document.getElementById("pauseBtn");
  btn.textContent = simPaused ? "Resume" : "Pause";
  btn.classList.toggle("active", simPaused);
  document.getElementById("simStatus").textContent = simPaused ? "Session paused" : "Session running…";
  document.getElementById("simStatus").className = "sim-status" + (simPaused ? "" : " running");
}

function stopSim() {
  simRunning = false;
  simPaused = false;
  clearInterval(simInterval); simInterval = null;
  clearInterval(renderLoop); renderLoop = null;
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
  document.getElementById("pauseBtn").disabled = true;
  document.getElementById("pauseBtn").textContent = "Pause";
  document.getElementById("pauseBtn").classList.remove("active");
  document.getElementById("simStatus").textContent = "Session stopped";
  document.getElementById("simStatus").className = "sim-status";
  document.getElementById("state-label").textContent = "Stopped";
  document.getElementById("state-label").classList.remove("active");
  clearFeed();
}

function pushWave(v, erd) {
  wavePts.push(v); erdPts.push(erd ?? latestFrame.erd);
  if (wavePts.length > WAVE_MAX) { wavePts.shift(); erdPts.shift(); }
}


// ═════════════════════════════════════════════════════════════
// EEG canvas
// ═════════════════════════════════════════════════════════════
function drawEEG(f) {
  const wrap = eegCanvas.parentElement;
  const W = eegCanvas.width = wrap.clientWidth * devicePixelRatio;
  const H = eegCanvas.height = wrap.clientHeight * devicePixelRatio;
  eegCtx.clearRect(0, 0, W, H);
  eegCtx.fillStyle = "#f9fafb"; eegCtx.fillRect(0, 0, W, H);

  if (wavePts.length < 2) {
    eegCtx.fillStyle = "#e5e7eb";
    eegCtx.font = `${11 * devicePixelRatio}px Inter,sans-serif`;
    eegCtx.textAlign = "center";
    eegCtx.fillText("Press Start to begin", W / 2, H / 2);
    return;
  }

  const n = Math.min(wavePts.length, WAVE_N);
  const startIdx = wavePts.length - n;
  const visWave = wavePts.slice(startIdx);
  const visErd = erdPts.slice(startIdx);

  // Grid
  eegCtx.strokeStyle = "#e5e7eb"; eegCtx.lineWidth = 0.5 * devicePixelRatio;
  for (const r of [0, 0.5, 1]) {
    const y = H - r * H * 0.88 - H * 0.06;
    eegCtx.beginPath(); eegCtx.moveTo(0, y); eegCtx.lineTo(W, y); eegCtx.stroke();
  }

  // Trace
  for (let i = 1; i < n; i++) {
    const x0 = ((i - 1) / (n - 1)) * W, x1 = (i / (n - 1)) * W;
    const y0 = voltY(visWave[i - 1], H), y1 = voltY(visWave[i], H);
    eegCtx.beginPath(); eegCtx.moveTo(x0, y0); eegCtx.lineTo(x1, y1);
    eegCtx.strokeStyle = visErd[i] ? "#00A8E0" : "#9eb8cc";
    eegCtx.lineWidth = 1.5 * devicePixelRatio; eegCtx.stroke();
  }

  // Centre dashed line
  const cy = voltY(1.65, H);
  eegCtx.strokeStyle = "#e5e7eb"; eegCtx.lineWidth = 0.8 * devicePixelRatio; eegCtx.setLineDash([4, 4]);
  eegCtx.beginPath(); eegCtx.moveTo(0, cy); eegCtx.lineTo(W, cy); eegCtx.stroke();
  eegCtx.setLineDash([]);
  eegCtx.fillStyle = "#9ca3af";
  eegCtx.font = `${9 * devicePixelRatio}px 'Share Tech Mono',monospace`;
  eegCtx.textAlign = "left";
  eegCtx.fillText("1.65V", 4 * devicePixelRatio, cy - 3 * devicePixelRatio);

  document.getElementById("annRelaxed").style.display = (f?.erd) ? "none" : "inline";
  document.getElementById("annErd").style.display = (f?.erd) ? "inline" : "none";
}

function voltY(v, H) { return H - ((v / 3.3) * H * 0.88 + H * 0.06); }

function updateMeter(f) {
  const pct = Math.min(100, (f.alpha_power / ALPHA_MAX) * 100).toFixed(0);
  const bar = document.getElementById("alphaBar");
  bar.style.width = pct + "%"; bar.style.background = f.erd ? "#0077A8" : "#00A8E0";
  document.getElementById("alphaVal").textContent = f.alpha_power.toFixed(4) + "V";
}

function updateStateLabel(f) {
  if (!simRunning) return;
  const labels = { relaxed: "Relaxed — α high", motor_imagery: "Motor imagery — ERD detected" };
  const el = document.getElementById("state-label");
  el.textContent = labels[f.state] || f.state;
  el.classList.toggle("active", f.erd);
}


// ═════════════════════════════════════════════════════════════
// Hand animation
// ═════════════════════════════════════════════════════════════
// FY_O / FH_O = OPEN finger positions, FY_C / FH_C = CLOSED
// Relaxed (high alpha) = hand OPEN, ERD (motor imagery) = hand CLOSING
const FY_O = [60, 50, 50, 55, 90], FY_C = [72, 72, 72, 72, 90];
const FH_O = [28, 35, 35, 30, 12], FH_C = [15, 15, 15, 15, 12];

function updateHand(f) {
  // Invert: relaxed (no ERD) → open (prog=1), ERD → closing (prog=0)
  const target = f.erd ? 0 : 1;
  handProg += (target - handProg) * 0.05;
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById("f" + i);
    el.setAttribute("y", FY_C[i] + (FY_O[i] - FY_C[i]) * handProg);
    el.setAttribute("height", FH_C[i] + (FH_O[i] - FH_C[i]) * handProg);
  }
  document.getElementById("ring").classList.toggle("active", !f.erd);
}


// ═════════════════════════════════════════════════════════════
// LIVE — TCP connection to AlphaWaveInputDriver.py
//
// Browsers cannot open raw TCP sockets. Two options:
//   1. Run a tiny WebSocket bridge alongside the driver (ws_bridge.py)
//   2. Use the included ws_server.py with --source raw_data
//
// This UI attempts a WebSocket connection to ws://127.0.0.1:8765
// which ws_bridge.py or ws_server.py provides.
// The UI shows the correct "not connected" state if nothing is running.
// ═════════════════════════════════════════════════════════════
let ws = null;
let liveRenderLoop = null;

function toggleConnect() {
  if (liveConnected) {
    disconnectLive();
  } else {
    connectLive();
  }
}

function showLiveConnected(connected) {
  document.getElementById("liveHwOverlay").style.display = connected ? "none" : "flex";
  document.getElementById("liveExerciseInfo").style.display = connected ? "none" : "flex";
  document.getElementById("focusGame").style.display = connected ? "grid" : "none";
}

function connectLive() {
  setLiveStatus("connecting", "Connecting to ws://127.0.0.1:8765…");
  document.getElementById("connectBtn").textContent = "Cancel";

  ws = new WebSocket("ws://127.0.0.1:8765");

  ws.onopen = () => {
    liveConnected = true;
    setLiveStatus("connected", "Connected — receiving EEG from ADS1115");
    document.getElementById("connectBtn").textContent = "Disconnect";
    document.getElementById("connectBtn").classList.add("connected");
    showLiveConnected(true);
    startFocusGame();
    addObs("Hardware connection established. ADS1115 signal streaming from Raspberry Pi.", "sys");
    callLLM(
      "The patient is now connected to the live EEG hardware. The Alpha Focus Game is starting. " +
      "Give a brief opening observation about what a successful session looks like (2 sentences).", "sys"
    );
  };

  ws.onmessage = e => {
    try {
      const pkt = JSON.parse(e.data);
      liveFrame = {
        direction: pkt.predicted_direction || "neutral",
        score: Math.abs(pkt.attention_score || 0),
        left_alpha: pkt.left_alpha_power || 0,
        right_alpha: pkt.right_alpha_power || 0,
      };
      // Draw live wave on liveEegCanvas
      drawLiveWave(pkt.left_uv || 0, pkt.right_uv || 0);
    } catch (e) { }
  };

  ws.onclose = () => {
    if (liveConnected) {
      liveConnected = false;
      setLiveStatus("", "Connection lost — run AlphaWaveInputDriver.py and try again");
      document.getElementById("connectBtn").textContent = "Reconnect";
      document.getElementById("connectBtn").classList.remove("connected");
      showLiveConnected(false);
      stopFocusGame();
    }
  };

  ws.onerror = () => {
    setLiveStatus("", "Cannot connect. Start ws_bridge.py or ws_server.py on the Pi first.");
    document.getElementById("connectBtn").textContent = "Connect";
    document.getElementById("connectBtn").classList.remove("connected");
  };
}

function disconnectLive() {
  if (ws) { ws.close(); ws = null; }
  liveConnected = false;
  setLiveStatus("", "Disconnected");
  document.getElementById("connectBtn").textContent = "Connect";
  document.getElementById("connectBtn").classList.remove("connected");
  showLiveConnected(false);
  stopFocusGame();
}

function setLiveStatus(state, msg) {
  const dot = document.getElementById("liveDot");
  const text = document.getElementById("liveStatusText");
  dot.className = "live-dot" + (state ? " " + state : "");
  text.textContent = msg;
  text.className = "live-status-text" + (state === "connected" ? " connected" : "");
}

// Live EEG wave (two channels from ADS1115)
const liveWavePts = { left: [], right: [] };
const LIVE_WAVE_N = 500;

function drawLiveWave(leftUv, rightUv) {
  liveWavePts.left.push(leftUv); if (liveWavePts.left.length > LIVE_WAVE_N) liveWavePts.left.shift();
  liveWavePts.right.push(rightUv); if (liveWavePts.right.length > LIVE_WAVE_N) liveWavePts.right.shift();

  const canvas = document.getElementById("liveEegCanvas");
  if (!canvas) return;
  const W = canvas.width = canvas.parentElement.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.parentElement.clientHeight * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#f9fafb"; ctx.fillRect(0, 0, W, H);

  const drawChannel = (pts, color, mid) => {
    if (pts.length < 2) return;
    const n = pts.length;
    const maxAbs = Math.max(...pts.map(Math.abs), 1);
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5 * devicePixelRatio;
    pts.forEach((v, i) => {
      const x = (i / (n - 1)) * W;
      const y = mid - (v / maxAbs) * (H * 0.22);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  // Grid lines
  ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 0.5 * devicePixelRatio;
  [H * 0.25, H * 0.5, H * 0.75].forEach(y => {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  });

  drawChannel(liveWavePts.left, "#00A8E0", H * 0.27);  // top half — left channel
  drawChannel(liveWavePts.right, "#0077A8", H * 0.73);  // bottom half — right channel

  // Labels
  ctx.fillStyle = "#9ca3af"; ctx.font = `${9 * devicePixelRatio}px 'Share Tech Mono',monospace`;
  ctx.textAlign = "left";
  ctx.fillText("L (A0)", 4 * devicePixelRatio, H * 0.14);
  ctx.fillText("R (A1)", 4 * devicePixelRatio, H * 0.60);

  // Update alpha bars
  const maxA = Math.max(liveFrame.left_alpha, liveFrame.right_alpha, 1);
  document.getElementById("liveLeftBar").style.width = Math.min(100, (liveFrame.left_alpha / maxA) * 100) + "%";
  document.getElementById("liveRightBar").style.width = Math.min(100, (liveFrame.right_alpha / maxA) * 100) + "%";
}


// ═════════════════════════════════════════════════════════════
// Alpha Focus Game (ported from AlphaWave_Focus_Activity.py)
//
// - Randomly picks left or right as TARGET each round
// - Shows directional arrow on the target side
// - Patient focuses alpha attention toward that side
// - When direction matches + confidence ≥ threshold for 0.45s → score
// - 3-second countdown between rounds
// ═════════════════════════════════════════════════════════════
const EMOJIS = ["🎯", "🌟", "🏆", "⚡", "🎪", "🦋", "🌊", "🔮"];

function startFocusGame() {
  gameScore = 0;
  updateScoreBadge();
  nextRound();
  liveRenderLoop = setInterval(updateFocusGame, 33);
}

function stopFocusGame() {
  clearInterval(liveRenderLoop); liveRenderLoop = null;
  holdStartedAt = null; roundReadyAt = null;
  setPanel("leftPanel", "");
  setPanel("rightPanel", "");
  document.getElementById("progressFill").style.width = "0%";
}

function nextRound() {
  gameLocked = false;
  holdStartedAt = null;
  roundReadyAt = performance.now() / 1000 + ROUND_COUNTDOWN_SECS;
  currentSide = Math.random() > 0.5 ? "left" : "right";

  // Reset panels
  setPanel("leftPanel", "");
  setPanel("rightPanel", "");
  document.getElementById("progressFill").style.width = "0%";

  // Place emoji on target side
  const emoji = EMOJIS[Math.floor(Math.random() * EMOJIS.length)];
  document.getElementById("leftImg").innerHTML = `<span style="font-size:48px">${currentSide === "left" ? emoji : ""}</span>`;
  document.getElementById("rightImg").innerHTML = `<span style="font-size:48px">${currentSide === "right" ? emoji : ""}</span>`;

  document.getElementById("leftTitle").textContent = currentSide === "left" ? "TARGET" : "Left";
  document.getElementById("rightTitle").textContent = currentSide === "right" ? "TARGET" : "Right";
  document.getElementById("leftTitle").className = "focus-panel-title" + (currentSide === "left" ? " target-label" : "");
  document.getElementById("rightTitle").className = "focus-panel-title" + (currentSide === "right" ? " target-label" : "");

  if (currentSide === "left") setPanel("leftPanel", "target");
  if (currentSide === "right") setPanel("rightPanel", "target");
}

function updateFocusGame() {
  if (!liveConnected) return;

  const now = performance.now() / 1000;
  updateSignalStrip();

  // Countdown
  if (roundReadyAt !== null && now < roundReadyAt) {
    const remaining = Math.ceil(roundReadyAt - now);
    showCountdown(remaining);
    return;
  }
  showCountdown(null);

  const { direction, score } = liveFrame;
  const matching = !gameLocked
    && direction === currentSide
    && score >= CONFIDENCE_THRESHOLD;

  if (!matching) {
    holdStartedAt = null;
    document.getElementById("progressFill").style.width = "0%";
    setPanel("leftPanel", direction === "left" ? "active" : (currentSide === "left" ? "target" : ""));
    setPanel("rightPanel", direction === "right" ? "active" : (currentSide === "right" ? "target" : ""));
    return;
  }

  if (holdStartedAt === null) holdStartedAt = now;
  const progress = Math.min((now - holdStartedAt) / HOLD_TIME_SECONDS, 1.0);
  document.getElementById("progressFill").style.width = (progress * 100) + "%";

  if (progress >= 1.0) markSuccess();
}

function markSuccess() {
  if (gameLocked) return;
  gameLocked = true;
  gameScore++;
  updateScoreBadge();
  setPanel(currentSide === "left" ? "leftPanel" : "rightPanel", "success");
  document.getElementById("progressFill").style.width = "100%";
  addObs(`Score! Patient successfully focused ${currentSide} hemisphere alpha. Score: ${gameScore}`, "score");
  callLLM(
    `The patient just scored a point by focusing alpha attention to the ${currentSide} side. Score is now ${gameScore}. ` +
    `Give a brief encouraging observation (1–2 sentences).`, "score"
  );
  setTimeout(nextRound, 700);
}

function setPanel(id, state) {
  const el = document.getElementById(id);
  el.className = "focus-panel" + (state ? " " + state : "");
}

function showCountdown(n) {
  const el = document.getElementById("countdownOverlay");
  if (n === null || n <= 0) { el.textContent = ""; el.className = "countdown-overlay"; return; }
  el.textContent = n;
  el.className = "countdown-overlay visible";
}

function updateScoreBadge() {
  document.getElementById("scoreBadge").textContent = `Score: ${gameScore}`;
}

function updateSignalStrip() {
  const { left_alpha, right_alpha, direction, score } = liveFrame;
  const maxA = Math.max(left_alpha, right_alpha, 1);
  document.getElementById("leftBar").style.width = Math.min(100, (left_alpha / maxA) * 100) + "%";
  document.getElementById("rightBar").style.width = Math.min(100, (right_alpha / maxA) * 100) + "%";
  document.getElementById("leftVal").textContent = left_alpha.toFixed(1);
  document.getElementById("rightVal").textContent = right_alpha.toFixed(1);
  const db = document.getElementById("directionBadge");
  db.textContent = direction.charAt(0).toUpperCase() + direction.slice(1);
  db.className = "direction-badge " + direction;
}


// ═════════════════════════════════════════════════════════════
// PT observations — state change callbacks
// ═════════════════════════════════════════════════════════════

// Track activity since last observation for summary
let erdEventCount = 0;
let relaxEventCount = 0;
let lastObsTime = 0;

// Called every state change — just count, don't call LLM
function trackStateChange(state) {
  if (state === IMAGINING) erdEventCount++;
  else relaxEventCount++;
}

function buildSummaryPrompt(currentIsErd, ap) {
  const elapsed = Math.round(sessionSec - lastObsTime);
  const totalEvts = erdEventCount + relaxEventCount;
  // Inverted: ERD = hand closing (focused), relaxed = hand open
  const hand = currentIsErd ? "closing (motor imagery)" : "open (relaxed)";
  const pct = totalEvts > 0 ? Math.round((erdEventCount / totalEvts) * 100) : 0;

  return (
    `Last ${elapsed}s: ${erdEventCount} motor imagery events (hand closing), ` +
    `${relaxEventCount} rest events (hand open). ` +
    `${pct}% of the time in active imagery. ` +
    `Hand is currently ${hand}, alpha at ${ap != null ? ap.toFixed(4) : "—"}V. ` +
    `Give the doctor a 1-2 sentence summary. No greeting. Mention the hand state. Be specific.`
  );
}

function onStateChange(state) {
  trackStateChange(state);
  if (!simRunning || ptBusy) return;
  if ((sessionSec - lastLLMCall) < LLM_MIN_INTERVAL) return;

  const isErd = state === IMAGINING;
  const ap = latestFrame.alpha_power;
  const type = isErd ? "erd" : "relax";

  ptBusy = true;
  callLLM(buildSummaryPrompt(isErd, ap), type).finally(() => {
    // Reset counters after each observation
    erdEventCount = 0;
    relaxEventCount = 0;
    lastObsTime = sessionSec;
    ptBusy = false;
  });
}

// Ambient commentary disabled — preserving 20 RPD free tier quota
// setInterval(() => { ... }, 15000);


// ═════════════════════════════════════════════════════════════
// LLM — Gemini 2.5 Flash-Lite (free tier)
// ═════════════════════════════════════════════════════════════
const PT_SYS = `You are a physical therapist sending quick updates to a doctor watching a motor imagery EEG session.
Alpha brainwaves (8-12 Hz) drive a hand animation — HIGH alpha (relaxed) means the hand is OPEN; LOW alpha (ERD, motor imagery) means the hand is CLOSING.
Rules: no greeting, no sign-off, no filler. Max 1-2 short sentences. Mention the hand state. Vary your phrasing every time.`;

async function callLLM(prompt, type) {
  lastLLMCall = sessionSec;
  const t = addTyping();
  let reply = null;
  try {
    if (GEMINI_API_KEY) {
      reply = await callGemini(prompt);
    } else {
      reply = "[Add GEMINI_API_KEY to your config.js to enable AI commentary]";
    }
  } catch (e) {
    console.error("[megamind] callLLM exception:", e);
    reply = null;
  }
  t.remove();
  if (reply) {
    addObs(reply, type);
  } else {
    console.warn("[megamind] callLLM got null reply — check console for errors above");
  }
  return reply;
}

async function callGemini(prompt, retry = true) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": GEMINI_API_KEY
    },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: PT_SYS }] },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 120, temperature: 0.7 }
    })
  });
  const data = await res.json();
  if (!res.ok) {
    // 503 = server overloaded, retry once after 5s
    if (res.status === 503 && retry) {
      console.warn("[megamind] Gemini 503 — retrying in 5s…");
      await new Promise(r => setTimeout(r, 5000));
      return callGemini(prompt, false);
    }
    console.error("[megamind] Gemini error:", data?.error?.message);
    return null;
  }
  return data.candidates?.[0]?.content?.parts?.[0]?.text || null;
}


// ═════════════════════════════════════════════════════════════
// Feed UI helpers
// ═════════════════════════════════════════════════════════════
function clearFeed() {
  document.getElementById("feedMessages").innerHTML = "";
  sessionSec = 0;
  lastLLMCall = -LLM_MIN_INTERVAL;
  erdEventCount = 0;
  relaxEventCount = 0;
  lastObsTime = 0;
  ptBusy = false;
}

function fmtTime(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; }

function ts() { return fmtTime(sessionSec); }

const obsLabels = {
  erd: { cls: "", text: "ERD — Physician Note" },
  relax: { cls: "relax", text: "Rest — Physician Note" },
  sys: { cls: "sys", text: "Session" },
  score: { cls: "score", text: "Score" },
};

function addObs(text, type = "relax") {
  const { cls, text: label } = obsLabels[type] || obsLabels.relax;
  const feed = document.getElementById("feedMessages");
  const div = document.createElement("div");
  div.className = `obs ${type}-event`;
  div.innerHTML = `
    <div class="obs-time">${ts()}</div>
    <div class="obs-label ${cls}">${label}</div>
    <div class="obs-text">${text}</div>`;
  feed.appendChild(div);
  div.scrollIntoView({ behavior: "smooth" });
}

function addTyping() {
  const feed = document.getElementById("feedMessages");
  const div = document.createElement("div");
  div.className = "typing-row";
  div.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div><span class="typing-label">PT observing…</span>`;
  feed.appendChild(div);
  div.scrollIntoView({ behavior: "smooth" });
  return div;
}


// ═════════════════════════════════════════════════════════════
// Boot — read config from config.js (window.ENV), then init
// ═════════════════════════════════════════════════════════════
function loadEnv() {
  if (window.ENV && window.ENV.GEMINI_API_KEY) {
    GEMINI_API_KEY = window.ENV.GEMINI_API_KEY;
    console.log("[megamind] GEMINI_API_KEY loaded from config.js");
  } else {
    console.warn("[megamind] window.ENV not found — make sure config.js is loaded before ui.js");
  }
}

loadEnv();
const modelLabel = document.getElementById("ptModelLabel");
if (modelLabel) modelLabel.textContent = "Model: Gemini 2.5 Flash-Lite (free)";
setMode("sim");
drawEEG({ erd: false, state: RELAXED, alpha_power: 0, voltage: 1.65 });

// ── ERD hover tooltip ─────────────────────────────────────────
const tooltip = document.getElementById("eegTooltip");

eegCanvas.addEventListener("mousemove", e => {
  if (wavePts.length < 2) return;
  const rect = eegCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const n = Math.min(wavePts.length, WAVE_N);
  const startIdx = wavePts.length - n;
  const idx = startIdx + Math.round((mouseX / rect.width) * (n - 1));
  const clamped = Math.max(0, Math.min(wavePts.length - 1, idx));

  if (erdPts[clamped]) {
    const t = (clamped / SR).toFixed(1);
    tooltip.textContent = `ERD — α suppressed  t=${t}s`;
    tooltip.style.left = e.clientX + "px";
    tooltip.style.top = e.clientY + "px";
    tooltip.style.opacity = "1";
  } else {
    tooltip.style.opacity = "0";
  }
});
eegCanvas.addEventListener("mouseleave", () => { tooltip.style.opacity = "0"; });