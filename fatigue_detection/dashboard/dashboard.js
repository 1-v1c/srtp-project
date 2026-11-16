"use strict";

const $ = (id) => document.getElementById(id);
const fatigueLabels = ["清醒", "轻度疲劳", "中度疲劳", "重度疲劳"];
const sourceLabels = {
  eye_close: "持续闭眼", micro_sleep: "微睡眠", perclos: "闭眼比例",
  yawn: "哈欠", yawn_freq: "哈欠频率", head_down: "持续低头", yaw_dev: "持续偏头",
  blink_rate: "眨眼频率", blink_dur: "眨眼时长",
};
const distractionLabels = {
  c0_safe: "正常驾驶", c1_text_r: "右手发消息", c2_call_r: "右手打电话",
  c3_text_l: "左手发消息", c4_call_l: "左手打电话", c5_radio: "操作中控",
  c6_drink: "喝水", c7_reach: "伸手取物", c8_makeup: "整理仪容", c9_talk: "交谈",
};
const chartWindow = 30;
const points = [];
const activeActions = new Set();
let lastState = null;
let revision = null;
let lastStreamAttempt = 0;
let timelineBreak = false;
let lastElapsed = 0;

function write(id, value) {
  const el = $(id);
  const text = String(value);
  if (el.textContent !== text) el.textContent = text;
}

function number(value, digits, unit = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${unit}` : "—";
}

function percentage(value, digits = 1) {
  return Number.isFinite(value) ? number(value * 100, digits, "%") : "—";
}

function indicator(id, available, label) {
  const element = $(id);
  const value = element.querySelector("strong");
  element.dataset.ok = available === null ? "" : String(available);
  if (value.textContent !== label) value.textContent = label;
}

function setVideo(live, message) {
  const video = $("video");
  const placeholder = $("video-placeholder");
  if (live && !video.hasAttribute("src") && performance.now() - lastStreamAttempt > 900) {
    lastStreamAttempt = performance.now();
    video.src = `/api/stream?start=${Date.now()}`;
  }
  if (!live) {
    video.removeAttribute("src");
    lastStreamAttempt = -Infinity;
  }
  const visible = live && video.hasAttribute("src");
  video.hidden = !visible;
  placeholder.hidden = visible;
  write("video-message", message);
}

$("video").addEventListener("error", () => {
  $("video").removeAttribute("src");
  $("video").hidden = true;
  $("video-placeholder").hidden = false;
  write("video-message", "正在重连画面");
});

function renderDistraction(data) {
  $("distraction").disabled = !data?.available || activeActions.has("distraction");
  if (!activeActions.has("distraction")) $("distraction").checked = Boolean(data?.enabled);
  const result = data?.result;
  let text = "等待监测";
  if (data && !data.available) text = "分心模型未加载";
  else if (data && !data.enabled) text = "已关闭";
  else if (data && !result) text = "等待分类结果";
  else if (result) {
    const label = distractionLabels[result.label] ?? "待识别";
    text = `${label} · ${percentage(result.confidence, 0)}${result.stable ? "" : " · 确认中"}`;
  }
  write("distraction-result", text);
  $("distraction-result").dataset.alert = String(Boolean(result?.stable && result.level >= 3));
}

function renderSources(data) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const labels = entries.map(([source, value]) => `${sourceLabels[source] ?? source} +${value.toFixed(2)}`);
  const list = $("sources");
  const signature = labels.join("|");
  if (list.dataset.signature === signature) return;
  list.dataset.signature = signature;
  const items = (labels.length ? labels : ["暂无活动信号"]).map((label) => {
    const item = document.createElement("li");
    if (!labels.length) item.className = "muted";
    item.textContent = label;
    return item;
  });
  list.replaceChildren(...items);
}

function render(state) {
  lastState = state;
  const live = state.status === "live";
  const paused = state.status === "paused";
  const resuming = state.status === "resuming";
  const states = {
    starting: "启动中", live: "实时监测", paused: "已暂停", resuming: "恢复中",
    stopped: "已停止", error: "设备异常", offline: "连接中断",
  };
  $("connection").dataset.state = state.status;
  write("connection-text", states[state.status] ?? "连接中");
  setVideo(live, state.message ?? "正在连接摄像头");

  const snapshot = $("snapshot");
  snapshot.setAttribute("aria-disabled", String(!live));
  snapshot.tabIndex = live ? 0 : -1;
  if (live) snapshot.href = "/api/frame.jpg";
  else snapshot.removeAttribute("href");
  for (const control of ["mesh", "reset", "calibrate"]) {
    $(control).disabled = !live || activeActions.has(control);
  }
  $("pause").disabled = (!live && !paused) || activeActions.has("pause");
  $("pause").dataset.resume = String(paused);
  write("pause", paused || resuming ? "继续监测" : "暂停监测");
  if (!activeActions.has("mesh")) $("mesh").checked = live && Boolean(state.mesh);

  if (live) lastElapsed = state.elapsed;
  else if (!paused && !resuming) lastElapsed = 0;
  const time = Math.floor(lastElapsed);
  write("session-time", `${String(Math.floor(time / 60)).padStart(2, "0")}:${String(time % 60).padStart(2, "0")}`);
  write("camera-meta", live ? `设备 ${state.camera} · ${state.resolution.join(" × ")}` : "本机摄像头");
  write("fps", number(state.fps, 1, " FPS"));
  indicator("face-status", live ? state.face_valid : null, live ? (state.face_valid ? "已定位" : "未检测") : "—");
  indicator("pose-status", live ? state.pose_valid : null, live ? (state.pose_valid ? "有效" : "不可用") : "—");
  const eyeReason = !state.face_valid ? "未定位人脸" : !state.pose_valid ? "头姿估计不可用" : "侧脸保护";
  indicator("eyes-status", live ? state.eye_valid : null, live ? (state.eye_valid ? "有效" : eyeReason) : "—");

  $("risk").dataset.level = live ? state.level : "idle";
  write("grade", live ? `L${state.level}` : "—");
  write("risk-title", live ? fatigueLabels[state.level] : (paused ? "监测已暂停" : "等待观测"));
  write("risk-note", live ? (state.eye_valid ? "眼部信号有效" : eyeReason) : "当前没有实时数据");
  write("score", number(state.score, 2));
  for (const [index, mark] of [...$("levels").children].entries()) {
    mark.classList.toggle("active", live && state.level === index);
  }

  write("ear", number(state.ear, 3));
  write("ear-detail", `左眼 ${number(state.ear_left, 3)} · 右眼 ${number(state.ear_right, 3)}`);
  write("mar", number(state.mar, 3));
  write("yawns", live ? `哈欠 ${state.yawns} 次` : "哈欠 — 次");
  write("perclos", percentage(state.perclos));
  write("perclos-detail", live ? `${state.perclos_window} 秒窗 · 有效 ${number(state.perclos_valid_sec, 1, " s")}` : "30 秒窗口");
  $("perclos").dataset.alert = String(live && state.perclos > state.perclos_threshold);
  write("closure", number(state.eye_closed_sec, 2, " s"));
  write("closure-detail", live && state.eye_valid ? (state.eye_closed_sec > 0 ? "正在累计闭眼" : "当前睁眼") : "等待有效眼部观测");
  for (const key of ["pitch", "yaw", "roll"]) write(key, number(state[key], 1, "°"));
  const baselineReady = live && Number.isFinite(state.pitch_baseline);
  $("pitch-calibration").dataset.ready = String(baselineReady);
  write("pitch-calibration", baselineReady ? `Pitch 已归零 · 基线 ${number(state.pitch_baseline, 1, "°")}`
    : live ? "Pitch 标定中 · 请保持自然坐姿看向前方" : "Pitch 等待观测");
  write("pitch-relative", number(state.pitch_relative, 1, "°"));
  write("blink-rate", number(state.blink_rate, 0, " 次/分"));
  write("blink-ms", number(state.blink_ms, 0, " ms"));
  renderDistraction(live ? state.distraction : null);
  renderSources(live ? state.sources : {});

  if (live) {
    if (revision !== state.revision || (points.length && state.elapsed < points.at(-1)[0])) {
      revision = state.revision;
      points.length = 0;
    }
    if (!points.length || state.elapsed > points.at(-1)[0]) {
      points.push([state.elapsed, state.score, timelineBreak]);
      timelineBreak = false;
    }
    while (points.length && state.elapsed - points[0][0] > chartWindow) points.shift();
  } else if (paused || resuming) {
    timelineBreak = true;
  } else {
    points.length = 0;
  }
  drawChart();
}

function drawChart() {
  const canvas = $("chart");
  const { width, height } = canvas.getBoundingClientRect();
  if (width < 100 || height < 60) return;
  const density = window.devicePixelRatio || 1;
  const context = canvas.getContext("2d");
  canvas.width = Math.round(width * density);
  canvas.height = Math.round(height * density);
  context.scale(density, density);
  const left = 36, right = width - 12, top = 12, bottom = height - 24;
  const maxScore = Math.max(12, Math.ceil(Math.max(0, ...points.map((point) => point[1])) / 6) * 6);
  const y = (value) => bottom - (bottom - top) * value / maxScore;
  const end = points.at(-1)?.[0] ?? 0;
  const x = (value) => right + (right - left) * (value - end) / chartWindow;
  context.font = '11px "Segoe UI", "Microsoft YaHei UI", sans-serif';
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const value = maxScore * i / 4;
    context.strokeStyle = "#293a47";
    context.beginPath();
    context.moveTo(left, y(value));
    context.lineTo(right, y(value));
    context.stroke();
    context.fillStyle = "#9aaab7";
    context.fillText(String(value), left - 7, y(value));
  }
  context.textBaseline = "top";
  context.textAlign = "center";
  for (const seconds of [30, 20, 10, 0]) {
    context.fillText(seconds ? `−${seconds}s` : "当前", x(end - seconds), bottom + 7);
  }
  if (!points.length) return;
  context.beginPath();
  points.forEach(([time, score, interrupted], index) => {
    if (index === 0 || interrupted || time - points[index - 1][0] > 1) context.moveTo(x(time), y(score));
    else context.lineTo(x(time), y(score));
  });
  context.strokeStyle = "#63dac1";
  context.lineWidth = 2;
  context.stroke();
  const [time, score] = points.at(-1);
  context.beginPath();
  context.arc(x(time), y(score), 3.5, 0, Math.PI * 2);
  context.fillStyle = "#63dac1";
  context.fill();
}

async function requestAction(action, enabled) {
  const control = action === "resume" ? "pause" : action;
  const expected = action === "resume" ? "paused" : "live";
  if (activeActions.has(control) || lastState?.status !== expected) return;
  activeActions.add(control);
  $(control).disabled = true;
  $("action-feedback").hidden = true;
  try {
    const response = await fetch("/api/actions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-DMS-Action": "1" },
      body: JSON.stringify({ action, enabled }),
    });
    if (!response.ok) throw new Error("action failed");
  } catch {
    write("action-feedback", "操作未成功，请检查本机监测是否仍在运行。");
    $("action-feedback").hidden = false;
  } finally {
    activeActions.delete(control);
  }
}

for (const action of ["mesh", "distraction"]) {
  $(action).addEventListener("change", (event) => requestAction(action, event.target.checked));
}
$("reset").addEventListener("click", () => requestAction("reset"));
$("calibrate").addEventListener("click", () => requestAction("calibrate"));
$("pause").addEventListener("click", () => requestAction(lastState?.status === "paused" ? "resume" : "pause"));
new ResizeObserver(drawChart).observe($("chart"));

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store", signal: AbortSignal.timeout(1500) });
    if (!response.ok) throw new Error("state unavailable");
    render(await response.json());
  } catch {
    render({ status: "offline", message: "与本机程序断开，请检查监测是否还在运行。" });
  } finally {
    window.setTimeout(refresh, 230);
  }
}
refresh();
