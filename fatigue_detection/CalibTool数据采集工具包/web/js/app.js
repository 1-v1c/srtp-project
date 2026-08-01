/* ═══════════════════════════════════════════════════════════════
   CalibTool 网页版 — 主逻辑
   算法与桌面版 (Python + MediaPipe) 完全一致：
   - EAR: 6 点眼部长宽比（阈值 0.22）
   - MAR: 3 组垂直测量嘴部长宽比（阈值 0.65）
   - HeadPose: faceTransformationMatrixes 提取欧拉角
   ═══════════════════════════════════════════════════════════════ */

import { FilesetResolver, FaceLandmarker } from "../vendor/vision_bundle.mjs";

/* ── 常量（与 Python config.py 对齐）── */
const EAR_THRESHOLD = 0.22;
const MAR_THRESHOLD = 0.65;
const SMOOTH_WINDOW = 5;

/* 关键点索引（MediaPipe Face Mesh 478 点） */
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH_VERTICAL_PAIRS = [[13, 14], [39, 181], [269, 405]];
const MOUTH_CORNER_LEFT  = 61;
const MOUTH_CORNER_RIGHT = 291;

/* 阶段与标签（与桌面版 PHASES 一致） */
const PHASES = {
  EAR:  ["正常睁眼", "眯眼", "完全闭眼", "快速眨眼", "侧脸睁眼"],
  MAR:  ["正常闭嘴", "微笑", "正常说话", "张嘴惊讶", "大笑", "真打哈欠"],
  HEAD: ["正对屏幕", "低头", "抬头", "左转", "右转", "左倾", "右倾"],
};
const PHASE_METRICS = {
  EAR:  ["EAR_L", "EAR_R", "EAR_avg"],
  MAR:  ["MAR"],
  HEAD: ["Pitch", "Yaw", "Roll"],
};

/* ── DOM ── */
const $ = (id) => document.getElementById(id);
const video     = $("video");
const canvas    = $("canvas");
const ctx       = canvas.getContext("2d");
const placeholder = $("camera-placeholder");
const placeholderText = $("placeholder-text");
const statusBanner = $("status-banner");
const recordList = $("record-list");

/* ── 几何工具 ── */
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

/* ── EAR ── */
function calcEar(lm, indices) {
  const [p1, p2, p3, p4, p5, p6] = indices.map(i => lm[i]);
  const v1 = dist(p2, p6);
  const v2 = dist(p3, p5);
  const h  = dist(p1, p4);
  if (h < 1e-6) return 0;
  return (v1 + v2) / (2 * h);
}

/* ── MAR（3 组垂直测量，DrowSAFE 增强版）── */
function calcMar(lm) {
  let vSum = 0;
  for (const [up, lo] of MOUTH_VERTICAL_PAIRS) vSum += dist(lm[up], lm[lo]);
  const h = dist(lm[MOUTH_CORNER_LEFT], lm[MOUTH_CORNER_RIGHT]);
  if (h < 1e-6) return 0;
  return vSum / (3 * h);
}

/* ── HeadPose：从 faceTransformationMatrixes 提取欧拉角 ──
   矩阵为 4x4 列主序 Float32Array[16]：
   R = [m0 m4 m8 ; m1 m5 m9 ; m2 m6 m10]
   公式与 Python 版 solvePnP 分解一致。 */
function calcPose(matrix) {
  if (!matrix) return { pitch: 0, yaw: 0, roll: 0 };
  const m = matrix;
  const r2d = 180 / Math.PI;
  // 约定与桌面版一致：pitch>0 低头, yaw>0 左转, roll>0 左歪
  let pitch = Math.atan2(m[6], m[10]);
  let yaw   = Math.asin(-m[2]);
  let roll  = Math.atan2(m[1], m[0]);
  if (!isFinite(pitch)) pitch = 0;
  if (!isFinite(yaw))   yaw   = 0;
  if (!isFinite(roll))  roll  = 0;
  return { pitch: pitch * r2d, yaw: yaw * r2d, roll: roll * r2d };
}

/* ── 滑动窗口平滑器 ── */
function makeSmoother(n) {
  const q = [];
  return (v) => {
    q.push(v);
    if (q.length > n) q.shift();
    return q.reduce((a, b) => a + b, 0) / q.length;
  };
}

/* ── 应用状态 ── */
const state = {
  phase: "EAR",
  labelIdx: 0,
  recording: false,
  recBuffer: [],        // 当前录制帧数据
  records: [],          // 完成录制的记录 { id, phase, label, time, stats, data }
  selectedRecord: null,
  pendingDelete: null,  // 待删除确认的记录 id
  earL: makeSmoother(SMOOTH_WINDOW),
  earR: makeSmoother(SMOOTH_WINDOW),
  mar: makeSmoother(SMOOTH_WINDOW),
  pose: makeSmoother(SMOOTH_WINDOW), // 返回 {pitch,yaw,roll} 平滑
  lastEar: 0, lastMar: 0, lastPose: { pitch: 0, yaw: 0, roll: 0 },
  faceFound: false,
  landmarker: null,
  lastVideoTime: -1,
};

/* 姿态平滑（三轴分别均值） */
function smoothPose(raw) {
  const q = state._poseQ || (state._poseQ = []);
  q.push(raw);
  if (q.length > SMOOTH_WINDOW) q.shift();
  return {
    pitch: q.reduce((a, p) => a + p.pitch, 0) / q.length,
    yaw:   q.reduce((a, p) => a + p.yaw, 0) / q.length,
    roll:  q.reduce((a, p) => a + p.roll, 0) / q.length,
  };
}

/* ═══════════ 初始化 ═══════════ */

async function init() {
  try {
    await loadModel();
    await startCamera();
    placeholder.classList.add("hidden");
    requestAnimationFrame(loop);
  } catch (e) {
    showStatus("初始化失败: " + e.message, "error");
  }
}

async function loadModel() {
  placeholderText.textContent = "正在加载 MediaPipe 模型…";
  // 注意：forVisionTasks 路径相对于页面 (index.html)，不是本文件
  const vision = await FilesetResolver.forVisionTasks("vendor");
  const opts = {
    baseOptions: {
      modelAssetPath: "../models/face_landmarker.task", // 模型在工具包根目录 models/
      delegate: "GPU",
    },
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 1,
  };
  try {
    state.landmarker = await FaceLandmarker.createFromOptions(vision, opts);
  } catch (e) {
    // GPU 加速不可用时回退 CPU（集成显卡/部分笔记本常见）
    console.warn("GPU delegate failed, fallback to CPU:", e);
    opts.baseOptions.delegate = "CPU";
    state.landmarker = await FaceLandmarker.createFromOptions(vision, opts);
  }
}

async function startCamera(w = 1280, h = 720) {
  placeholderText.textContent = "正在请求摄像头权限…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: w, height: h },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((res) => { video.onloadedmetadata = res; });
  await video.play();
  state.lastVideoTime = -1; // 换流后时间戳重置，避免跳帧
}

/* 切换分辨率：停旧流 → 改 canvas → 开新流 */
async function changeResolution(value) {
  const [w, h] = value.split("x").map(Number);
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  canvas.width = w;
  canvas.height = h;
  try {
    await startCamera(w, h);
    showStatus(`分辨率已切换为 ${w}×${h}`, "ok", 2500);
  } catch (err) {
    showStatus("切换分辨率失败: " + err.message, "error", 4000);
    // 回退到低分辨率重试
    $("res-select").value = "640x480";
    canvas.width = 640;
    canvas.height = 480;
    try { await startCamera(640, 480); } catch (e) { /* 摄像头可能被占用，用户手动重试 */ }
  }
}

/* ═══════════ 主循环 ═══════════ */

function loop(timestamp) {
  requestAnimationFrame(loop);

  if (!state.landmarker || video.readyState < 2) return;
  if (video.currentTime === state.lastVideoTime) return; // 帧未更新
  state.lastVideoTime = video.currentTime;

  let results = null;
  try {
    results = state.landmarker.detectForVideo(video, timestamp);
  } catch (e) {
    return; // 模型尚未就绪时的偶发异常，跳过本帧
  }

  const lm = results.faceLandmarks?.[0];
  const poseMatrix = results.facialTransformationMatrixes?.[0]?.data;
  state.faceFound = !!lm;

  /* 特征计算 */
  let earL = 0, earR = 0, earAvg = 0, mar = 0;
  let pose = { pitch: 0, yaw: 0, roll: 0 };
  if (lm) {
    earL = calcEar(lm, LEFT_EYE);
    earR = calcEar(lm, RIGHT_EYE);
    earAvg = state.earL(earL) + state.earR(earR);
    earAvg /= 2;
    mar = state.mar(calcMar(lm));
    pose = smoothPose(calcPose(poseMatrix));
  } else {
    state.earL(0); state.earR(0); state.mar(0);
  }
  state.lastEar = earAvg;
  state.lastMar = mar;
  state.lastPose = pose;

  /* 录制采集（无人脸帧跳过，与桌面版一致） */
  if (state.recording && lm) {
    state.recBuffer.push({
      t: (state.recBuffer.length / 30).toFixed(2),
      ear_l: +earL.toFixed(4),
      ear_r: +earR.toFixed(4),
      ear_avg: +earAvg.toFixed(4),
      mar: +mar.toFixed(4),
      pitch: +pose.pitch.toFixed(2),
      yaw: +pose.yaw.toFixed(2),
      roll: +pose.roll.toFixed(2),
    });
    updateStats();
  }

  draw(lm, earAvg, mar, pose);
  updateLiveData(earAvg, mar, pose);
}

/* ═══════════ 绘制 ═══════════ */

function draw(lm, earAvg, mar, pose) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  /* 视频帧 */
  ctx.save();
  ctx.scale(-1, 1); // 镜像显示（与桌面版一致，更像照镜子）
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (!lm) {
    drawText("未检测到人脸", canvas.width / 2, 60, "#8E8E93", 22);
    return;
  }

  /* 眼睛轮廓 */
  ctx.strokeStyle = earAvg < EAR_THRESHOLD ? "#FF453A" : "#30D158";
  ctx.lineWidth = 2;
  drawPolygon(lm, LEFT_EYE);
  drawPolygon(lm, RIGHT_EYE);

  /* 嘴部轮廓 */
  ctx.strokeStyle = mar > MAR_THRESHOLD ? "#FF9F0A" : "#0A84FF";
  drawPolygon(lm, [13, 14]);
  drawLine(lm[MOUTH_CORNER_LEFT], lm[MOUTH_CORNER_RIGHT]);

  /* 低头警告（基于有效低头角 = 原始 pitch − 偏移） */
  const off = +$("pitch-offset").value || 0;
  const effPitch = pose.pitch - off;
  if (effPitch > 22) {
    ctx.fillStyle = "rgba(255,69,58,0.85)";
    ctx.fillRect(0, 0, canvas.width, 34);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText("HEAD DOWN — 低头", canvas.width / 2, 24);
  }
}

function drawPolygon(lm, indices) {
  ctx.beginPath();
  // 视频帧已水平镜像，关键点 x 坐标同步取反 (1-x)
  ctx.moveTo((1 - lm[indices[0]].x) * canvas.width, lm[indices[0]].y * canvas.height);
  for (let i = 1; i < indices.length; i++) {
    ctx.lineTo((1 - lm[indices[i]].x) * canvas.width, lm[indices[i]].y * canvas.height);
  }
  ctx.stroke();
}

function drawLine(a, b) {
  ctx.beginPath();
  ctx.moveTo((1 - a.x) * canvas.width, a.y * canvas.height);
  ctx.lineTo((1 - b.x) * canvas.width, b.y * canvas.height);
  ctx.stroke();
}

function drawText(text, x, y, color, size) {
  ctx.font = `600 ${size}px 'Microsoft YaHei', sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, x, y);
}

/* ═══════════ 实时数据 ═══════════ */

function updateLiveData(ear, mar, pose) {
  const fmt = (v, d = 3) => (v === 0 && !state.faceFound ? "--" : v.toFixed(d));
  $("live-ear").innerHTML  = `EAR <b>${fmt(ear)}</b>`;
  $("live-mar").innerHTML  = `MAR <b>${fmt(mar)}</b>`;
  $("live-pose").innerHTML =
    `P <b>${fmt(pose.pitch, 1)}</b> Y <b>${fmt(pose.yaw, 1)}</b> R <b>${fmt(pose.roll, 1)}</b>`;
}

/* ═══════════ 录制 ═══════════ */

function currentLabel() { return PHASES[state.phase][state.labelIdx]; }

function startRecording() {
  state.recording = true;
  state.recBuffer = [];
  const btn = $("btn-record");
  btn.textContent = "空格 · 停止录制";
  btn.classList.add("recording");
  const st = $("rec-status");
  st.textContent = `正在录制 [${currentLabel()}]`;
  st.classList.add("recording");
  showStatus(`开始录制 [${currentLabel()}]`, "ok", 1500);
}

function stopRecording() {
  state.recording = false;
  const btn = $("btn-record");
  btn.textContent = "空格 · 开始录制";
  btn.classList.remove("recording");
  const st = $("rec-status");
  st.textContent = "未录制";
  st.classList.remove("recording");

  const data = state.recBuffer;
  if (data.length < 5) {
    showStatus("数据过少（<5 帧），未保存", "warn", 2500);
    return;
  }

  const stats = computeStats(data);
  const rec = {
    id: Date.now(),
    phase: state.phase,
    label: currentLabel(),
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    duration: (data.length / 30).toFixed(1) + "s",
    stats,
    data,
  };
  state.records.unshift(rec);
  state.selectedRecord = rec.id;
  renderRecords();
  updateStats();
  showStatus(`已保存 [${rec.label}] ${rec.duration}`, "ok", 2000);
}

function computeStats(data) {
  const metrics = PHASE_METRICS[state.phase];
  const stats = {};
  const keys = {
    EAR:  ["ear_l", "ear_r", "ear_avg"],
    MAR:  ["mar"],
    HEAD: ["pitch", "yaw", "roll"],
  }[state.phase];
  for (const k of keys) {
    const vals = data.map(d => d[k]).filter(v => isFinite(v));
    if (!vals.length) { stats[k] = { min: 0, max: 0, avg: 0 }; continue; }
    stats[k] = {
      min: Math.min(...vals),
      max: Math.max(...vals),
      avg: vals.reduce((a, b) => a + b, 0) / vals.length,
    };
  }
  return stats;
}

/* 录制中实时统计（侧面板） */
function updateStats() {
  const src = state.recording ? state.recBuffer : (state.selectedRecord ? selectedRecordData() : null);
  if (!src) {
    $("stat-cur").textContent = "--";
    $("stat-avg").textContent = "--";
    $("stat-min").textContent = "--";
    $("stat-max").textContent = "--";
    $("stat-hint").textContent = "录制中实时更新";
    return;
  }
  const key = { EAR: "ear_avg", MAR: "mar", HEAD: "pitch" }[state.phase];
  const vals = src.map(d => d[key]).filter(v => isFinite(v));
  if (!vals.length) return;
  const cur = vals[vals.length - 1];
  $("stat-cur").textContent = cur.toFixed(3);
  $("stat-avg").textContent = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3);
  $("stat-min").textContent = Math.min(...vals).toFixed(3);
  $("stat-max").textContent = Math.max(...vals).toFixed(3);
  $("stat-hint").textContent = state.recording ? "录制中…" : "选中记录";
}

function selectedRecordData() {
  const rec = state.records.find(r => r.id === state.selectedRecord);
  return rec ? rec.data : null;
}

/* ═══════════ 记录卡片 ═══════════ */

function renderRecords() {
  recordList.innerHTML = "";
  if (!state.records.length) {
    recordList.innerHTML = '<div class="record-empty">暂无记录，按空格开始录制</div>';
    return;
  }
  for (const rec of state.records) {
    const card = document.createElement("div");
    card.className = "record-card" + (rec.id === state.selectedRecord ? " selected" : "");
    const metrics = Object.entries(rec.stats)
      .map(([k, s]) => `${k}: ${s.avg.toFixed(3)}`)
      .join("  ");
    card.innerHTML = `
      <div class="rc-head">
        <span class="rc-label">[${rec.phase}] ${rec.label}</span>
        <span class="rc-time">${rec.time} · ${rec.duration}</span>
      </div>
      <div class="rc-metrics">${metrics}</div>
      <button class="rc-del">删除这条记录</button>`;
    card.addEventListener("click", () => {
      state.selectedRecord = rec.id;
      state.pendingDelete = null;
      renderRecords();
      updateStats();
    });
    card.querySelector(".rc-del").addEventListener("click", (e) => {
      e.stopPropagation();
      askDelete(rec);
    });
    recordList.appendChild(card);
  }
}

function askDelete(rec) {
  state.pendingDelete = rec.id;
  $("del-modal-text").textContent = `[${rec.phase}] ${rec.label} · ${rec.duration} · ${rec.time}`;
  $("del-modal").classList.remove("hidden");
}

function confirmDelete() {
  state.records = state.records.filter(r => r.id !== state.pendingDelete);
  state.pendingDelete = null;
  state.selectedRecord = null;
  $("del-modal").classList.add("hidden");
  renderRecords();
  updateStats();
  showStatus("记录已删除", "ok", 1500);
}

/* ═══════════ CSV 导出 ═══════════ */

function exportCSV() {
  if (!state.records.length) {
    showStatus("没有可导出的记录", "warn", 2000);
    return;
  }
  const header = "record,phase,label,time,t,ear_l,ear_r,ear_avg,mar,pitch,yaw,roll";
  const lines = [header];
  for (const rec of state.records) {
    for (const d of rec.data) {
      lines.push([
        rec.id, rec.phase, rec.label, rec.time,
        d.t, d.ear_l, d.ear_r, d.ear_avg, d.mar, d.pitch, d.yaw, d.roll,
      ].join(","));
    }
  }
  const csv = lines.join("\n");
  const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  const ts = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  a.href = URL.createObjectURL(blob);
  a.download = `calib_${ts}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  showStatus(`已导出 ${state.records.length} 条记录`, "ok", 2000);
}

/* ═══════════ UI：阶段 / 标签切换 ═══════════ */

function switchPhase(phase) {
  state.phase = phase;
  state.labelIdx = 0;
  document.querySelectorAll(".phase-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.phase === phase));
  $("phase-name").textContent = phase;
  renderLabelButtons();
}

function switchLabel(delta) {
  const labels = PHASES[state.phase];
  state.labelIdx = (state.labelIdx + delta + labels.length) % labels.length;
  renderLabelButtons();
}

function renderLabelButtons() {
  const wrap = $("label-buttons");
  wrap.innerHTML = "";
  PHASES[state.phase].forEach((label, i) => {
    const b = document.createElement("button");
    b.className = "label-btn" + (i === state.labelIdx ? " active" : "");
    b.textContent = label;
    b.addEventListener("click", () => { state.labelIdx = i; renderLabelButtons(); });
    wrap.appendChild(b);
  });
}

/* ═══════════ 工具 ═══════════ */

function showStatus(text, type, ms = 3000) {
  statusBanner.textContent = text;
  statusBanner.className = `status-banner ${type}`;
  clearTimeout(showStatus._t);
  showStatus._t = setTimeout(() => statusBanner.classList.add("hidden"), ms);
}

/* ═══════════ 事件绑定 ═══════════ */

/* 模式按钮 */
document.querySelectorAll(".phase-btn").forEach(b =>
  b.addEventListener("click", () => switchPhase(b.dataset.phase)));

/* 录制按钮 */
$("btn-record").addEventListener("click", () => {
  if (state.recording) stopRecording(); else startRecording();
});

/* 分辨率切换 */
$("res-select").addEventListener("change", (e) => changeResolution(e.target.value));

/* 导出 */
$("btn-export").addEventListener("click", exportCSV);

/* 帮助 */
$("help-close").addEventListener("click", () => $("help-modal").classList.add("hidden"));

/* 删除确认 */
$("del-confirm").addEventListener("click", confirmDelete);
$("del-cancel").addEventListener("click", () => {
  state.pendingDelete = null;
  $("del-modal").classList.add("hidden");
});

/* 点击弹窗外关闭 */
document.querySelectorAll(".modal").forEach(m =>
  m.addEventListener("click", (e) => {
    if (e.target === m) m.classList.add("hidden");
  }));

/* 快捷键 */
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return; // 输入框内不拦截
  switch (e.key) {
    case " ":        e.preventDefault(); state.recording ? stopRecording() : startRecording(); break;
    case "Tab":      e.preventDefault(); switchLabel(e.shiftKey ? -1 : 1); break;
    case "b": case "B": switchLabel(-1); break;
    case "1": switchPhase("EAR"); break;
    case "2": switchPhase("MAR"); break;
    case "3": switchPhase("HEAD"); break;
    case "e": case "E": exportCSV(); break;
    case "h": case "H": $("help-modal").classList.toggle("hidden"); break;
  }
});

/* ═══════════ 启动 ═══════════ */

renderLabelButtons();
init();
