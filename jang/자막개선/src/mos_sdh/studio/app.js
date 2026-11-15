// MoS-SDH Studio v1.0 — Interactive 4-Quadrant Client Controller (ES2024 Module)

const SPEAKER_HEX = [
  "#ffffff", // 화자_1: Pure White
  "#ffe885", // 화자_2: Soft Warm Yellow
  "#a7f3d0", // 화자_3: Mint Green
  "#93c5fd", // 화자_4: Soft Sky Blue
  "#f5d0fe", // 화자_5: Soft Lavender
  "#fed7aa", // 화자_6: Soft Peach
];

const STRATEGY_LABELS = {
  positional_tag_standard: "표준 위치 태그 (CAPE·정서·강조 통합)",
  tokuyama_intensifier_insertion: "정도 부사 보강 (Tokuyama 강조 삽입)",
  concise_low_cps: "저-CPS 고가독성 간결형",
  llm_augmented_zeroshot: "로컬 4-bit LLM 증강 제로샷 { } 재작성",
};

const EMOTION_COLORS = {
  분노: "#e11d48",
  냉소: "#9333ea",
  슬픔: "#2563eb",
  기쁨: "#059669",
  공포: "#d97706",
  중립: "#475569",
};

const state = {
  projectId: null,
  doc: null,
  projects: [],
  selectedSegmentId: null,
  subtitleMode: "ass", // 'ass' | 'srt' | 'raw'
  showDebugOverlay: true,
  activeFilter: "all",
  speakerFilter: null,
  searchQuery: "",
  inputTab: "upload",
};

// DOM References
const el = {
  gpuPill: document.getElementById("gpu-status-pill"),
  projectSelect: document.getElementById("project-select"),
  btnOpenAnalyze: document.getElementById("btn-open-analyze"),
  btnOpenSwer: document.getElementById("btn-open-swer"),
  videoStage: document.getElementById("video-stage"),
  video: document.getElementById("studio-video"),
  videoPlaceholder: document.getElementById("video-placeholder"),
  playerTimecode: document.getElementById("player-timecode"),
  chkDebug: document.getElementById("chk-debug-overlay"),
  debugLayer: document.getElementById("debug-overlay-layer"),
  hudLeft: document.getElementById("hud-top-left"),
  hudRight: document.getElementById("hud-top-right"),
  ocrBoxesLayer: document.getElementById("ocr-boxes-layer"),
  subSlot: document.getElementById("subtitle-overlay-slot"),
  subBox: document.getElementById("subtitle-overlay-box"),
  btnClearSpkFilter: document.getElementById("btn-clear-spk-filter"),
  paneRelations: document.getElementById("macro-pane-relations"),
  paneQuality: document.getElementById("macro-pane-quality"),
  relationSvg: document.getElementById("relation-svg"),
  sceneCardsList: document.getElementById("scene-cards-list"),
  kpiGrid: document.getElementById("kpi-grid"),
  scoreTextBadge: document.getElementById("score-text-badge"),
  scoreDistBars: document.getElementById("score-dist-bars"),
  rngCueCap: document.getElementById("rng-cue-cap"),
  lblCueCap: document.getElementById("lbl-cue-cap"),
  rngCpsCap: document.getElementById("rng-cps-cap"),
  lblCpsCap: document.getElementById("lbl-cps-cap"),
  btnApplyPrune: document.getElementById("btn-apply-prune"),
  vramBars: document.getElementById("vram-bars"),
  tracksCanvas: document.getElementById("tracks-canvas"),
  timeRuler: document.getElementById("time-ruler"),
  laneProsody: document.getElementById("lane-prosody"),
  laneEmotion: document.getElementById("lane-emotion"),
  laneHonorific: document.getElementById("lane-honorific"),
  laneOcr: document.getElementById("lane-ocr"),
  playhead: document.getElementById("timeline-playhead"),
  filterPills: document.getElementById("filter-pills"),
  searchInput: document.getElementById("seg-search-input"),
  segmentTbody: document.getElementById("segment-tbody"),
  inspectorHeading: document.getElementById("inspector-heading"),
  inspectorSub: document.getElementById("inspector-sub"),
  spkRenameGroup: document.getElementById("spk-rename-group"),
  inpSpeakerId: document.getElementById("inp-speaker-id"),
  btnSaveSpeaker: document.getElementById("btn-save-speaker"),
  inspectorBody: document.getElementById("inspector-body"),
  toast: document.getElementById("studio-toast"),
  analyzeDialog: document.getElementById("analyze-dialog"),
  analyzeForm: document.getElementById("analyze-form"),
  paneInputUpload: document.getElementById("pane-input-upload"),
  paneInputPath: document.getElementById("pane-input-path"),
  analyzeProgress: document.getElementById("analyze-progress-box"),
  btnSubmitAnalyze: document.getElementById("btn-submit-analyze"),
  swerDialog: document.getElementById("swer-dialog"),
  btnCalcSwer: document.getElementById("btn-calc-swer"),
  swerResultBox: document.getElementById("swer-result-box"),
};

function esc(val) {
  return String(val ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtMs(ms) {
  const safe = Math.max(0, Math.round(ms || 0));
  const totalSec = Math.floor(safe / 1000);
  const milli = safe % 1000;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(milli).padStart(3, "0")}`;
}

let toastTimer = null;
function showToast(msg) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.hidden = true;
  }, 3000);
}

function getSpeakerColor(speakerId) {
  if (!state.doc) return SPEAKER_HEX[0];
  const speakers = [...new Set(state.doc.segments.map((s) => s.speaker_id))].sort();
  const idx = Math.max(0, speakers.indexOf(speakerId));
  return SPEAKER_HEX[idx % SPEAKER_HEX.length];
}

// ============================================================================
// Initialization & API Calls
// ============================================================================
async function initStudio() {
  bindEvents();
  await fetchHealth();
  await refreshProjectList();
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/v1/health");
    const data = await res.json();
    if (data.cuda_available) {
      el.gpuPill.textContent = `${data.gpu_name} · ${data.total_vram_gb} GB · CUDA 활성`;
    } else {
      el.gpuPill.textContent = `CPU 폴백 모드 · v${data.version}`;
    }
  } catch {
    el.gpuPill.textContent = "오프라인";
  }
}

async function refreshProjectList(selectedId = null) {
  const res = await fetch("/api/v1/projects");
  state.projects = await res.json();

  el.projectSelect.innerHTML = "";
  const placeholderOpt = document.createElement("option");
  placeholderOpt.value = "";
  placeholderOpt.textContent = state.projects.length === 0 ? "분석된 프로젝트 없음" : "프로젝트 선택…";
  el.projectSelect.appendChild(placeholderOpt);
  if (state.projects.length === 0) {
    return;
  }

  for (const p of state.projects) {
    const opt = document.createElement("option");
    opt.value = p.project_id;
    opt.textContent = `${p.label} [${p.total_segments}문장]`;
    el.projectSelect.appendChild(opt);
  }

  if (selectedId) {
    el.projectSelect.value = selectedId;
  }
}

async function loadProject(projectId) {
  if (!projectId) return;
  const res = await fetch(`/api/v1/projects/${encodeURIComponent(projectId)}`);
  if (!res.ok) {
    showToast("프로젝트를 불러오지 못했습니다.");
    return;
  }
  const doc = await res.json();
  state.projectId = projectId;
  state.doc = doc;
  state.speakerFilter = null;
  el.btnClearSpkFilter.hidden = true;
  el.projectSelect.value = projectId;

  const projMeta = state.projects.find((p) => p.project_id === projectId);
  if (projMeta && projMeta.has_media) {
    el.video.src = `/api/v1/projects/${encodeURIComponent(projectId)}/media`;
    el.videoPlaceholder.hidden = true;
  } else {
    el.video.removeAttribute("src");
    el.video.load();
    el.videoPlaceholder.hidden = false;
    el.videoPlaceholder.innerHTML = `<p>미디어 원본 파일이 없거나 경로가 이동되었습니다.<br/>타임라인 및 자막 후보 편집은 그대로 사용할 수 있습니다.</p>`;
  }

  state.selectedSegmentId = doc.segments.length > 0 ? doc.segments[0].segment_id : null;
  renderAll();
}

function renderAll() {
  if (!state.doc) return;
  renderFilterCounts();
  renderRelationGraphAndScenes();
  renderMacroQualityAndVram();
  renderMultitrackTimeline();
  renderDiffTable();
  renderInspector();
  updateVideoOverlays();
}

// ============================================================================
// Quadrant 1: Video Preview + Live .ASS/.SRT + Debug OCR/ASD Overlays
// ============================================================================
function getCurrentTimeMs() {
  if (el.video.readyState >= 1 && !Number.isNaN(el.video.currentTime)) {
    return Math.round(el.video.currentTime * 1000);
  }
  const sel = getSelectedSegment();
  return sel ? sel.start_ms + 50 : 0;
}

function findActiveSegmentAt(tMs) {
  if (!state.doc) return null;
  return (
    state.doc.segments.find(
      (s) => tMs >= s.start_ms && tMs <= Math.max(s.start_ms + 300, s.end_ms)
    ) || null
  );
}

function getRenderedVideoRect() {
  const stageRect = el.videoStage.getBoundingClientRect();
  const vw = el.video.videoWidth || 16;
  const vh = el.video.videoHeight || 9;
  const stageRatio = stageRect.width / Math.max(1, stageRect.height);
  const vidRatio = vw / Math.max(1, vh);

  let width = stageRect.width;
  let height = stageRect.height;
  let left = 0;
  let top = 0;

  if (vidRatio > stageRatio) {
    height = stageRect.width / vidRatio;
    top = (stageRect.height - height) / 2;
  } else {
    width = stageRect.height * vidRatio;
    left = (stageRect.width - width) / 2;
  }
  return { left, top, width, height };
}

function updateVideoOverlays() {
  if (!state.doc) return;
  const tMs = getCurrentTimeMs();
  const totalMs = Math.max(
    state.doc.duration_ms || 1,
    Math.round((el.video.duration || 0) * 1000)
  );
  el.playerTimecode.textContent = `${fmtMs(tMs)} / ${fmtMs(totalMs)}`;

  const pct = Math.min(100, Math.max(0, (tMs / Math.max(1, totalMs)) * 100));
  el.playhead.style.left = `${pct.toFixed(3)}%`;

  const activeSeg = findActiveSegmentAt(tMs);
  renderLiveSubtitleOverlay(activeSeg);
  renderDebugOverlay(activeSeg, tMs);
}

function renderLiveSubtitleOverlay(seg) {
  if (!seg) {
    el.subBox.hidden = true;
    return;
  }
  el.subBox.hidden = false;

  // Position top/bottom & left/center/right
  const isAssMode = state.subtitleMode === "ass";
  const topAvoid = isAssMode && Boolean(seg.visual?.avoid_bottom_region);
  const zone = isAssMode ? seg.visual?.speaker_zone || "center" : "center";

  el.subSlot.className = `subtitle-overlay-slot ${topAvoid ? "pos-top" : "pos-bottom"} pos-${zone}`;

  if (state.subtitleMode === "raw") {
    el.subBox.style.color = "#e2e8f0";
    el.subBox.textContent = seg.existing_subtitle || seg.raw_transcript;
    return;
  }

  if (state.subtitleMode === "srt") {
    el.subBox.style.color = "#ffffff";
    el.subBox.textContent = seg.sdh_srt_text;
    return;
  }

  // Rich .ASS Visual Typography Preview
  el.subBox.style.color = getSpeakerColor(seg.speaker_id);
  const spkLabel =
    seg.visual?.on_screen_speaker === false
      ? `[${seg.speaker_id}(화면 밖)]`
      : `[${seg.speaker_id}]`;

  const parts = [`<span>${esc(spkLabel)}</span> `];

  if (seg.emotion_irony?.irony_detected) {
    parts.push(`<span class="sub-tone-irony">(비꼬는 어조로)</span>`);
  } else if (seg.honorific?.consistency_status === "dramatic_shift") {
    const note = seg.honorific.is_polite_group === false ? "격앙되어 반말로" : "어조 전환";
    parts.push(`<span class="sub-tone-dramatic">(${esc(note)})</span>`);
  }

  // Word-level <toe_k> visual scaling when matching selected text, else clean ASS stripped body
  if (
    seg.words &&
    seg.words.length > 0 &&
    seg.honorific?.consistency_status !== "mismatch_error"
  ) {
    const styledWords = seg.words.map((w) => {
      if (w.emphasis_level >= 4) return `<span class="emph-lv4">${esc(w.text)}</span>`;
      if (w.emphasis_level === 3) return `<span class="emph-lv3">${esc(w.text)}</span>`;
      if (w.emphasis_level === 2) return `<span class="emph-lv2">${esc(w.text)}</span>`;
      return esc(w.text);
    });
    parts.push(styledWords.join(" "));
  } else {
    // Strip raw ASS brace tags from seg.sdh_ass_text for clean HTML preview
    const stripped = (seg.sdh_ass_text || seg.sdh_srt_text)
      .replace(/\{[^}]*\}/g, "")
      .replace(/^\[[^\]]+\]\s*/, "")
      .replace(/^\([^)]+\)\s*/, "");
    parts.push(esc(stripped));
  }

  if (seg.cues && seg.cues.length > 0) {
    const cueTxt = seg.cues.map((c) => c.label_ko).join(", ");
    parts.push(`<span class="sub-cue-tail">(${esc(cueTxt)})</span>`);
  }

  el.subBox.innerHTML = parts.join(" ");
}

function renderDebugOverlay(seg, tMs) {
  el.debugLayer.hidden = !state.showDebugOverlay;
  if (!state.showDebugOverlay || !state.doc) return;

  const currentScene =
    state.doc.scenes.find((sc) => tMs >= sc.start_ms && tMs <= sc.end_ms) ||
    state.doc.scenes[0];

  const leftChips = [];
  if (currentScene) {
    leftChips.push(`<span class="hud-chip">씬 #${currentScene.scene_id}</span>`);
  }
  if (seg) {
    const asdTxt =
      seg.visual?.on_screen_speaker === false
        ? "ASD: 화면 밖 발화"
        : `ASD: 화면 내 (${seg.visual?.speaker_zone || "center"} · lip ${seg.visual?.lip_activity_score?.toFixed(3) ?? "0"})`;
    leftChips.push(`<span class="hud-chip">${esc(asdTxt)}</span>`);
    if (seg.visual?.facial_expression) {
      leftChips.push(`<span class="hud-chip">표정(FER): ${esc(seg.visual.facial_expression)}</span>`);
    }
  }
  el.hudLeft.innerHTML = leftChips.join("");

  const rightChips = [];
  if (seg) {
    rightChips.push(
      `<span class="hud-chip">정서: ${esc(seg.emotion_irony?.dominant_emotion || "중립")} · 각성 Z=${seg.arousal_zscore >= 0 ? "+" : ""}${seg.arousal_zscore.toFixed(2)}</span>`
    );
    if (seg.emotion_irony?.irony_detected) {
      rightChips.push(
        `<span class="hud-chip chip-irony">⚡ 반어법 감지 (I=${seg.emotion_irony.irony_score.toFixed(2)})</span>`
      );
    }
    if (seg.visual?.avoid_bottom_region) {
      rightChips.push(`<span class="hud-chip chip-ocr">⬆ 하단 OCR 충돌 → 상단 {\\an8} 회피</span>`);
    }
  }
  el.hudRight.innerHTML = rightChips.join("");

  // Normalized OCR bounding boxes mapped to rendered video frame
  el.ocrBoxesLayer.innerHTML = "";
  const boxes = seg?.visual?.ocr_boxes || [];
  if (boxes.length > 0) {
    const vRect = getRenderedVideoRect();
    for (const b of boxes) {
      const div = document.createElement("div");
      div.className = "ocr-bbox";
      div.style.left = `${vRect.left + b.x_min * vRect.width}px`;
      div.style.top = `${vRect.top + b.y_min * vRect.height}px`;
      div.style.width = `${Math.max(16, (b.x_max - b.x_min) * vRect.width)}px`;
      div.style.height = `${Math.max(12, (b.y_max - b.y_min) * vRect.height)}px`;

      const lbl = document.createElement("span");
      lbl.className = "ocr-bbox-label";
      lbl.textContent = `OCR: ${b.text} (${Math.round(b.confidence * 100)}%)`;
      div.appendChild(lbl);
      el.ocrBoxesLayer.appendChild(div);
    }
  }
}

// ============================================================================
// Quadrant 2: Speaker 6-Speech-Level Graph, Scene Cards & Macro Quality
// ============================================================================
function renderRelationGraphAndScenes() {
  const doc = state.doc;
  const speakers = [...new Set(doc.segments.map((s) => s.speaker_id))].sort();

  // Gather relations from scenes; if single speaker + '상대방', include '상대방' virtual node
  const relations = [];
  for (const sc of doc.scenes || []) {
    for (const r of sc.inferred_relations || []) {
      if (!relations.some((x) => x.from_speaker === r.from_speaker && x.to_speaker === r.to_speaker)) {
        relations.push(r);
      }
    }
  }

  const nodeNames = [...speakers];
  for (const r of relations) {
    if (!nodeNames.includes(r.from_speaker)) nodeNames.push(r.from_speaker);
    if (!nodeNames.includes(r.to_speaker)) nodeNames.push(r.to_speaker);
  }
  if (nodeNames.length === 1) {
    nodeNames.push("대화 상대");
  }

  const W = 540;
  const H = 165;
  const cx = W / 2;
  const cy = H / 2 + 4;
  const rx = Math.min(185, W * 0.34);
  const ry = 52;

  const coords = {};
  nodeNames.forEach((name, idx) => {
    const angle = Math.PI + (idx * 2 * Math.PI) / Math.max(1, nodeNames.length);
    coords[name] = {
      x: cx + rx * Math.cos(angle),
      y: nodeNames.length <= 2 ? cy : cy + ry * Math.sin(angle),
    };
  });

  // Fallback empirical edge if no explicit scene relation
  const edgesToDraw =
    relations.length > 0
      ? relations
      : speakers.map((sp) => {
          const domLevel = doc.segments.find((s) => s.speaker_id === sp)?.honorific?.speech_level || "미분류";
          const target = nodeNames.find((n) => n !== sp) || "대화 상대";
          return {
            from_speaker: sp,
            to_speaker: target,
            relation_ko: "주류 화계",
            prior_speech_level: domLevel,
          };
        });

  let svgHtml = `
    <defs>
      <marker id="arrow-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
      </marker>
    </defs>
  `;

  edgesToDraw.forEach((edge, i) => {
    const p1 = coords[edge.from_speaker] || { x: 110, y: cy };
    const p2 = coords[edge.to_speaker] || { x: 430, y: cy };
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy) || 1;
    const pad = 30;
    const sx = p1.x + (dx / len) * pad;
    const sy = p1.y + (dy / len) * pad;
    const ex = p2.x - (dx / len) * pad;
    const ey = p2.y - (dy / len) * pad;
    const curveOffset = i % 2 === 0 ? -18 : 18;
    const mx = (sx + ex) / 2;
    const my = (sy + ey) / 2 + curveOffset;

    svgHtml += `
      <path d="M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}" fill="none" stroke="#38bdf8" stroke-width="1.8" marker-end="url(#arrow-head)" opacity="0.85" />
      <rect x="${mx - 68}" y="${my - 11}" width="136" height="18" rx="4" fill="#0f172a" stroke="#334155" />
      <text x="${mx}" y="${my + 2}" text-anchor="middle" fill="#fde68a" font-size="10" font-weight="700">
        ${esc(edge.prior_speech_level)} · ${esc(edge.relation_ko.slice(0, 12))}
      </text>
    `;
  });

  for (const name of nodeNames) {
    const pt = coords[name];
    const isRealSpk = speakers.includes(name);
    const isActive = state.speakerFilter === name;
    const stroke = isActive ? "#fbbf24" : isRealSpk ? getSpeakerColor(name) : "#475569";
    svgHtml += `
      <g class="spk-node" data-speaker="${esc(name)}" style="cursor: ${isRealSpk ? "pointer" : "default"}">
        <circle cx="${pt.x}" cy="${pt.y}" r="25" fill="#162136" stroke="${stroke}" stroke-width="${isActive ? 3 : 2}" />
        <text x="${pt.x}" y="${pt.y + 4}" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">${esc(name)}</text>
      </g>
    `;
  }

  el.relationSvg.innerHTML = svgHtml;
  el.relationSvg.querySelectorAll(".spk-node").forEach((node) => {
    node.addEventListener("click", () => {
      const spk = node.getAttribute("data-speaker");
      if (!speakers.includes(spk)) return;
      state.speakerFilter = state.speakerFilter === spk ? null : spk;
      el.btnClearSpkFilter.hidden = !state.speakerFilter;
      renderRelationGraphAndScenes();
      renderDiffTable();
    });
  });

  // Scene Context Cards
  if (!doc.scenes || doc.scenes.length === 0) {
    el.sceneCardsList.innerHTML = `<div class="empty-hint">오디오 전용 실행이거나 시각 씬 정보가 없습니다.</div>`;
    return;
  }

  el.sceneCardsList.innerHTML = doc.scenes
    .map((sc) => {
      const ocrChips = (sc.ocr_summary || [])
        .map((txt) => `<span class="ocr-chip">OCR: ${esc(txt)}</span>`)
        .join("");
      return `
        <div class="scene-card" data-seek-ms="${sc.start_ms}">
          <div class="scene-card-head">
            <span>씬 #${sc.scene_id}</span>
            <span class="mono-pill">${fmtMs(sc.start_ms)} ~ ${fmtMs(sc.end_ms)}</span>
          </div>
          <div class="scene-card-desc">${esc(sc.description_ko || `씬 ${sc.scene_id}`)}</div>
          ${ocrChips ? `<div class="scene-ocr-chips">${ocrChips}</div>` : ""}
        </div>
      `;
    })
    .join("");

  el.sceneCardsList.querySelectorAll(".scene-card").forEach((card) => {
    card.addEventListener("click", () => {
      seekVideoMs(Number(card.getAttribute("data-seek-ms") || 0));
    });
  });
}

function renderMacroQualityAndVram() {
  const m = state.doc.metrics;
  const swerTxt = m.swer_comparison
    ? `${m.swer_comparison.swer_percent.toFixed(1)}% (WER ${m.swer_comparison.wer_percent.toFixed(1)}%)`
    : "원음 단독(Mode A)";

  const kpis = [
    { label: "6화계 문맥 일치율", val: `${m.honorific_consistency_rate.toFixed(1)}%`, color: "#38bdf8" },
    { label: "의미 보존 C_i · 조화 O_i", val: `${m.mean_semantic_ci.toFixed(2)} / ${m.mean_harmonic_oi.toFixed(2)}`, color: "#34d399" },
    { label: "반어법 · 화계오류 · 전환", val: `${m.irony_segment_count} / ${m.mismatch_error_count} / ${m.dramatic_shift_count}건`, color: "#fbbf24" },
    { label: "기존 자막 공백정규화 sWER", val: swerTxt, color: "#c084fc" },
    { label: "평균 CPS · 지문 밀도", val: `${m.mean_cps.toFixed(1)}자/s · ${(m.cue_attachment_ratio * 100).toFixed(0)}%`, color: "#f8fafc" },
    { label: "화면밖 ASD · OCR 상단회피", val: `${m.off_screen_segment_count}건 / ${m.ocr_collision_avoided_count}건`, color: "#6ee7b7" },
  ];

  el.kpiGrid.innerHTML = kpis
    .map(
      (k) => `
      <div class="kpi-card">
        <div class="kpi-label">${esc(k.label)}</div>
        <div class="kpi-val" style="color: ${k.color}">${esc(k.val)}</div>
      </div>
    `
    )
    .join("");

  el.scoreTextBadge.textContent = `Score_text: ${m.score_text_intensity.toFixed(1)} / 100`;
  const bins = m.score_dist_5bin?.length === 5 ? m.score_dist_5bin : [0, 0, 0, 0, 0];
  el.scoreDistBars.innerHTML = bins
    .map((val, idx) => {
      const hPct = Math.max(6, Math.min(100, val));
      return `
        <div class="dist-col" title="구간 ${idx + 1}: ${val.toFixed(1)}">
          <div class="dist-bar" style="height: ${hPct}%"></div>
          <span class="dist-lbl">Q${idx + 1} (${Math.round(val)})</span>
        </div>
      `;
    })
    .join("");

  // VRAM Telemetry
  const vram = state.doc.vram_telemetry || [];
  if (vram.length === 0) {
    el.vramBars.innerHTML = `<div class="empty-hint" style="padding:6px">기록된 GPU 텔레메트리가 없습니다.</div>`;
  } else {
    const maxMb = Math.max(1024, ...vram.map((v) => v.peak_allocated_mb));
    el.vramBars.innerHTML = vram
      .map((t) => {
        const wPct = Math.min(100, (t.peak_allocated_mb / maxMb) * 100);
        const shortName = t.stage_name.replace(/^Stage\s+/, "");
        return `
          <div class="vram-row" title="진입 ${t.allocated_before_mb}MB → 피크 ${t.peak_allocated_mb}MB → 회수 후 ${t.allocated_after_mb}MB">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(shortName)}</span>
            <div class="vram-track"><div class="vram-fill" style="width:${wPct.toFixed(1)}%"></div></div>
            <span class="mono-pill" style="text-align:right">${t.peak_allocated_mb.toFixed(0)}→${t.allocated_after_mb.toFixed(0)}MB</span>
          </div>
        `;
      })
      .join("");
  }
}

// ============================================================================
// Middle Band: 4-Track Synchronized MoS Timeline Scrubber
// ============================================================================
function renderMultitrackTimeline() {
  const doc = state.doc;
  const totalMs = Math.max(
    1000,
    doc.duration_ms || 0,
    ...doc.segments.map((s) => s.end_ms + 200)
  );

  // Ruler ticks (6 intervals)
  let rulerHtml = "";
  for (let i = 0; i <= 5; i++) {
    const pct = i * 20;
    const tickMs = (totalMs * i) / 5;
    rulerHtml += `<span class="ruler-tick" style="left:${pct}%">${fmtMs(tickMs)}</span>`;
  }
  el.timeRuler.innerHTML = rulerHtml;

  let prosodyHtml = "";
  let emotionHtml = "";
  let honHtml = "";
  let ocrHtml = "";

  for (const seg of doc.segments) {
    const leftPct = Math.max(0, (seg.start_ms / totalMs) * 100);
    const widthPct = Math.max(1.2, ((seg.end_ms - seg.start_ms) / totalMs) * 100);
    const selCls = seg.segment_id === state.selectedSegmentId ? "is-selected" : "";
    const posStyle = `left:${leftPct.toFixed(2)}%; width:${widthPct.toFixed(2)}%;`;

    // Track 1: Prosody & <toe_k> Emphasis
    const maxEmph = Math.max(0, ...(seg.words || []).map((w) => w.emphasis_level));
    const pColor = maxEmph >= 3 ? "rgba(251, 191, 36, 0.42)" : "rgba(56, 189, 248, 0.28)";
    const stars = maxEmph >= 3 ? `★e${maxEmph} ` : "";
    prosodyHtml += `<div class="tl-seg ${selCls}" data-seg-id="${seg.segment_id}" style="${posStyle} background:${pColor}" title="#${seg.segment_id} 각성 Z=${seg.arousal_zscore.toFixed(2)}, 최대 강조=${maxEmph}">${esc(stars)}Z=${seg.arousal_zscore.toFixed(1)}</div>`;

    // Track 2: Emotion & Irony (CmGI)
    const emo = seg.emotion_irony?.dominant_emotion || "중립";
    const emoBg = EMOTION_COLORS[emo] || "#475569";
    const ironyTag = seg.emotion_irony?.irony_detected ? "⚡반어 " : "";
    emotionHtml += `<div class="tl-seg ${selCls}" data-seg-id="${seg.segment_id}" style="${posStyle} background:${emoBg}" title="#${seg.segment_id} 정서=${emo}, 반어점수=${seg.emotion_irony?.irony_score?.toFixed(2) ?? 0}">${esc(ironyTag + emo)}</div>`;

    // Track 3: Korean 6-Speech-Levels & Consistency
    const st = seg.honorific?.consistency_status;
    const hBg =
      st === "mismatch_error"
        ? "#e11d48"
        : st === "dramatic_shift"
        ? "#ea580c"
        : seg.honorific?.is_polite_group
        ? "#0284c7"
        : "#4f46e5";
    const hPrefix = st === "mismatch_error" ? "🔴" : st === "dramatic_shift" ? "🟠" : "";
    honHtml += `<div class="tl-seg ${selCls}" data-seg-id="${seg.segment_id}" style="${posStyle} background:${hBg}" title="#${seg.segment_id} ${seg.honorific?.speech_level} (${st})">${esc(hPrefix + (seg.honorific?.speech_level || ""))}</div>`;

    // Track 4: Subtitle Block & OCR Collision / ASD
    const ocrHit = seg.visual?.avoid_bottom_region;
    const oBg = ocrHit ? "rgba(16, 185, 129, 0.48)" : "rgba(100, 116, 139, 0.35)";
    const oTxt = ocrHit ? "⬆\\an8 회피" : seg.visual?.on_screen_speaker === false ? "화면밖" : seg.speaker_id;
    ocrHtml += `<div class="tl-seg ${selCls}" data-seg-id="${seg.segment_id}" style="${posStyle} background:${oBg}" title="#${seg.segment_id} ${seg.sdh_srt_text}">${esc(oTxt)}</div>`;
  }

  el.laneProsody.innerHTML = prosodyHtml;
  el.laneEmotion.innerHTML = emotionHtml;
  el.laneHonorific.innerHTML = honHtml;
  el.laneOcr.innerHTML = ocrHtml;

  el.tracksCanvas.querySelectorAll(".tl-seg").forEach((segNode) => {
    segNode.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const sid = Number(segNode.getAttribute("data-seg-id"));
      selectSegment(sid, true);
    });
  });
}

// ============================================================================
// Bottom-Left: Subtitle Diff Table & Filters
// ============================================================================
function matchesFilter(seg) {
  if (state.speakerFilter && seg.speaker_id !== state.speakerFilter) return false;

  if (state.activeFilter === "irony" && !seg.emotion_irony?.irony_detected) return false;
  if (state.activeFilter === "mismatch" && seg.honorific?.consistency_status !== "mismatch_error") return false;
  if (state.activeFilter === "dramatic" && seg.honorific?.consistency_status !== "dramatic_shift") return false;
  if (state.activeFilter === "ocr" && !seg.visual?.avoid_bottom_region) return false;
  if (
    state.activeFilter === "emphasis" &&
    !(seg.words || []).some((w) => w.emphasis_level >= 3)
  ) {
    return false;
  }

  if (state.searchQuery) {
    const q = state.searchQuery.toLowerCase();
    const hay = `${seg.speaker_id} ${seg.raw_transcript} ${seg.existing_subtitle || ""} ${seg.sdh_srt_text}`.toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function renderFilterCounts() {
  const segs = state.doc?.segments || [];
  const counts = {
    all: segs.length,
    irony: segs.filter((s) => s.emotion_irony?.irony_detected).length,
    mismatch: segs.filter((s) => s.honorific?.consistency_status === "mismatch_error").length,
    dramatic: segs.filter((s) => s.honorific?.consistency_status === "dramatic_shift").length,
    ocr: segs.filter((s) => s.visual?.avoid_bottom_region).length,
    emphasis: segs.filter((s) => (s.words || []).some((w) => w.emphasis_level >= 3)).length,
  };
  const labels = {
    all: "전체",
    irony: "⚡ 반어법",
    mismatch: "🔴 화계 오류",
    dramatic: "🟠 극적 전환",
    ocr: "⬆ OCR 회피",
    emphasis: "★ 강한 강조",
  };
  el.filterPills.querySelectorAll(".pill-btn").forEach((btn) => {
    const key = btn.getAttribute("data-filter");
    btn.textContent = `${labels[key]} (${counts[key] ?? 0})`;
  });
}

function renderDiffTable() {
  const filtered = (state.doc?.segments || []).filter(matchesFilter);
  if (filtered.length === 0) {
    el.segmentTbody.innerHTML = `<tr><td colspan="7" class="empty-cell">조건에 해당하는 대사 세그먼트가 없습니다.</td></tr>`;
    return;
  }

  el.segmentTbody.innerHTML = filtered
    .map((seg) => {
      const isSel = seg.segment_id === state.selectedSegmentId;
      const origTxt = seg.existing_subtitle ?? seg.raw_transcript;
      const badges = [];

      badges.push(`<span class="badge" style="background:#1e293b;color:#e2e8f0">${esc(seg.honorific?.speech_level || "미분류")}</span>`);
      if (seg.emotion_irony?.irony_detected) {
        badges.push(`<span class="badge badge-irony">⚡ 반어 ${seg.emotion_irony.irony_score.toFixed(2)}</span>`);
      }
      if (seg.honorific?.consistency_status === "mismatch_error") {
        badges.push(`<span class="badge badge-err">🔴 화계오류→CAPE</span>`);
      } else if (seg.honorific?.consistency_status === "dramatic_shift") {
        badges.push(`<span class="badge badge-shift">🟠 극적전환</span>`);
      }
      if (seg.visual?.avoid_bottom_region) {
        badges.push(`<span class="badge badge-ocr">⬆ \\an8</span>`);
      }
      if ((seg.words || []).some((w) => w.emphasis_level >= 3)) {
        badges.push(`<span class="badge badge-emph">★ 강조</span>`);
      }

      const locTxt = seg.visual?.on_screen_speaker === false ? "화면 밖" : seg.visual?.speaker_zone || "center";
      const spkColor = getSpeakerColor(seg.speaker_id);

      return `
        <tr class="${isSel ? "is-selected" : ""}" data-row-seg="${seg.segment_id}">
          <td class="col-id">${seg.segment_id}</td>
          <td class="col-time">${fmtMs(seg.start_ms)}<br/>~${fmtMs(seg.end_ms)}</td>
          <td class="col-spk">
            <strong style="color:${spkColor}">${esc(seg.speaker_id)}</strong>
            <div style="font-size:10px;color:var(--text-dim)">${esc(locTxt)}</div>
          </td>
          <td class="col-orig" style="color:var(--text-muted)">${esc(origTxt)}</td>
          <td class="col-sdh"><strong>${esc(seg.sdh_srt_text)}</strong></td>
          <td class="col-badges">${badges.join("")}</td>
          <td class="col-cps ${seg.cps_exceeded ? "cps-warn" : ""}">${seg.cps.toFixed(1)}</td>
        </tr>
      `;
    })
    .join("");

  el.segmentTbody.querySelectorAll("tr[data-row-seg]").forEach((row) => {
    row.addEventListener("click", () => {
      const sid = Number(row.getAttribute("data-row-seg"));
      selectSegment(sid, true);
    });
  });
}

// ============================================================================
// Bottom-Right: Segment Evidence & AI 4-Candidate Inspector
// ============================================================================
function getSelectedSegment() {
  if (!state.doc || state.selectedSegmentId == null) return null;
  return state.doc.segments.find((s) => s.segment_id === state.selectedSegmentId) || null;
}

function selectSegment(segmentId, seekPlayer = false) {
  state.selectedSegmentId = segmentId;
  const seg = getSelectedSegment();
  if (seekPlayer && seg) {
    seekVideoMs(seg.start_ms + 30);
  }
  renderMultitrackTimeline();
  renderDiffTable();
  renderInspector();
  updateVideoOverlays();
}

function seekVideoMs(ms) {
  if (el.video.readyState >= 1) {
    el.video.currentTime = Math.max(0, ms / 1000);
  }
  updateVideoOverlays();
}

function renderInspector() {
  const seg = getSelectedSegment();
  if (!seg) {
    el.spkRenameGroup.hidden = true;
    el.inspectorSub.textContent = "대사를 선택하세요.";
    el.inspectorBody.innerHTML = `<div class="empty-hint">선택된 세그먼트가 없습니다.</div>`;
    return;
  }

  el.spkRenameGroup.hidden = false;
  el.inpSpeakerId.value = seg.speaker_id;
  el.inspectorHeading.textContent = `세그먼트 #${seg.segment_id} 상세 인스펙터`;
  el.inspectorSub.textContent = `${fmtMs(seg.start_ms)} ~ ${fmtMs(seg.end_ms)} · 원음 STT: "${seg.raw_transcript}"`;

  const gates = seg.emotion_irony?.gate_weights_mean || { audio: 0.5, lexical: 0.5, visual: 0.0 };
  const ga = Math.round((gates.audio || 0) * 100);
  const gl = Math.round((gates.lexical || 0) * 100);
  const gv = Math.max(0, 100 - ga - gl);

  const wordChips = (seg.words || [])
    .map((w) => {
      const hi = w.emphasis_level >= 3 ? "lv-high" : "";
      const tok = w.emphasis_token ? ` ${w.emphasis_token}` : "";
      return `<span class="word-chip ${hi}" title="F0=${w.pitch_hz.toFixed(0)}Hz, 에너지=${w.energy_db.toFixed(1)}dB, Z=${w.z_emphasis.toFixed(2)}">${esc(w.text)}${esc(tok)}</span>`;
    })
    .join("");

  const candCards = (seg.candidates || [])
    .map((c) => {
      const label = STRATEGY_LABELS[c.strategy] || c.strategy;
      const selCls = c.selected ? "is-selected" : "";
      const guardCls = c.passed_guard ? "" : "failed-guard";
      return `
        <div class="cand-card ${selCls} ${guardCls}">
          <div class="cand-head">
            <span class="cand-strategy">#${c.candidate_id} ${esc(label)}</span>
            <div>
              ${!c.passed_guard ? `<span class="badge badge-err">의미가드 미달</span>` : ""}
              <button type="button" class="btn btn-xs ${c.selected ? "btn-primary" : "btn-accent"} btn-pick-cand" data-cand-id="${c.candidate_id}">
                ${c.selected ? "✓ 현재 채택됨" : "이 후보 채택"}
              </button>
            </div>
          </div>
          <div class="cand-text">${esc(c.srt_text)}</div>
          <div class="cand-metrics">
            <div>
              <span>의미보존 C_i: ${(c.semantic_ci * 100).toFixed(1)}%</span>
              <div class="mini-meter"><span style="width:${(c.semantic_ci * 100).toFixed(0)}%;background:#34d399"></span></div>
            </div>
            <div>
              <span>화법적합 S_MoS: ${(c.mos_fidelity * 100).toFixed(1)}%</span>
              <div class="mini-meter"><span style="width:${(c.mos_fidelity * 100).toFixed(0)}%;background:#fbbf24"></span></div>
            </div>
            <div>
              <span><strong>종합 조화 O_i: ${(c.harmonic_oi * 100).toFixed(1)}%</strong></span>
              <div class="mini-meter"><span style="width:${(c.harmonic_oi * 100).toFixed(0)}%;background:#38bdf8"></span></div>
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  el.inspectorBody.innerHTML = `
    <div class="evidence-grid">
      <div class="ev-card">
        <div class="ev-title">Trimodal CmGI 모달리티 게이트 비중</div>
        <div class="gate-stack">
          <div class="gate-a" style="width:${ga}%" title="음향 ${ga}%"></div>
          <div class="gate-l" style="width:${gl}%" title="어휘 ${gl}%"></div>
          <div class="gate-v" style="width:${gv}%" title="시각 ${gv}%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted)">
          <span>음향 ${ga}%</span><span>어휘 ${gl}%</span><span>시각 ${gv}%</span>
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-title">반어 괴리도 (Gao et al. 응용)</div>
        <div>어휘 극성 <code>${seg.emotion_irony.lexical_polarity.toFixed(2)}</code> vs 시청각 <code>${seg.emotion_irony.audiovisual_polarity.toFixed(2)}</code></div>
        <div style="margin-top:2px">
          <strong>I_irony = ${seg.emotion_irony.irony_score.toFixed(2)}</strong>
          (${seg.emotion_irony.irony_detected ? `<span style="color:#fbbf24">반어법 확정</span>` : "일치/중립"})
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-title">한국어 3축 경어 · 6화계 문맥 진단</div>
        <div>판정: <strong>${esc(seg.honorific.speech_level)}</strong> (<code>${esc(seg.honorific.consistency_status)}</code>)</div>
        <div style="color:var(--text-muted);font-size:10px">
          주체높임: ${seg.honorific.subject_honorific ? "예(-시-)" : "아니오"} · 기대화계: ${esc(seg.honorific.expected_speech_level || "-")}
          ${seg.honorific.suggested_correction ? `<br/>CAPE 교정: <strong style="color:#34d399">${esc(seg.honorific.suggested_correction)}</strong>` : ""}
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-title">단어별 5단계 음향 강조 (&lt;toe_k&gt; · 마우스 오버 시 F0/dB)</div>
        <div class="word-chips">${wordChips || "<span style='color:var(--text-dim)'>단어 정렬 정보 없음</span>"}</div>
      </div>
    </div>

    <div style="font-size:11.5px;font-weight:700;color:#e2e8f0;margin-top:2px">
      AI 생성 배리어프리 후보 (${seg.candidates?.length || 0}선 · de Rivero 조화평균 O_i 순)
    </div>
    <div class="cand-list">${candCards || "<div class='empty-hint'>생성된 후보가 없습니다.</div>"}</div>

    <div class="manual-edit-box">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <strong style="font-size:11px;color:#e2e8f0">수동 미세 편집 (저장 시 .SRT / .VTT / .ASS 즉시 재출력)</strong>
        <button type="button" id="btn-save-manual" class="btn btn-xs btn-primary">수동 수정 저장</button>
      </div>
      <label>
        <span>표준 지문형 (.SRT / .VTT)</span>
        <input type="text" id="inp-manual-srt" class="studio-input" value="${esc(seg.sdh_srt_text)}" />
      </label>
      <label>
        <span>시각 스타일형 (.ASS 오버라이드 태그 포함)</span>
        <input type="text" id="inp-manual-ass" class="studio-input" value="${esc(seg.sdh_ass_text)}" />
      </label>
    </div>
  `;

  el.inspectorBody.querySelectorAll(".btn-pick-cand").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cid = Number(btn.getAttribute("data-cand-id"));
      await applyCandidateSelection(seg.segment_id, cid);
    });
  });

  document.getElementById("btn-save-manual")?.addEventListener("click", async () => {
    const srtVal = document.getElementById("inp-manual-srt").value;
    const assVal = document.getElementById("inp-manual-ass").value;
    await saveManualSegmentEdit(seg.segment_id, { sdh_srt_text: srtVal, sdh_ass_text: assVal });
  });
}

async function applyCandidateSelection(segmentId, candidateId) {
  const res = await fetch(
    `/api/v1/projects/${encodeURIComponent(state.projectId)}/segments/${segmentId}/select-candidate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: candidateId }),
    }
  );
  if (!res.ok) {
    showToast("후보 교체 중 오류가 발생했습니다.");
    return;
  }
  const data = await res.json();
  syncUpdatedSegment(data);
  showToast(`세그먼트 #${segmentId} → 후보 #${candidateId} 채택 및 자막 파일 갱신 완료!`);
}

async function saveManualSegmentEdit(segmentId, payload) {
  const res = await fetch(
    `/api/v1/projects/${encodeURIComponent(state.projectId)}/segments/${segmentId}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!res.ok) {
    showToast("수동 편집 저장 실패");
    return;
  }
  const data = await res.json();
  syncUpdatedSegment(data);
  showToast(`세그먼트 #${segmentId} 수동 편집이 .SRT/.VTT/.ASS에 반영되었습니다.`);
}

function syncUpdatedSegment(resp) {
  const idx = state.doc.segments.findIndex((s) => s.segment_id === resp.segment.segment_id);
  if (idx >= 0) {
    state.doc.segments[idx] = resp.segment;
  }
  state.doc.metrics.mean_semantic_ci = resp.mean_semantic_ci;
  state.doc.metrics.mean_harmonic_oi = resp.mean_harmonic_oi;
  state.doc.metrics.mean_cps = resp.mean_cps;
  state.doc.metrics.cps_compliance_rate = resp.cps_compliance_rate;
  state.doc.metrics.cue_attachment_ratio = resp.cue_attachment_ratio;
  renderAll();
}

// ============================================================================
// Event Bindings, Modals & Keyboard Shortcuts
// ============================================================================
function bindEvents() {
  el.projectSelect.addEventListener("change", () => {
    if (el.projectSelect.value) loadProject(el.projectSelect.value);
  });

  // Export download buttons
  document.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.projectId) {
        showToast("먼저 프로젝트를 로드해 주세요.");
        return;
      }
      const fmt = btn.getAttribute("data-export");
      window.location.href = `/api/v1/projects/${encodeURIComponent(state.projectId)}/export/${fmt}`;
    });
  });

  // Subtitle preview mode toggle
  document.querySelectorAll("[data-sub-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-sub-mode]").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.subtitleMode = btn.getAttribute("data-sub-mode");
      updateVideoOverlays();
    });
  });

  el.chkDebug.addEventListener("change", () => {
    state.showDebugOverlay = el.chkDebug.checked;
    updateVideoOverlays();
  });

  // Video playback sync
  el.video.addEventListener("timeupdate", updateVideoOverlays);
  el.video.addEventListener("loadedmetadata", updateVideoOverlays);
  window.addEventListener("resize", updateVideoOverlays);

  // Quadrant 2 sub-tabs
  document.querySelectorAll("[data-macro-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-macro-tab]").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");
      const tab = btn.getAttribute("data-macro-tab");
      el.paneRelations.hidden = tab !== "relations";
      el.paneQuality.hidden = tab !== "quality";
    });
  });

  el.btnClearSpkFilter.addEventListener("click", () => {
    state.speakerFilter = null;
    el.btnClearSpkFilter.hidden = true;
    renderRelationGraphAndScenes();
    renderDiffTable();
  });

  // Over-stylization Pruner controls
  el.rngCueCap.addEventListener("input", () => {
    el.lblCueCap.textContent = `${el.rngCueCap.value}%`;
  });
  el.rngCpsCap.addEventListener("input", () => {
    el.lblCpsCap.textContent = `${el.rngCpsCap.value}자/초`;
  });
  el.btnApplyPrune.addEventListener("click", async () => {
    if (!state.projectId) return;
    const res = await fetch(`/api/v1/projects/${encodeURIComponent(state.projectId)}/prune`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_cue_density_ratio: Number(el.rngCueCap.value) / 100,
        recommended_max_cps: Number(el.rngCpsCap.value),
      }),
    });
    if (res.ok) {
      state.doc = await res.json();
      renderAll();
      showToast("과잉 지문 밀도 및 CPS 억제 기준이 전체 세그먼트에 재적용되었습니다.");
    }
  });

  // Timeline click-to-scrub
  el.tracksCanvas.addEventListener("click", (ev) => {
    if (!state.doc) return;
    const rect = el.tracksCanvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
    const totalMs = Math.max(1000, state.doc.duration_ms || 0);
    const targetMs = Math.round(ratio * totalMs);
    seekVideoMs(targetMs);
    const hit = findActiveSegmentAt(targetMs);
    if (hit) selectSegment(hit.segment_id, false);
  });

  // Filter pills & search
  el.filterPills.querySelectorAll(".pill-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      el.filterPills.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.activeFilter = btn.getAttribute("data-filter");
      renderDiffTable();
    });
  });

  el.searchInput.addEventListener("input", () => {
    state.searchQuery = el.searchInput.value.trim();
    renderDiffTable();
  });

  // Speaker rename
  el.btnSaveSpeaker.addEventListener("click", async () => {
    const seg = getSelectedSegment();
    const newSpk = el.inpSpeakerId.value.trim();
    if (!seg || !newSpk) return;
    await saveManualSegmentEdit(seg.segment_id, { speaker_id: newSpk });
  });

  // Native <dialog> modals + backdrop light dismiss
  el.btnOpenAnalyze.addEventListener("click", () => el.analyzeDialog.showModal());
  el.btnOpenSwer.addEventListener("click", () => {
    el.swerDialog.showModal();
    runSwerCalc();
  });

  document.querySelectorAll("[data-close-dialog]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById(btn.getAttribute("data-close-dialog"))?.close();
    });
  });

  for (const dlg of [el.analyzeDialog, el.swerDialog]) {
    dlg.addEventListener("click", (ev) => {
      if (ev.target === dlg) dlg.close();
    });
  }

  // Analyze Dialog input mode toggle & submit
  document.querySelectorAll("[data-input-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-input-tab]").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.inputTab = btn.getAttribute("data-input-tab");
      el.paneInputUpload.hidden = state.inputTab !== "upload";
      el.paneInputPath.hidden = state.inputTab !== "path";
    });
  });

  el.analyzeForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await submitNewAnalysis();
  });

  el.btnCalcSwer.addEventListener("click", runSwerCalc);

  // Keyboard navigation
  document.addEventListener("keydown", (ev) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (el.analyzeDialog.open || el.swerDialog.open) return;

    if (ev.code === "Space") {
      ev.preventDefault();
      if (el.video.paused) el.video.play().catch(() => {});
      else el.video.pause();
    } else if (ev.code === "ArrowDown" || ev.code === "ArrowUp") {
      ev.preventDefault();
      if (!state.doc?.segments?.length) return;
      const idx = state.doc.segments.findIndex((s) => s.segment_id === state.selectedSegmentId);
      const nextIdx =
        ev.code === "ArrowDown"
          ? Math.min(state.doc.segments.length - 1, idx + 1)
          : Math.max(0, idx - 1);
      selectSegment(state.doc.segments[nextIdx].segment_id, true);
    }
  });
}

async function submitNewAnalysis() {
  const sep = document.getElementById("opt-demucs").checked;
  const vis = document.getElementById("opt-visual").checked;
  const vlm = document.getElementById("opt-vlm").checked;
  const llm = document.getElementById("opt-llm").checked;

  el.analyzeProgress.hidden = false;
  el.btnSubmitAnalyze.disabled = true;

  try {
    let res;
    if (state.inputTab === "upload") {
      const mediaFile = document.getElementById("file-media").files[0];
      const subFile = document.getElementById("file-subtitle").files[0];
      if (!mediaFile) {
        showToast("분석할 영상 또는 오디오 파일을 선택해 주세요.");
        return;
      }
      const fd = new FormData();
      fd.append("media_file", mediaFile);
      if (subFile) fd.append("subtitle_file", subFile);
      fd.append("enable_source_separation", String(sep));
      fd.append("enable_visual_stage", String(vis));
      fd.append("enable_vlm_stage", String(vlm));
      fd.append("enable_llm_stage", String(llm));
      res = await fetch("/api/v1/analyze-upload", { method: "POST", body: fd });
    } else {
      const mediaPath = document.getElementById("inp-media-path").value.trim();
      const subPath = document.getElementById("inp-sub-path").value.trim();
      if (!mediaPath) {
        showToast("로컬 영상 파일 경로를 입력해 주세요.");
        return;
      }
      res = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_media_path: mediaPath,
          input_subtitle_path: subPath || null,
          enable_source_separation: sep,
          enable_visual_stage: vis,
          enable_vlm_stage: vlm,
          enable_llm_stage: llm,
        }),
      });
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast(`분석 실패: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    el.analyzeDialog.close();
    await refreshProjectList(data.project_id);
    await loadProject(data.project_id);
    showToast("5-Stage 화법 분석 및 배리어프리 프로토타입 생성 완료!");
  } finally {
    el.analyzeProgress.hidden = true;
    el.btnSubmitAnalyze.disabled = false;
  }
}

async function runSwerCalc() {
  const refLines = document.getElementById("swer-ref").value.split("\n").filter((x) => x.trim());
  const hypLines = document.getElementById("swer-hyp").value.split("\n").filter((x) => x.trim());
  if (!refLines.length || !hypLines.length) return;

  const res = await fetch("/api/v1/eval-swer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref_lines: refLines, hyp_lines: hypLines }),
  });
  if (!res.ok) return;
  const d = await res.json();
  el.swerResultBox.innerHTML = `
    <div>비교 문장 수: <strong>${d.compared_utterances}문장</strong></div>
    <div>음절 오류율 (CER): <strong>${d.cer_percent.toFixed(2)}%</strong></div>
    <div>일반 어절 오류율 (WER): <span style="color:#f43f5e">${d.wer_percent.toFixed(2)}%</span></div>
    <div>공백 정규화 어절 오류율 (<strong>sWER</strong>): <strong style="color:#34d399">${d.swer_percent.toFixed(2)}%</strong></div>
  `;
}

initStudio();
