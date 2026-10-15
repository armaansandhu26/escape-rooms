const DEFAULT_PRESET = "ideal_cooperation";
const PRESET_BY_MODE = {
  collaborative: "ideal_cooperation",
  competitive: "shared_competitive",
};

const WING_UI = {
  A: {
    panel: document.querySelector("#wing-panel-a"),
    stage: document.querySelector("#wing-a-stage"),
    nodes: document.querySelector("#wing-a-nodes"),
    avatar: document.querySelector("#wing-a-avatar"),
    title: document.querySelector("#wing-a-title"),
  },
  B: {
    panel: document.querySelector("#wing-panel-b"),
    stage: document.querySelector("#wing-b-stage"),
    nodes: document.querySelector("#wing-b-nodes"),
    avatar: document.querySelector("#wing-b-avatar"),
    title: document.querySelector("#wing-b-title"),
  },
};

const elements = {
  roundBanner: document.querySelector("#round-banner"),
  captionPrimary: document.querySelector("#caption-primary"),
  captionSecondary: document.querySelector("#caption-secondary"),
  resetButton: document.querySelector("#reset-button"),
  startButton: document.querySelector("#start-button"),
  toast: document.querySelector("#toast"),
  speedButtons: document.querySelectorAll(".speed-btn"),
  modeToggle: document.querySelector("#mode-toggle"),
  modeButtons: document.querySelectorAll(".mode-btn"),
  trajectoryLog: document.querySelector("#trajectory-log"),
};

let gameState = null;
let busy = false;
let cinematicPeekAvailable = true;
let warnedAboutServerRestart = false;
let speedMultiplier = 1;
let sessionToken = null;

const BASE_SIM_TIMING = {
  moveMs: 1400,
  beforeMs: 800,
  afterMs: 1300,
  betweenMs: 450,
};
let toastTimer = null;
const selectedNodeId = { A: null, B: null };
const agentPositions = { A: null, B: null };
let trajectorySyncedLength = 0;

const SPATIAL_GLYPHS = { clue: "?", spot: "◆", container: "▣", door: "EXIT" };

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  let body = options.body;
  if (
    path.startsWith("/api/") &&
    path !== "/api/catalog" &&
    (options.method || "GET") === "POST"
  ) {
    const bodyObj = body ? JSON.parse(body) : {};
    if (sessionToken) bodyObj.session = sessionToken;
    body = JSON.stringify(bodyObj);
  }
  const response = await fetch(path, { ...options, headers, body });
  const raw = await response.text();
  let payload = null;
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    if (response.status === 404) {
      throw new Error(
        "Server is out of date. Stop it (Ctrl+C) and restart: .venv/bin/python scripts/play_web.py",
      );
    }
    throw new Error(`Unexpected server response (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  if (payload.session) sessionToken = payload.session;
  return payload;
}

function isSingleAgent() {
  return Boolean(gameState?.room?.single_agent);
}

function isSharedRoom() {
  return gameState?.room?.layout === "shared";
}

function activeWingAgents() {
  return isSingleAgent() ? ["A"] : ["A", "B"];
}

function uiForAgent(agent) {
  return isSharedRoom() ? WING_UI.A : WING_UI[agent];
}

async function loadCatalogFlags() {
  const catalog = await request("/api/catalog");
  cinematicPeekAvailable = catalog.features?.peek_auto_step !== false;
}

function currentMode() {
  return gameState?.simulation?.mode || "collaborative";
}

function syncModeToggleUi() {
  const mode = currentMode();
  const show = isSharedRoom() && !isSingleAgent();
  elements.modeToggle?.classList.toggle("hidden", !show);
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
}

async function configurePreset(presetKey) {
  const payload = await request("/api/configure", {
    method: "POST",
    body: JSON.stringify({ preset: presetKey }),
  });
  gameState = payload.state;
  selectedNodeId.A = null;
  selectedNodeId.B = null;
  agentPositions.A = null;
  agentPositions.B = null;
  syncModeToggleUi();
}

async function configureDefaultEpisode() {
  await configurePreset(DEFAULT_PRESET);
}

async function setIncentiveMode(mode) {
  if (busy || mode === currentMode()) return;
  const preset = PRESET_BY_MODE[mode];
  if (!preset) return;
  try {
    busy = true;
    await configurePreset(preset);
    resetTrajectoryLog();
    elements.roundBanner.classList.add("hidden");
    setCaption(
      mode === "competitive"
        ? "Competitive replay — first agent through the exit wins."
        : "Cooperative replay — both agents escape together.",
      "Press Start to watch the scripted trajectory.",
    );
    render();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    busy = false;
  }
}

async function loadState() {
  gameState = await request("/api/state");
  render();
}

function getSimTiming() {
  const multiplier = speedMultiplier > 0 ? speedMultiplier : 1;
  return {
    moveMs: Math.round(BASE_SIM_TIMING.moveMs / multiplier),
    beforeMs: Math.round(BASE_SIM_TIMING.beforeMs / multiplier),
    afterMs: Math.round(BASE_SIM_TIMING.afterMs / multiplier),
    betweenMs: Math.round(BASE_SIM_TIMING.betweenMs / multiplier),
  };
}

function applyPlaybackSpeed() {
  const timing = getSimTiming();
  document.documentElement.style.setProperty("--sim-move-ms", `${timing.moveMs}ms`);
}

function setSpeed(multiplier) {
  speedMultiplier = multiplier;
  elements.speedButtons.forEach((button) => {
    button.classList.toggle("is-active", Number(button.dataset.speed) === multiplier);
  });
  applyPlaybackSpeed();
}

function agentLabel(agent) {
  return isSingleAgent() ? "You" : `Agent ${agent}`;
}

function labelId(id) {
  return String(id).replace(/_/g, " ");
}

function humanizeAction(action) {
  const parts = parseActionParts(action);
  switch (parts.kind) {
    case "say":
      return `says “${parts.text}”`;
    case "inspect":
      return `reads ${labelId(parts.nodeId)}`;
    case "take":
      return `takes ${labelId(parts.itemId)}`;
    case "enter":
      return `opens ${labelId(parts.nodeId)}`;
    case "use":
      return `uses a key on ${labelId(parts.nodeId)}`;
    case "give":
      return "passes an item through the slot";
    case "wait":
      return "waits";
    default:
      return action.replace(/_/g, " ");
  }
}

function setCaption(primary, secondary = "") {
  elements.captionPrimary.textContent = primary;
  elements.captionSecondary.textContent = secondary;
  elements.captionSecondary.classList.toggle("is-visible", Boolean(secondary));
}

function resetTrajectoryLog() {
  trajectorySyncedLength = 0;
  elements.trajectoryLog.replaceChildren();
}

const TRAJECTORY_TAG_LABELS = {
  explore: "Explore",
  unlock: "Unlock",
  collect: "Collect",
  share: "Share",
  idle: "Idle",
  fail: "Failed",
};

function groupEpisodeLogByStep(log) {
  const order = [];
  const groups = new Map();
  for (const entry of log) {
    const step = entry.env_step ?? 0;
    if (!groups.has(step)) {
      groups.set(step, []);
      order.push(step);
    }
    groups.get(step).push(entry);
  }
  return order.map((env_step) => ({ env_step, entries: groups.get(env_step) }));
}

function buildTrajectoryAgentLine(entry) {
  const line = createElement("div", `trajectory-agent-line agent-${entry.agent.toLowerCase()}`);
  if (!entry.valid) line.classList.add("is-invalid");
  if (entry.tag) line.dataset.tag = entry.tag;

  const header = createElement("div", "trajectory-header");
  header.append(createElement("span", "trajectory-agent", `Agent ${entry.agent}`));
  if (entry.tag && TRAJECTORY_TAG_LABELS[entry.tag]) {
    header.append(
      createElement("span", `trajectory-tag tag-${entry.tag}`, TRAJECTORY_TAG_LABELS[entry.tag]),
    );
  }
  line.append(header);

  const narrative =
    entry.narrative ||
    `${humanizeAction(entry.action)}${entry.result ? ` → ${entry.result}` : ""}`;
  line.append(createElement("p", "trajectory-narrative", narrative));

  if (entry.action && entry.narrative) {
    line.append(createElement("span", "trajectory-action-raw", entry.action));
  } else if (!entry.narrative && entry.result) {
    line.append(createElement("span", "trajectory-result", entry.result));
  }
  return line;
}

function buildTrajectoryStepGroup({ env_step, entries }) {
  const item = createElement("li", "trajectory-step-group");
  item.append(createElement("div", "trajectory-step-title", `Step ${env_step}:`));
  const body = createElement("div", "trajectory-step-body");
  for (const entry of entries) {
    body.append(buildTrajectoryAgentLine(entry));
  }
  item.append(body);
  return item;
}

function renderTrajectoryLog(log) {
  resetTrajectoryLog();
  const groups = groupEpisodeLogByStep(log);
  for (const group of groups) {
    elements.trajectoryLog.append(buildTrajectoryStepGroup(group));
  }
  const stepGroups = elements.trajectoryLog.querySelectorAll(".trajectory-step-group");
  stepGroups.forEach((node, index) => {
    node.classList.toggle("is-latest", index === stepGroups.length - 1);
  });
  const latest = stepGroups[stepGroups.length - 1];
  if (latest) {
    latest.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  trajectorySyncedLength = log.length;
}

function syncTrajectoryLog() {
  const log = gameState?.episode_log || [];
  if (log.length === trajectorySyncedLength) return;
  renderTrajectoryLog(log);
}

function parseActionParts(action) {
  const sayMatch = action.match(/^say\((.*)\)$/);
  if (sayMatch) return { kind: "say", text: sayMatch[1] };
  const inspectMatch = action.match(/^inspect\(([^)]+)\)$/);
  if (inspectMatch) return { kind: "inspect", nodeId: inspectMatch[1] };
  const takeMatch = action.match(/^take\(([^)]+)\)$/);
  if (takeMatch) return { kind: "take", itemId: takeMatch[1] };
  const enterMatch = action.match(/^enter\([^,]+,\s*([^)]+)\)$/);
  if (enterMatch) return { kind: "enter", nodeId: enterMatch[1] };
  const useMatch = action.match(/^use\([^,]+,\s*([^)]+)\)$/);
  if (useMatch) return { kind: "use", nodeId: useMatch[1] };
  if (action.startsWith("give(")) return { kind: "give" };
  if (action.startsWith("wait(")) return { kind: "wait" };
  return { kind: "other" };
}

function findNodeForItem(agent, itemId) {
  const nodes = gameState.wings[agent]?.nodes || [];
  return nodes.find((node) =>
    (node.available_items || []).some((item) => item.id === itemId),
  );
}

async function animateAgentForAction(agent, action) {
  const timing = getSimTiming();
  const ui = uiForAgent(agent);
  const parts = parseActionParts(action);
  let nodeId = null;

  if (parts.kind === "inspect" || parts.kind === "enter" || parts.kind === "use") {
    nodeId = parts.nodeId;
  } else if (parts.kind === "take") {
    const node = findNodeForItem(agent, parts.itemId);
    nodeId = node?.id ?? null;
  } else if (parts.kind === "give") {
    const mat = gameState.wings[agent]?.nodes.find((node) => node.id === "mat");
    nodeId = mat?.id ?? null;
  } else if (parts.kind === "say") {
    moveAvatar(WING_UI[agent].avatar, agent === "A" ? 42 : 58, 72, true);
    await delay(Math.round(timing.moveMs * 0.4));
    return;
  } else if (parts.kind === "wait") {
    moveAvatar(ui.avatar, 50, 82, true);
    await delay(timing.moveMs * 0.6);
    return;
  }

  if (nodeId) {
    await animateAgentToNode(agent, nodeId, timing.moveMs);
    return;
  }
  moveAvatar(ui.avatar, 50, 82, true);
  await delay(timing.moveMs * 0.5);
}

function clearActionTargets() {
  document.querySelectorAll(".spatial-object.action-target").forEach((object) => {
    object.classList.remove("action-target");
  });
}

async function animateAgentToNode(agent, nodeId, durationMs) {
  const view = gameState.wings[agent];
  if (!view) return;
  const node = view.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) return;

  selectedNodeId[agent] = nodeId;
  const target = {
    x: node.position.x,
    y: node.position.y > 78 ? node.position.y - 10 : node.position.y + 10,
  };
  agentPositions[agent] = target;
  syncWingMap(agent, view, { skipAvatarMove: true });

  const avatar = WING_UI[agent].avatar;
  avatar.classList.add("acting");
  moveAvatar(avatar, target.x, target.y, true);
  highlightActionTarget(agent, nodeId);
  await delay(durationMs);
  avatar.classList.remove("acting");
}

function highlightActionTarget(agent, nodeId) {
  const nodesRoot = isSharedRoom() ? WING_UI.A.nodes : WING_UI[agent].nodes;
  nodesRoot.querySelectorAll(".spatial-object").forEach((object) => {
    object.classList.toggle("action-target", Boolean(nodeId && object.dataset.nodeId === nodeId));
  });
}

async function peekUpcomingTurn() {
  if (!cinematicPeekAvailable) return null;
  try {
    const payload = await request("/api/peek-auto-step", {
      method: "POST",
      body: "{}",
    });
    return payload;
  } catch {
    cinematicPeekAvailable = false;
    if (!warnedAboutServerRestart) {
      warnedAboutServerRestart = true;
      showToast("Restart play_web.py for smoother movement.", true);
    }
    return null;
  }
}

async function executeAutoStep() {
  return request("/api/auto-step", {
    method: "POST",
    body: "{}",
  });
}

function capitalize(text) {
  if (!text) return text;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function stepActionsFromPeek(peekPayload) {
  if (peekPayload?.actions) return peekPayload.actions;
  if (peekPayload?.agent && peekPayload?.action) {
    return { [peekPayload.agent]: peekPayload.action };
  }
  return null;
}

function jointStepPreviewLines(actions) {
  return Object.entries(actions)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(
      ([agent, action]) =>
        `${capitalize(agentLabel(agent))} ${humanizeAction(action)}`,
    )
    .join(" · ");
}

async function animateJointActions(actions) {
  const timing = getSimTiming();
  const agents = Object.keys(actions).sort();
  highlightActiveAgents(agents);
  clearActionTargets();
  await Promise.all(
    agents.map((agent) => animateAgentForAction(agent, actions[agent])),
  );
  clearActionTargets();
  highlightActiveAgents(agents);
  await delay(Math.round(timing.betweenMs * 0.35));
}

async function playCinematicStep() {
  applyPlaybackSpeed();
  const timing = getSimTiming();
  const upcoming = await peekUpcomingTurn();
  const upcomingActions = stepActionsFromPeek(upcoming);

  if (upcomingActions) {
    const stepNum = upcoming?.env_step ?? (gameState?.env_step ?? 0) + 1;
    setCaption(
      `Step ${stepNum} — both agents act together`,
      jointStepPreviewLines(upcomingActions),
    );
    await animateJointActions(upcomingActions);
  } else {
    await delay(timing.beforeMs);
  }

  const payload = await executeAutoStep();
  gameState = payload.state;
  const stepAgents = payload.step?.agents || [];
  if (!upcomingActions && stepAgents.length) {
    const preview = Object.fromEntries(
      stepAgents.map((record) => [record.agent, record.action]),
    );
    await animateJointActions(preview);
  }

  renderAfterSimStep();
  highlightActiveAgents(gameState?.active_agents);
  if (stepAgents.length) {
    const lines = stepAgents.map(
      (record) =>
        `${capitalize(agentLabel(record.agent))}: ${record.narrative || humanizeAction(record.action)}`,
    );
    setCaption(
      `Step ${payload.env_step || gameState.env_step} complete`,
      lines.join(" · "),
    );
  }
  await delay(timing.afterMs);
}

function turnsLeftLabel() {
  const view = gameState?.view;
  if (!view) return "";
  return `${view.turns_left} turns left`;
}

function renderAfterSimStep() {
  if (!gameState) return;
  document.body.classList.toggle("solo-mode", isSingleAgent());
  document.body.classList.toggle("shared-mode", isSharedRoom());

  if (isSharedRoom()) {
    syncSharedMap();
  } else {
    for (const agent of activeWingAgents()) {
      const wing = gameState.wings[agent];
      if (wing) syncWingMap(agent, wing);
    }
  }

  syncTrajectoryLog();

  if (gameState.round_over) {
    renderRoundOver();
  }
}

async function runEpisodeAnimated() {
  if (busy) return;
  busy = true;
  document.body.classList.add("simulation-running");
  setControlsDisabled(true);
  try {
    resetTrajectoryLog();
    await resetGame(false);
    highlightActiveAgents(gameState?.active_agents);
    setCaption("Episode started — one step, both agents move together.", turnsLeftLabel());
    const timing = getSimTiming();
    while (!gameState.round_over && gameState.scripted_turn) {
      await playCinematicStep();
      if (!gameState.round_over && gameState.scripted_turn) {
        await delay(timing.betweenMs);
      }
    }
    if (gameState.round_over) {
      renderRoundOver();
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    document.body.classList.remove("simulation-running");
    clearActionTargets();
    busy = false;
    setControlsDisabled(false);
  }
}

async function resetGame(showMessage = true) {
  const wasBusy = busy;
  busy = true;
  try {
    const payload = await request("/api/reset", {
      method: "POST",
      body: "{}",
    });
    gameState = payload.state;
    selectedNodeId.A = null;
    selectedNodeId.B = null;
    agentPositions.A = null;
    agentPositions.B = null;
    resetTrajectoryLog();
    elements.roundBanner.classList.add("hidden");
    if (showMessage) {
      setCaption("Room reset. Press Start when you are ready.", "");
    }
    render();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    busy = wasBusy ? true : false;
  }
}

function render() {
  if (!gameState) return;
  document.body.classList.toggle("solo-mode", isSingleAgent());
  document.body.classList.toggle("shared-mode", isSharedRoom());
  syncModeToggleUi();

  if (isSharedRoom()) {
    syncSharedMap();
  } else {
    for (const agent of activeWingAgents()) {
      const wing = gameState.wings[agent];
      if (wing) syncWingMap(agent, wing);
    }
  }

  if (gameState.round_over) {
    renderRoundOver();
    return;
  }

  highlightActiveAgents(gameState.active_agents);

  syncTrajectoryLog();

  if (!document.body.classList.contains("simulation-running")) {
    setCaption(
      "Press Start — each step, both agents act at the same time.",
      gameState.scripted_step || gameState.scripted_turn ? turnsLeftLabel() : "",
    );
  }
  setControlsDisabled(false);
}

function highlightActiveAgents(agents) {
  const list = Array.isArray(agents)
    ? agents
    : gameState?.active_agents?.length
      ? gameState.active_agents
      : gameState?.active_agent
        ? [gameState.active_agent]
        : [];

  if (isSharedRoom()) {
    WING_UI.A.panel?.classList.add("active-wing");
    WING_UI.A.avatar?.classList.toggle("active-agent", list.includes("A"));
    WING_UI.B.avatar?.classList.toggle("active-agent", list.includes("B"));
    return;
  }

  for (const wingAgent of ["A", "B"]) {
    const on = list.includes(wingAgent);
    WING_UI[wingAgent].panel?.classList.toggle("active-wing", on);
    WING_UI[wingAgent].avatar?.classList.toggle("active-agent", on);
  }
}

function syncSharedMap() {
  const view = gameState.wings.A || gameState.wings.B;
  if (!view) return;
  const ui = WING_UI.A;
  ui.title.textContent = gameState.room?.space || view.wing;
  const roomTag = ui.panel?.querySelector(".room-tag");
  if (roomTag) roomTag.textContent = gameState.room?.space || view.wing;

  const stage = ui.stage;
  if (stage && WING_UI.B.avatar.parentElement !== stage) {
    stage.appendChild(WING_UI.B.avatar);
  }

  syncWingMap("A", view, {
    skipAvatarMove: true,
    forceActive: gameState.active_agent === "A",
  });

  if (!agentPositions.A) agentPositions.A = { x: 38, y: 88 };
  if (!agentPositions.B) agentPositions.B = { x: 62, y: 88 };
  const animateMove = document.body.classList.contains("simulation-running");
  moveAvatar(WING_UI.A.avatar, agentPositions.A.x, agentPositions.A.y, animateMove);
  moveAvatar(WING_UI.B.avatar, agentPositions.B.x, agentPositions.B.y, animateMove);
  WING_UI.A.avatar.classList.remove("hidden");
  WING_UI.B.avatar.classList.remove("hidden");
  highlightActiveAgents(gameState.active_agents);
}

function ensureSpatialObject(agent, node, isActive) {
  const ui = uiForAgent(agent);
  let object = ui.nodes.querySelector(`[data-node-id="${CSS.escape(node.id)}"]`);
  if (!object) {
    object = createElement("button", "spatial-object");
    object.type = "button";
    object.dataset.kind = node.kind;
    object.dataset.nodeId = node.id;
    object.disabled = true;
    object.append(
      createElement("span", "object-glyph", SPATIAL_GLYPHS[node.kind] || "•"),
      createElement("span", "object-label", labelId(node.id)),
    );
    ui.nodes.append(object);
  }
  object.style.left = `${node.position.x}%`;
  object.style.top = `${node.position.y}%`;
  object.querySelector(".object-label").textContent = labelId(node.id);
  object.disabled = !isActive || busy;
  object.classList.toggle("selected", node.id === selectedNodeId[agent]);
  object.classList.toggle("opened", Boolean(node.opened));
  return object;
}

function syncWingMap(agent, view, options = {}) {
  const { skipAvatarMove = false, forceActive } = options;
  if (isSharedRoom() && agent !== "A") {
    return;
  }
  const ui = isSharedRoom() ? WING_UI.A : WING_UI[agent];
  const avatar = WING_UI[agent].avatar;
  if (!isSharedRoom()) {
    ui.title.textContent = view.wing;
  }
  avatar.classList.remove("hidden");

  if (!skipAvatarMove) {
    const position = agentPositions[agent] || { x: 50, y: 88 };
    const animateMove = document.body.classList.contains("simulation-running");
    moveAvatar(avatar, position.x, position.y, animateMove);
  }

  const isActive =
    forceActive !== undefined ? forceActive : agent === gameState.active_agent;
  const seen = new Set();
  for (const node of view.nodes) {
    seen.add(node.id);
    ensureSpatialObject(agent, node, isActive);
  }
  for (const child of [...ui.nodes.children]) {
    if (!seen.has(child.dataset.nodeId)) child.remove();
  }
}

function moveAvatar(avatar, x, y, animate = true) {
  if (!animate) avatar.style.transition = "none";
  avatar.style.left = `${x}%`;
  avatar.style.top = `${y}%`;
  if (!animate) {
    requestAnimationFrame(() => {
      avatar.style.transition = "";
    });
  }
}

function renderRoundOver() {
  const winner = gameState.race_winner;
  const escaped = Object.entries(gameState.agents)
    .filter(([, agent]) => agent.escaped)
    .map(([agentId]) => agentId);
  let summary;
  if (winner) {
    summary = `Agent ${winner} wins — first through the exit.`;
  } else if (escaped.length) {
    summary = `Agents ${escaped.join(" & ")} escaped!`;
  } else {
    summary = "Time ran out — nobody escaped.";
  }

  elements.roundBanner.textContent = summary;
  elements.roundBanner.classList.remove("hidden");
  setCaption(summary, "Toggle mode or press Start again.");
  setControlsDisabled(true);
}

function setControlsDisabled(disabled) {
  elements.startButton.disabled = disabled;
  elements.resetButton.disabled = disabled;
  document.querySelectorAll(".spatial-object").forEach((button) => {
    button.disabled = disabled || busy;
  });
}

function showToast(message, isError = false) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 2600);
}

elements.speedButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setSpeed(Number.parseFloat(button.dataset.speed));
  });
});
elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setIncentiveMode(button.dataset.mode);
  });
});
elements.startButton.addEventListener("click", runEpisodeAnimated);
elements.resetButton.addEventListener("click", () => resetGame(true));

async function bootstrap() {
  try {
    applyPlaybackSpeed();
    await loadCatalogFlags();
    await configureDefaultEpisode();
    render();
  } catch (error) {
    showToast(error.message, true);
  }
}

bootstrap();
