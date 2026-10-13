const DEFAULT_PRESET = "ideal_cooperation";

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

function activeWingAgents() {
  return isSingleAgent() ? ["A"] : ["A", "B"];
}

async function loadCatalogFlags() {
  const catalog = await request("/api/catalog");
  cinematicPeekAvailable = catalog.features?.peek_auto_step !== false;
}

async function configureDefaultEpisode() {
  const payload = await request("/api/configure", {
    method: "POST",
    body: JSON.stringify({ preset: DEFAULT_PRESET }),
  });
  gameState = payload.state;
  selectedNodeId.A = null;
  selectedNodeId.B = null;
  agentPositions.A = null;
  agentPositions.B = null;
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

function buildTrajectoryItem(entry) {
  const item = createElement("li", `trajectory-item agent-${entry.agent.toLowerCase()}`);
  if (!entry.valid) item.classList.add("is-invalid");
  const actionText = humanizeAction(entry.action);
  item.append(
    createElement(
      "span",
      "trajectory-line",
      "",
    ),
  );
  const line = item.querySelector(".trajectory-line");
  line.append(
    createElement("span", "trajectory-agent", `Agent ${entry.agent}: `),
    createElement("span", "trajectory-action", actionText),
  );
  if (entry.result) {
    item.append(createElement("span", "trajectory-result", entry.result));
  }
  return item;
}

function appendTrajectoryEntries(entries) {
  for (const entry of entries) {
    elements.trajectoryLog.append(buildTrajectoryItem(entry));
  }
  const items = elements.trajectoryLog.querySelectorAll(".trajectory-item");
  items.forEach((node, index) => {
    node.classList.toggle("is-latest", index === items.length - 1);
  });
  const latest = items[items.length - 1];
  if (latest) {
    latest.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function syncTrajectoryLog(fullRebuild = false) {
  const log = gameState?.episode_log || [];
  if (fullRebuild || log.length < trajectorySyncedLength) {
    resetTrajectoryLog();
    appendTrajectoryEntries(log);
    trajectorySyncedLength = log.length;
    return;
  }
  if (log.length === trajectorySyncedLength) return;
  appendTrajectoryEntries(log.slice(trajectorySyncedLength));
  trajectorySyncedLength = log.length;
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

function clearWingBubbles() {
  for (const agent of ["A", "B"]) {
    WING_UI[agent].stage
      ?.querySelectorAll(".wing-action-bubble")
      .forEach((node) => node.remove());
  }
}

async function animateAgentForAction(agent, action) {
  const timing = getSimTiming();
  const ui = WING_UI[agent];
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
    showWingBubble(agent, parts.text);
    moveAvatar(ui.avatar, 50, 72, true);
    await delay(timing.moveMs);
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

function showWingBubble(agent, text) {
  const ui = WING_UI[agent];
  ui.stage.querySelectorAll(".wing-action-bubble").forEach((node) => node.remove());
  ui.stage.append(createElement("div", "wing-action-bubble", text));
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

  const ui = WING_UI[agent];
  ui.avatar.classList.add("acting");
  moveAvatar(ui.avatar, target.x, target.y, true);
  highlightActionTarget(agent, nodeId);
  await delay(durationMs);
  ui.avatar.classList.remove("acting");
  highlightActionTarget(agent, null);
}

function highlightActionTarget(agent, nodeId) {
  WING_UI[agent].nodes.querySelectorAll(".spatial-object").forEach((object) => {
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
    return payload.turn;
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

async function playCinematicStep() {
  applyPlaybackSpeed();
  const timing = getSimTiming();
  const upcoming = await peekUpcomingTurn();

  if (upcoming) {
    const { agent, action } = upcoming;
    highlightActiveWing(agent);
    setCaption(
      `${capitalize(agentLabel(agent))} ${humanizeAction(action)}…`,
      turnsLeftLabel(),
    );
    await animateAgentForAction(agent, action);
    await delay(timing.beforeMs);
  } else {
    await delay(timing.beforeMs);
  }

  const payload = await executeAutoStep();
  gameState = payload.state;
  const turn = payload.turn || gameState.episode_log?.[gameState.episode_log.length - 1];
  if (turn && !upcoming) {
    highlightActiveWing(turn.agent);
    setCaption(
      `${capitalize(agentLabel(turn.agent))} ${humanizeAction(turn.action)}…`,
      turnsLeftLabel(),
    );
    await animateAgentForAction(turn.agent, turn.action);
  }

  renderAfterSimStep(turn, payload.outcome);
  if (turn) {
    const ok = turn.valid !== false && payload.outcome.valid !== false;
    const resultText = turn.result || payload.outcome.result;
    setCaption(
      `${capitalize(agentLabel(turn.agent))} ${humanizeAction(turn.action)}`,
      ok ? resultText : `Failed — ${resultText}`,
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

  for (const agent of activeWingAgents()) {
    const wing = gameState.wings[agent];
    if (wing) syncWingMap(agent, wing);
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
    await configureDefaultEpisode();
    resetTrajectoryLog();
    await resetGame(false);
    setCaption("Episode started — two honest agents take turns.", turnsLeftLabel());
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
    clearWingBubbles();
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

  for (const agent of activeWingAgents()) {
    const wing = gameState.wings[agent];
    if (wing) syncWingMap(agent, wing);
  }

  if (gameState.round_over) {
    renderRoundOver();
    return;
  }

  const view = gameState.view;
  if (view) {
    highlightActiveWing(view.agent);
  }

  syncTrajectoryLog(true);

  if (!document.body.classList.contains("simulation-running")) {
    setCaption(
      "Press Start to watch two agents cooperate and escape.",
      gameState.scripted_turn ? turnsLeftLabel() : "",
    );
  }
  setControlsDisabled(false);
}

function highlightActiveWing(agent) {
  for (const wingAgent of ["A", "B"]) {
    WING_UI[wingAgent].panel?.classList.toggle("active-wing", wingAgent === agent);
  }
}

function ensureSpatialObject(agent, node, isActive) {
  const ui = WING_UI[agent];
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
  const { skipAvatarMove = false } = options;
  const ui = WING_UI[agent];
  ui.title.textContent = view.wing;
  ui.avatar.classList.remove("hidden");

  if (!skipAvatarMove) {
    const position = agentPositions[agent] || { x: 50, y: 88 };
    const animateMove = document.body.classList.contains("simulation-running");
    moveAvatar(ui.avatar, position.x, position.y, animateMove);
  }

  const isActive = agent === gameState.active_agent;
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
  const escaped = Object.entries(gameState.agents)
    .filter(([, agent]) => agent.escaped)
    .map(([agentId]) => agentId);
  const summary = escaped.length
    ? `Agents ${escaped.join(" & ")} escaped!`
    : "Time ran out — nobody escaped.";

  elements.roundBanner.textContent = summary;
  elements.roundBanner.classList.remove("hidden");
  setCaption(summary, "Reset or Start again.");
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
