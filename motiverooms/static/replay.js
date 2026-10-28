/* Recorded state only: this workspace never calls a policy or advances a game. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let recording = null, tick = 0, timer = null, filter = "all", selectedObject = null;
  let loadSequence = 0;
  const readable = (id) => String(id || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const shortAgent = (id) => id.startsWith("worker_") ? `Worker ${id.slice(7)}` : `Agent ${id}`;
  const color = (agent) => {
    const index = Math.max(0, recording.agents.indexOf(agent));
    return index < 4 ? `var(--worker-${index})` : `hsl(${(index * 137.508) % 360} 65% 38%)`;
  };
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function agentChip(agent, escaped = false) {
    const chip = el("span", `worker-chip${escaped ? " escaped" : ""}`);
    chip.style.setProperty("--agent-color", color(agent));
    chip.append(el("span", "worker-orb"), document.createTextNode(shortAgent(agent) + (escaped ? " ✓" : "")));
    return chip;
  }
  // Small, local vector marks keep the UI usable without an icon dependency.
  const icons = {
    book: ["M4 4h6a2 2 0 0 1 2 2v15a3 3 0 0 0-3-2H4z", "M20 4h-6a2 2 0 0 0-2 2v15a3 3 0 0 1 3-2h5z"],
    calendar: ["M4 5h16v16H4z", "M8 3v4m8-4v4M4 10h16m-11 4h1m4 0h1m-6 3h1m4 0h1"],
    box: ["M4 8h16v13H4z", "M3 4h18v4H3zm7 8h4"],
    note: ["M6 3h9l4 4v14H6z", "M14 3v5h5M9 12h7m-7 4h5"],
    door: ["M5 21V3h14v18M2 21h20", "M15 12h.01"],
    painting: ["M3 4h18v16H3z", "m3 17 5-6 4 4 3-3 6 5M7 8h.01"],
    torch: ["M7 3h10v5l-3 4v9h-4v-9L7 8z", "M7 7h10m-7 9h4"],
    sofa: ["M5 12V6h14v6M3 12h18v7H3z", "M6 19v3m12-3v3M7 9v6m10-6v6"],
    mat: ["M3 8h18v9H3z", "M6 10v5m3-5v5m3-5v5m3-5v5m3-5v5"],
  };
  function objectIcon(node) {
    const id = node.id;
    let key = node.kind === "door" ? "door" : node.kind === "clue" ? "note" : "box";
    if (/book/.test(id)) key = "book";
    else if (/calendar/.test(id)) key = "calendar";
    else if (/painting/.test(id)) key = "painting";
    else if (/torch/.test(id)) key = "torch";
    else if (/sofa/.test(id)) key = "sofa";
    else if (/mat/.test(id)) key = "mat";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [name, value] of Object.entries({viewBox:"0 0 24 24",fill:"none",stroke:"currentColor","stroke-linecap":"round","stroke-linejoin":"round","aria-hidden":"true"})) svg.setAttribute(name, value);
    for (const d of icons[key]) {
      const path = document.createElementNS(svg.namespaceURI, "path"); path.setAttribute("d", d); svg.append(path);
    }
    return svg;
  }
  function objectName(node) {
    if (node.id.startsWith("books_room_")) return "Bookshelf";
    if (node.id.startsWith("calendar_room_")) return "Calendar";
    if (node.id === "unpowered_torch") return "Torch station";
    return readable(node.id);
  }
  function roomName(id) { return recording.locations[id] || readable(id); }
  function current() { return recording.frames[tick]; }
  function reading(node, frame) {
    return Object.values(frame.state.agents).map((a) => a.read_codes[node.id]).find((value) => value !== undefined);
  }
  function nodeVisible(node, state) {
    return !node.hidden || recording.nodes.some((parent) => (parent.contents || []).includes(node.id) && state.opened_nodes.includes(parent.id));
  }
  function nodeStatus(node, frame) {
    const read = reading(node, frame), opened = frame.state.opened_nodes.includes(node.id);
    if (read !== undefined) return String(read).length > 42 ? "Clue read" : String(read);
    if (opened && node.id === "unpowered_torch") return "Powered";
    if (opened) return node.kind === "search" ? "Searched" : "Open";
    if (node.requires) return node.requires.code ? "Code lock" : "Item mechanism";
    if (node.kind === "clue") return "Unread clue";
    if (node.kind === "search") return "Searchable";
    return "Inspectable";
  }
  function renderMap() {
    const frame = current(), state = frame.state;
    const visited = new Set();
    for (const prior of recording.frames.slice(0, tick + 1)) Object.values(prior.state.agents).forEach((a) => visited.add(a.location));
    const targets = new Map();
    for (const action of frame.actions) {
      if (!["inspect", "take", "use", "enter"].includes(action.verb)) continue;
      for (const id of action.args) {
        if (!recording.nodes.some((node) => node.id === id)) continue;
        if (!targets.has(id)) targets.set(id, new Set());
        targets.get(id).add(action.agent);
      }
    }
    const map = $("room-map"); map.replaceChildren(); map.classList.toggle("four", Object.keys(recording.locations).length === 4);
    Object.entries(recording.locations).forEach(([id, name], index) => {
      const workers = recording.agents.filter((agent) => (state.agents[agent].location || agent) === id);
      const explored = visited.has(id) || !Object.values(state.agents).some((a) => a.location);
      const room = el("section", `room${explored ? "" : " unvisited"}${workers.length ? " active" : ""}`);
      room.dataset.room = id;
      const header = el("div", "room-header"), label = el("div");
      label.append(el("span", "room-number", String(index + 1).padStart(2, "0") + " / ROOM"), el("h3", "", name), el("span", "room-id", id));
      header.append(label, el("span", "room-status", explored ? "Explored" : "Unvisited")); room.append(header);
      const objects = el("div", "room-objects");
      if (explored) {
        const nodes = recording.nodes.filter((node) => (node.location || node.wing) === id && nodeVisible(node, state));
        for (const node of nodes) {
          const opened = state.opened_nodes.includes(node.id), read = reading(node, frame) !== undefined;
          const interacting = [...(targets.get(node.id) || [])];
          const button = el("button", `object${opened ? " opened" : ""}${read ? " read" : ""}${interacting.length ? " targeted" : ""}${node.kind === "door" ? " exit" : ""}`);
          if (interacting.length) button.style.setProperty("--agent-color", color(interacting[0]));
          button.type = "button"; button.dataset.object = node.id;
          button.setAttribute("aria-label", `${objectName(node)}: ${nodeStatus(node, frame)}. ${node.desc}`);
          button.append(objectIcon(node), el("span", "object-name", objectName(node)), el("span", "object-meta", nodeStatus(node, frame)));
          if (interacting.length) {
            const badges = el("span", "object-workers");
            interacting.forEach((agent) => badges.append(agentChip(agent)));
            button.append(badges);
          }
          button.addEventListener("click", () => { selectedObject = selectedObject === node.id ? null : node.id; renderObjectDetail(); });
          objects.append(button);
        }
      } else objects.append(el("p", "room-unexplored", "Contents appear when\na worker explores this room."));
      room.append(objects, el("span", "room-door"));
      const occupants = el("div", "room-workers");
      workers.forEach((agent) => occupants.append(agentChip(agent, state.agents[agent].escaped)));
      if (!workers.length) occupants.append(el("span", "room-empty", "No workers here"));
      room.append(occupants); map.append(room);
    });
    renderObjectDetail();
  }
  function renderObjectDetail() {
    const panel = $("object-detail"), node = recording.nodes.find((n) => n.id === selectedObject);
    const frame = current();
    if (!node || !nodeVisible(node, frame.state)) { panel.hidden = true; return; }
    panel.hidden = false; panel.replaceChildren(el("strong", "", `${objectName(node)} · ${node.id}`), el("p", "", node.desc));
    const read = reading(node, frame);
    if (read !== undefined) panel.append(el("p", "", `Recorded reading: ${read}`));
    const opened = frame.state.opened_nodes.includes(node.id);
    const items = (node.items || []).concat(opened ? node.contents || [] : []).filter((id) => recording.items[id] && !frame.state.taken_items.includes(id));
    if (items.length) panel.append(el("p", "", "Available here: " + items.map((id) => recording.items[id]).join(", ")));
    const holders = recording.agents.filter((agent) => frame.state.agents[agent].bag.length);
    const inventory = holders.map((agent) => `${shortAgent(agent)}: ${frame.state.agents[agent].bag.map((id) => recording.items[id] || id).join(", ")}`);
    if (inventory.length) panel.append(el("p", "", "Current inventories — " + inventory.join(" · ")));
  }
  function taskDetails(assignments) {
    const section = el("details"); section.append(el("summary", "", `${Object.keys(assignments).length} worker objectives · expand`));
    for (const [agent, task] of Object.entries(assignments)) {
      const row = el("div", "task-message"); row.append(agentChip(agent), el("p", "", task)); section.append(row);
    }
    return section;
  }
  function renderMessages() {
    const log = $("message-log"), nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
    const visible = recording.messages.filter((message) => message.step <= tick && (
      filter === "all" || (filter === "inbox" && message.kind === "report") || (filter === "outbox" && ["spawn","check_in","exit"].includes(message.kind))));
    log.replaceChildren();
    for (const message of visible) {
      const manager = ["spawn", "check_in", "exit"].includes(message.kind);
      const card = el("article", `message-card${manager ? " manager" : ""}${message.kind === "exit" ? " exit-message" : ""}${message.kind === "warning" ? " warning" : ""}`);
      const heading = el("div", "message-card-heading");
      let title = `${recording.agents.includes(message.sender) ? shortAgent(message.sender) : message.sender} → ${message.recipient}`;
      if (message.kind === "spawn") title = `Team created · ${message.workers.length} workers`;
      if (message.kind === "check_in") title = `Manager check-in · ${message.received} report${message.received === 1 ? "" : "s"} received`;
      if (message.kind === "exit") title = "Exit briefing → every worker";
      if (message.kind === "warning") title = "Task-board update rejected";
      heading.append(el("span", "", title), el("time", "", message.step === 0 ? "SETUP" : `T${String(message.step).padStart(2,"0")}`));
      card.append(heading);
      if (message.kind === "exit") {
        const b = message.briefing;
        card.append(el("p", "", `${b.node_id} · ${b.room_id ? roomName(b.room_id) : "location not yet recorded"} · code ${b.code || "not yet known"}`));
        card.append(el("span", "delivery", "Persistent team briefing · reported, unverified"));
      } else if (message.kind === "check_in") {
        const reasons = (message.trigger || []).map((reason) => ({new_reports:"new reports",silent_workers:"silent-worker check",team_chat:"team chat",scheduled:"per-tick coordination"}[reason] || reason));
        card.append(el("p", "", reasons.join(" + ")));
        if (Object.keys(message.assignments).length) card.append(taskDetails(message.assignments));
        else card.append(el("p", "", "Current objectives retained."));
        if (message.text) {
          const detail = el("details"); detail.append(el("summary", "", "Manager rationale"), el("p", "", message.text)); card.append(detail);
        }
      } else {
        card.append(el("p", "", message.text));
        if (message.kind === "spawn") card.append(taskDetails(message.assignments));
      }
      if (message.kind === "report") {
        const received = message.received_step !== null && message.received_step <= tick;
        card.append(el("span", "delivery", received ? `PRIVATE · received by manager before tick ${message.received_step}` : "PRIVATE · queued for the next manager check-in"));
      }
      log.append(card);
    }
    if (!visible.length) log.append(el("p", "message-placeholder", filter === "inbox" ? "No worker reports have been sent yet." : "No messages at this point in the recording."));
    if (nearBottom) log.scrollTop = log.scrollHeight;
    const calls = recording.messages.filter((m) => m.step <= tick && ["spawn","check_in"].includes(m.kind)).length;
    $("manager-status").textContent = `Manager calls so far: ${calls} · ${current().manager ? "checked in this tick" : tick ? "workers acting independently" : "team ready"}`;
  }
  function actionTitle(action) {
    if (!action.valid) return "Action rejected";
    if (action.result === "wrong") return "Wrong attempt";
    return ({inspect:"Inspect object",take:"Collect item",use:"Use item",enter:"Enter code",move:`Move to ${roomName(action.to_room)}`,report:"Report to manager",wait:"Wait",say:"Message team",give:"Transfer item",give_to:"Transfer item",write_note:"Write shared note",read_notes:"Read shared notes"})[action.verb] || readable(action.verb);
  }
  function buildTrajectory() {
    const head = el("tr"); head.append(el("th", "", "TICK"));
    recording.agents.forEach((agent) => { const th = el("th"); th.append(agentChip(agent)); head.append(th); });
    $("trajectory-head").replaceChildren(head);
    const body = $("trajectory-body"); body.replaceChildren();
    for (const frame of recording.frames.slice(1)) {
      const row = el("tr"); row.dataset.tick = frame.step;
      const index = el("td"), jump = el("button", "tick-button", String(frame.step).padStart(2, "0"));
      jump.type = "button"; jump.setAttribute("aria-label", `Go to tick ${frame.step}`);
      jump.addEventListener("click", () => { pause(); setTick(frame.step); $("room-map").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block:"start"}); });
      index.append(jump); if (frame.manager) index.append(el("span", "tick-manager", "M check-in")); row.append(index);
      recording.agents.forEach((agent) => {
        const cell = el("td"), action = frame.actions.find((a) => a.agent === agent);
        if (!action) cell.append(el("span", "action-idle", "Inactive"));
        else {
          const title = el("div", "action-heading", actionTitle(action));
          title.append(el("span", `action-state${!action.valid || action.result === "wrong" ? " rejected" : ""}`, !action.valid ? "REJECTED" : action.result === "wrong" ? "WRONG" : "OK"));
          cell.append(title, el("div", "raw-action", action.raw), el("div", "action-result", action.result));
        }
        row.append(cell);
      });
      body.append(row);
    }
  }
  function setTick(value) {
    if (!recording) return;
    tick = Math.max(0, Math.min(recording.frames.length - 1, Number(value)));
    $("seek").value = tick; $("tick-value").textContent = `${tick} / ${recording.frames.length - 1}`;
    $("previous").disabled = tick === 0; $("next").disabled = tick === recording.frames.length - 1;
    renderMap(); renderMessages(); renderPuzzleGuide();
    const frame = current();
    $("moment").textContent = tick === 0 ? "Team formed. Press play or step through the recording. The map and messages follow the same tick." :
      frame.state.agents && Object.values(frame.state.agents).every((a) => a.escaped) ? `Tick ${tick} · Everyone escaped. The recorded episode is complete.` :
      `Tick ${tick} · ` + frame.actions.map((a) => `${shortAgent(a.agent)}: ${actionTitle(a).toLowerCase()}`).join(" · ");
    $("trajectory-body").querySelectorAll("tr").forEach((row) => {
      const step = Number(row.dataset.tick); row.classList.toggle("current", step === tick); row.classList.toggle("future", step > tick);
      row.hidden = $("past-only").checked && step > tick;
      row.querySelector("button").setAttribute("aria-current", step === tick ? "step" : "false");
    });
    if (tick === recording.frames.length - 1) pause();
  }
  function pause() {
    clearInterval(timer); timer = null;
    $("play-label").textContent = recording && tick === recording.frames.length - 1 ? "Replay episode" : "Play episode";
    $("play-icon").textContent = "▶";
  }
  function play() {
    if (!recording) return;
    if (timer) { pause(); return; }
    if (tick === recording.frames.length - 1) setTick(0);
    $("play-label").textContent = "Pause"; $("play-icon").textContent = "Ⅱ";
    timer = setInterval(() => setTick(tick + 1), Number($("speed").value));
  }
  function renderPuzzleGuide() {
    const flow = $("puzzle-flow"); flow.replaceChildren();
    const required = ["doormat", "desk_drawer", "unpowered_torch", "tool_cabinet", "painting", "sofa", "display_case", "shipping_receipt", "wall_plaque", "final_exit"];
    if (!required.every((id) => recording.nodes.some((node) => node.id === id))) {
      flow.append(el("p", "guide-note", "An easy interpretation is not available for this puzzle yet."));
      return;
    }
    const state = current().state;
    const opened = (id) => state.opened_nodes.includes(id);
    const taken = (id) => state.taken_items.includes(id);
    const discoveries = Object.assign({}, ...recording.frames.slice(0, tick + 1).map((frame) =>
      Object.assign({}, ...Object.values(frame.state.agents).map((agent) => agent.read_codes))));
    const location = (id) => roomName(recording.nodes.find((node) => node.id === id).location);
    const step = (text, room, done = false) => {
      const box = el("li", `flow-step${done ? " discovered" : ""}`);
      if (done) box.append(el("strong", "flow-status", "✓ Discovered / completed"));
      if (room) box.append(el("small", "", room));
      box.append(el("span", "", text)); return box;
    };
    const branch = (title, steps) => {
      const box = el("section", "flow-branch"); box.append(el("h3", "", title));
      const list = el("ol", "flow-steps"); steps.forEach(([text, room, done]) => list.append(step(text, room, done)));
      box.append(list); return box;
    };
    const paths = el("div", "flow-paths");
    const torch = branch("1 · Reveal the moon clue", [
      ["Search under the doormat → find the desk key", location("doormat"), opened("doormat")],
      ["Use the key to unlock the drawer → reveal the torch station", location("desk_drawer"), opened("desk_drawer")],
    ]);
    const battery = el("div", `flow-input${opened("tool_cabinet") ? " discovered" : ""}`, `${opened("tool_cabinet") ? "✓ " : ""}Also needed: search the tool cabinet in ${location("tool_cabinet")} → find batteries.`);
    torch.append(battery);
    const torchEnd = el("ol", "flow-steps");
    [["Bring batteries to the torch station → take the working UV torch", taken("uv_torch")], ["Use the UV torch on the Lounge painting", opened("painting")], [discoveries.painting_inscription || "Read the moon inscription · not discovered yet", Boolean(discoveries.painting_inscription)]].forEach(([text, done]) => torchEnd.append(step(text, null, done)));
    torch.append(torchEnd); paths.append(torch);
    paths.append(branch("2 · Reveal the sun clue", [
      ["Search the sofa cushions → find a brass token", location("sofa"), opened("sofa")],
      ["Use the token to open the display case", location("display_case"), opened("display_case")],
      [discoveries.case_inscription || "Read the sun inscription · not discovered yet", null, Boolean(discoveries.case_inscription)],
    ]));
    paths.append(branch("3 · Read the star clue", [
      ["Inspect the shipping receipt", location("shipping_receipt"), Boolean(discoveries.shipping_receipt)],
      [discoveries.shipping_receipt || "Read the star inscription · not discovered yet", null, Boolean(discoveries.shipping_receipt)],
    ]));
    flow.append(paths, el("div", "flow-merge", "All three clue paths come together ↓"));
    const plaque = discoveries.wall_plaque || "Read the wall plaque to learn the symbol order · not discovered yet.";
    const order = plaque.match(/order:\s*(.*?)\./)?.[1]?.split(/\s*→\s*/);
    const fragments = Object.fromEntries(Object.values(discoveries).flatMap((text) => {
      const match = String(text).match(/^(\w+):\s*(\d+)$/); return match ? [[match[1], match[2]]] : [];
    }));
    const code = order && order.every((symbol) => fragments[symbol]) ? order.map((symbol) => fragments[symbol]).join("") : null;
    flow.append(branch("4 · Assemble the code and exit", [
      [plaque, location("wall_plaque"), Boolean(discoveries.wall_plaque)],
      [code ? `${order.map((symbol) => `${symbol} (${fragments[symbol]})`).join(" → ")} = ${code}` : "Combine the discovered numbers in the plaque’s order.", null, Boolean(code)],
      [code ? `Return to ${location("final_exit")} and enter ${code} at the final exit` : `Return to ${location("final_exit")} and enter the code at the final exit`, null, opened("final_exit")],
    ]));
  }
  function showRecording(data) {
    pause(); recording = data; tick = 0; selectedObject = null;
    $("workspace").hidden = false; $("empty-state").hidden = true; $("error").hidden = true;
    $("episode-name").textContent = data.summary.room_id;
    const badge = $("outcome-badge"); badge.textContent = data.outcome === "success" ? "● Successful recording" : `● ${readable(data.outcome)}`;
    badge.classList.toggle("failed", data.outcome !== "success");
    $("episode-stats").replaceChildren(...[[data.summary.rooms,"rooms"],[data.agents.length,"workers"],[data.frames.length - 1,"ticks"],[data.summary.manager_calls,"manager calls"]].map(([n, label]) => {
      const span = el("span"); span.append(el("strong", "", String(n)), document.createTextNode(" " + label)); return span;
    }));
    $("worker-count").textContent = `${data.agents.length} workers`;
    $("seek").max = data.frames.length - 1;
    $("worker-legend").replaceChildren(...data.agents.map((agent) => {
      const label = el("span", "legend-item"); label.style.setProperty("--agent-color", color(agent)); label.append(el("span", "worker-orb"), document.createTextNode(shortAgent(agent))); return label;
    }));
    buildTrajectory(); setTick(0);
  }
  function error(message) { $("error").textContent = message; $("error").hidden = false; }
  async function jsonResponse(response) {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to load this recording.");
    return data;
  }
  async function load(id) {
    const request = ++loadSequence; pause(); $("workspace").classList.add("loading");
    try {
      const data = await jsonResponse(await fetch(`/api/recording?id=${encodeURIComponent(id)}`));
      if (request !== loadSequence) return;
      showRecording(data);
      const url = new URL(location.href); url.searchParams.set("episode", id); history.replaceState(null,"",url);
    } catch (e) { if (request === loadSequence) error(e.message); }
    finally { if (request === loadSequence) $("workspace").classList.remove("loading"); }
  }
  $("play").addEventListener("click", play);
  $("restart").addEventListener("click", () => { pause(); setTick(0); });
  $("previous").addEventListener("click", () => { pause(); setTick(tick - 1); });
  $("next").addEventListener("click", () => { pause(); setTick(tick + 1); });
  $("seek").addEventListener("input", (event) => { pause(); setTick(event.target.value); });
  $("speed").addEventListener("change", () => { if (timer) { pause(); play(); } });
  $("past-only").addEventListener("change", () => setTick(tick));
  document.querySelectorAll(".message-tab").forEach((button) => button.addEventListener("click", () => {
    filter = button.dataset.filter;
    document.querySelectorAll(".message-tab").forEach((b) => { b.classList.toggle("active", b === button); b.setAttribute("aria-pressed", String(b === button)); });
    if (recording) renderMessages();
  }));
  ["import-button","empty-import"].forEach((id) => $(id).addEventListener("click", () => $("file-input").click()));
  $("file-input").addEventListener("change", async (event) => {
    const file = event.target.files[0]; if (!file) return;
    pause(); const request = ++loadSequence;
    try {
      if (file.size > 20_000_000) throw new Error("Choose a recording smaller than 20 MB.");
      const kinds = new Set(["start","joint_step","orchestrator_decision","end","metrics"]);
      const events = (await file.text()).split(/\r?\n/).filter((line) => line.trim()).map((line, i) => {
        try { return JSON.parse(line); } catch { throw new Error(`Invalid JSON on line ${i + 1}.`); }
      }).filter((row) => kinds.has(row.type));
      const data = await jsonResponse(await fetch("/api/recordings/preview", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({events})}));
      if (request !== loadSequence) return;
      showRecording(data);
    } catch (e) { if (request === loadSequence) error(e.message); }
    finally { event.target.value = ""; }
  });
  document.addEventListener("keydown", (event) => {
    if (!recording || /INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY/.test(document.activeElement.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); play(); }
    if (event.key === "ArrowRight") { event.preventDefault(); pause(); setTick(tick + 1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); pause(); setTick(tick - 1); }
  });
  (async () => {
    try {
      const catalog = await jsonResponse(await fetch("/api/recordings"));
      const requested = new URL(location.href).searchParams.get("episode");
      const id = catalog.recordings.some((r) => r.id === requested) ? requested : catalog.default_id;
      if (id) await load(id);
      else $("empty-state").hidden = false;
    } catch (e) { error(e.message); $("empty-state").hidden = false; }
  })();
})();
