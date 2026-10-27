/* Product Studio — Studio Floor.
 *
 * The isometric scene is built here from small boxes (same CSS engine as the MQ-MAS quant floor:
 * .b with top/left/right faces). Furniture is grouped and each group gets a z-index from its
 * footprint centre, so desks, chairs and walking agents overlap correctly.
 *
 * Live data: WebSocket /ws (agent_status, activity, meeting_*, typing, sale) + REST polling of the
 * executive endpoints (/api/dashboard/*, /api/sectors, /api/products, /api/ideas, /api/meetings).
 */
"use strict";

const ROLES = ["product", "builder", "growth", "sales", "cfo"];
const WALK_MS = 2400;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const usd = (v, d = 2) => `${v < 0 ? "−" : ""}$${Math.abs(Number(v) || 0).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;
const hhmm = (iso) => { try { return new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }); } catch { return ""; } };
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

// ======================================================================= scene primitives
const world = $("world"), overlay = $("overlay");

function box(x, y, w, d, h, c, opt = {}) {
  const b = el("div", `b ${opt.cls || ""}`);
  let st = `--x:${x}px;--y:${y}px;--w:${w}px;--d:${d}px;--h:${h}px;--z:${opt.z || 0}px;--c:${c}`;
  if (opt.cl) st += `;--cl:${opt.cl}`;
  if (opt.cr) st += `;--cr:${opt.cr}`;
  b.setAttribute("style", st);
  b.append(el("span", "t"), el("span", "l"), el("span", "r"));
  return b;
}

/** A group of boxes painted in authored order; groups are ordered by depth (footprint centre x + y). */
function group(depth, ...items) {
  const g = el("div", "grp");
  g.style.zIndex = String(Math.round(depth * 2) + 2000);
  g.append(...items);
  world.append(g);
  return g;
}

function floorLabel(x, y, text, color) {
  const f = el("div", "fl room-label", text);
  f.setAttribute("style", `--x:${x}px;--y:${y}px;color:${color}`);
  f.style.zIndex = "5";
  world.append(f);
}

function desk(x, y, screen = "scr") {
  const legs = [[x + 2, y + 2], [x + 99, y + 2], [x + 2, y + 43], [x + 99, y + 43]].map(([lx, ly]) => box(lx, ly, 3, 3, 22, "#3A3F52"));
  const mon = box(x + 26, y + 5, 52, 3, 30, "#1A1F2E", { cls: screen, z: 36 });
  group(x + y + 76, ...legs, box(x, y, 104, 48, 4, "#D9B68A", { cls: "wood", z: 22 }),
    box(x + 40, y + 6, 16, 8, 2, "#2B3145", { z: 26 }), box(x + 46, y + 8, 3, 3, 10, "#8A93A8", { z: 28 }), mon,
    box(x + 32, y + 26, 34, 10, 2, "#EDEFF4", { cls: "keys", z: 26 }), box(x + 72, y + 28, 5, 7, 2, "#EDEFF4", { z: 26 }),
    box(x + 10, y + 22, 7, 7, 8, "#FFFFFF", { z: 26 }));
  return mon;
}

function chair(x, y, seat, back) {
  group(x + y + 21, box(x + 1, y + 2, 20, 20, 3, "#2B3145"), box(x + 9, y + 10, 4, 4, 11, "#9AA2B5", { z: 3 }),
    box(x, y, 22, 20, 5, seat, { z: 14 }));
  group(x + y + 42, box(x, y + 18, 22, 4, 22, back, { z: 17 }));
}

function stool(x, y, c) {
  group(x + y + 18, box(x + 6, y + 6, 4, 4, 12, "#9AA2B5"), box(x, y, 16, 16, 4, c, { z: 12 }));
}

function plant(x, y) {
  group(x + y + 30, box(x, y, 18, 18, 18, "#E9E4DC"), box(x - 1, y - 1, 20, 20, 3, "#F5F2EC", { z: 16 }),
    box(x + 1, y + 1, 16, 16, 12, "#2F9E5F", { z: 19, cl: "#2A8C54", cr: "#237848" }),
    box(x + 4, y + 3, 11, 11, 10, "#47C27B", { z: 31, cl: "#3CAE6C", cr: "#33985E" }),
    box(x + 7, y + 6, 6, 6, 8, "#5BD68E", { z: 41 }));
}

// ======================================================================= the floor
const ROOMS = {
  product: { label: "Research Lab", color: "#7C6CFF", rug: "#DDD8FF", r: [24, 24, 236, 180] },
  builder: { label: "Dev Room", color: "#19B394", rug: "#CFEFE6", r: [284, 24, 216, 180] },
  board: { label: "Board Room", color: "#8A93A8", rug: "#E6EAF3", r: [528, 24, 168, 276] },
  growth: { label: "Growth Studio", color: "#E24C94", rug: "#FFDCEC", r: [24, 228, 236, 228] },
  sales: { label: "Sales Desk", color: "#D98A0B", rug: "#FFEBC6", r: [284, 228, 216, 228] },
  cfo: { label: "Finance", color: "#1D8FD6", rug: "#D6ECFF", r: [528, 324, 168, 132] },
};

function buildScene() {
  const base = el("div", "grp"); base.style.zIndex = "1";
  base.append(box(0, 0, 720, 480, 16, "#E9DAC3", { cls: "floor", z: -16, cl: "#BFA987", cr: "#A89272" }),
    box(-12, 0, 12, 480, 92, "#F8F9FC", { cr: "#E2E6EF" }), box(0, -12, 720, 12, 92, "#F8F9FC", { cl: "#ECEFF5" }),
    box(-2, 300, 2, 130, 50, "#BFD6FF", { z: 24, cr: "#B5D0FF" }), box(300, -2, 180, 2, 50, "#BFD6FF", { z: 24, cl: "#B5D0FF" }));
  for (const k of Object.keys(ROOMS)) { const [x, y, w, d] = ROOMS[k].r; base.append(box(x, y, w, d, 2, ROOMS[k].rug, { cls: "rug" })); }
  world.append(base);
  floorLabel(36, 180, "Research Lab", "#5B4BE0"); floorLabel(296, 180, "Dev Room", "#0E7A62"); floorLabel(540, 276, "Board Room", "#5B6478");
  floorLabel(36, 432, "Growth Studio", "#B42371"); floorLabel(296, 432, "Sales Desk", "#8A5A00"); floorLabel(540, 432, "Finance", "#0F5E94");

  // back-wall props
  group(30, box(60, -3, 160, 3, 56, "#FFFFFF", { z: 22, cl: "#FFFFFF" }),
    ...[[72, "#FFE27A"], [100, "#FFB3C7"], [128, "#B5F0D9"], [156, "#C9C0FF"], [184, "#FFE27A"]].map(([px, c], i) =>
      box(px, -4, 20, 1, 18, c, { z: 50 - (i % 2) * 24, cl: c })));
  group(140, box(0, 120, 16, 72, 66, "#C9A57B", { cls: "shelf", cr: "#B48F63" }));
  group(500, box(560, -3, 120, 3, 58, "#0E1322", { cls: "scr2", z: 20 }));
  group(560, box(466, 30, 26, 26, 76, "#1B2132", { cls: "rack" }));
  group(330, box(-2, 250, 3, 80, 56, "#FFFFFF", { z: 24, cr: "#FFF0F6" }));

  // desks, chairs: one per agent
  const DESK = { product: [80, 64, "scr"], builder: [340, 64, "scr2"], growth: [80, 290, "scr"], sales: [340, 290, "scr2"], cfo: [560, 350, "scr2"] };
  const SEAT = { product: ["#B7ADFF", "#9C8FFF"], builder: ["#93E2CE", "#3FC7A5"], growth: ["#FFB8D9", "#F472B6"], sales: ["#FFD58A", "#F5A524"], cfo: ["#A8D8FF", "#38BDF8"] };
  const monitors = {};
  for (const r of ROLES) {
    const [x, y, scr] = DESK[r];
    monitors[r] = desk(x, y, scr);
    chair(x + 38, y + 60, ...SEAT[r]);
    AGENTS[r].home = [x + 49, y + 76];
  }
  // Board room: long table with stools
  group(560 + 50 + 90 + 75, box(574, 94, 4, 4, 22, "#3A3F52"), box(662, 94, 4, 4, 22, "#3A3F52"), box(574, 232, 4, 4, 22, "#3A3F52"),
    box(662, 232, 4, 4, 22, "#3A3F52"), box(570, 90, 100, 150, 4, "#F7F5F1", { z: 22, cl: "#E1DCD2", cr: "#CEC8BC" }),
    box(600, 120, 18, 18, 3, "#7C6CFF", { z: 26 }), box(630, 180, 8, 8, 7, "#FFFFFF", { z: 26 }), box(590, 200, 14, 10, 1, "#FFFFFF", { z: 26 }));
  const MEET = [[548, 112], [548, 162], [548, 212], [692, 132], [692, 192]];
  ROLES.forEach((r, i) => { const [x, y] = MEET[i]; stool(x - 8, y - 8, "#2B3145"); AGENTS[r].meet = [x, y]; });
  group(516 + 24, box(516, 24, 4, 276, 50, "#BFD6FF", { cls: "glass" }));  // drawn early: it is the far wall of the board room
  // finance safe + coins, coffee, plants
  group(660 + 336 + 30, box(660, 336, 30, 28, 34, "#2B3145", { cl: "#363D54", cr: "#232838" }), box(667, 364, 12, 1, 12, "#9AA2B5", { z: 12 }));
  group(632 + 340 + 10, box(630, 340, 8, 8, 10, "#F5C04A"), box(640, 342, 8, 8, 6, "#F5C04A"));
  group(680 + 30 + 14, box(680, 30, 14, 14, 26, "#3A3F52"), box(682, 32, 10, 10, 3, "#E8574A", { z: 26 }));
  [[250, 30], [490, 210], [30, 440], [490, 440], [700, 300], [250, 230]].forEach(([x, y]) => plant(x, y));
  return monitors;
}

// ======================================================================= agents
const AGENTS = {
  product: { name: "Iris", color: "#7C6CFF", vars: "--skin:#F1C27D;--hair:#3B2A1A;--shirt:#6C5CE7;--pants:#2D2A4A",
    body: '<span class="sd"></span><span class="hb"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="hd"><span class="hr" style="height:9px;border-radius:0 0 60% 20%"></span><span class="ey e1"></span><span class="ey e2"></span><span class="ck c1"></span><span class="ck c2"></span><span class="mo"></span></span>' },
  builder: { name: "Theo", color: "#19B394", vars: "--skin:#8D5524;--hair:#141414;--shirt:#138F77;--pants:#1E3A5F;--under:#0E6B58",
    body: '<span class="sd"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="hd"><span class="hr" style="height:7px"></span><span class="ey e1"></span><span class="ey e2"></span><span class="mo"></span></span><span style="left:7px;bottom:calc(var(--lh) + 42px);width:26px;height:15px;box-sizing:border-box;border:3px solid #2B3145;border-bottom:0;border-radius:14px 14px 0 0"></span><span style="left:5px;bottom:calc(var(--lh) + 35px);width:6px;height:10px;border-radius:3px;background:#2B3145"></span><span style="left:29px;bottom:calc(var(--lh) + 35px);width:6px;height:10px;border-radius:3px;background:#2B3145"></span>' },
  growth: { name: "Maya", color: "#F472B6", vars: "--skin:#C68642;--hair:#4A1414;--shirt:#E24C94;--pants:#2B2B35;--under:#FFF4EE",
    body: '<span class="sd"></span><span style="left:15px;bottom:calc(var(--lh) + 51px);width:10px;height:10px;border-radius:50%;background:var(--hair)"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="er r1"></span><span class="er r2"></span><span class="hd"><span class="hr" style="height:8px;border-radius:0 0 40% 40%"></span><span class="ey e1"></span><span class="ey e2"></span><span class="ck c1"></span><span class="ck c2"></span><span class="mo"></span></span>' },
  sales: { name: "Leo", color: "#F5A524", vars: "--skin:#E0AC69;--hair:#6B4423;--shirt:#E99A16;--pants:#2D3A4F;--under:#FFFFFF",
    body: '<span class="sd"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span style="left:18px;bottom:calc(var(--lh) + 9px);width:4px;height:14px;background:#1E3A5F;border-radius:1px"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="er r1"></span><span class="er r2"></span><span class="hd"><span class="hr" style="height:6px;border-radius:0 0 30% 70%"></span><span class="ey e1"></span><span class="ey e2"></span><span class="mo"></span></span>' },
  cfo: { name: "Nina", color: "#38BDF8", vars: "--skin:#F3D2B3;--hair:#1F1A17;--shirt:#1D8FD6;--pants:#22283A;--under:#E8F4FF",
    body: '<span class="sd"></span><span class="hb"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="hd"><span class="hr" style="height:8px"></span><span class="ey e1"></span><span class="ey e2"></span><span style="left:3px;top:9px;width:7px;height:6px;box-sizing:border-box;border:1.5px solid #1E2230;border-radius:2px"></span><span style="left:12px;top:9px;width:7px;height:6px;box-sizing:border-box;border:1.5px solid #1E2230;border-radius:2px"></span><span class="mo"></span></span>' },
};
const STATE_LABEL = { working: "TRABALHANDO", thinking: "PENSANDO", idle: "NA MESA", meeting: "REUNIÃO", blocked: "BLOQUEADO", waiting: "AGUARDANDO" };
const VOTE_LABEL = { APPROVE: "APROVA", REJECT: "REJEITA", ABSTAIN: "ABSTÉM" };

const monitors = buildScene();

function setPos(node, [x, y]) { node.style.setProperty("--x", `${x}px`); node.style.setProperty("--y", `${y}px`); }
function depthOf([x, y]) { return String(Math.round((x + y + 1) * 2) + 2000); }

for (const r of ROLES) {
  const a = AGENTS[r];
  a.where = "home"; a.state = "idle"; a.text = "";
  a.node = el("div", "a agent");
  a.node.style.setProperty("--z", "4px");
  a.pp = el("div", "pp seated"); a.pp.setAttribute("style", a.vars); a.pp.innerHTML = a.body;
  a.node.append(a.pp);
  setPos(a.node, a.home); a.node.style.zIndex = depthOf(a.home);
  world.append(a.node);
  a.ov = el("div", "a agent-ov"); a.ov.style.setProperty("--z", "4px"); setPos(a.ov, a.home);
  a.tag = el("div", "tag"); a.thought = el("div", "thought"); a.bubbleBox = el("div");
  a.ov.append(a.thought, a.tag, a.bubbleBox);
  overlay.append(a.ov);
  renderTag(r);
}

function renderTag(r) {
  const a = AGENTS[r];
  a.tag.replaceChildren();
  const dot = el("span", "dot"); dot.style.background = a.color;
  a.tag.append(dot, document.createTextNode(a.name));
  const st = a.where === "meet" ? "meeting" : a.state;
  a.tag.append(el("span", `stx s-${st}`, STATE_LABEL[st] || st.toUpperCase()));
  a.thought.textContent = a.where === "meet" ? "" : a.text;
  a.pp.classList.toggle("typing", a.where === "home" && (a.state === "working" || a.state === "thinking"));
  if (monitors[r]) monitors[r].classList.toggle("on", a.state === "working" || a.state === "thinking");
}

function moveAgent(r, where) {
  const a = AGENTS[r];
  if (a.where === where) return;
  a.where = where;
  const target = where === "meet" ? a.meet : a.home;
  a.pp.classList.remove("seated"); a.pp.classList.add("walking");
  a.node.style.zIndex = String(Math.max(Number(a.node.style.zIndex), Number(depthOf(target))));
  setPos(a.node, target); setPos(a.ov, target);
  clearTimeout(a.walkTimer);
  a.walkTimer = setTimeout(() => { a.pp.classList.remove("walking"); a.pp.classList.add("seated"); a.node.style.zIndex = depthOf(target); }, WALK_MS + 50);
  renderTag(r);
}

function bubbleSide(a) { const [x] = a.where === "meet" ? a.meet : a.home; return x > 450 ? "to-left" : "to-right"; }
function clearBubble(a) { clearTimeout(a.bubbleTimer); clearInterval(a.typeTimer); a.bubbleBox.replaceChildren(); }

function showTyping(r) {
  const a = AGENTS[r]; clearBubble(a);
  const b = el("div", `bubble typing ${bubbleSide(a)}`); b.append(el("span", "tdot"), el("span", "tdot"), el("span", "tdot"));
  a.bubbleBox.append(b);
}

function showBubble(r, text, vote, title) {
  for (const o of ROLES) if (o !== r) clearBubble(AGENTS[o]);
  const a = AGENTS[r]; clearBubble(a);
  for (const o of ROLES) AGENTS[o].ov.classList.toggle("front", o === r);
  const b = el("div", `bubble ${bubbleSide(a)}`);
  const head = el("div", "bh");
  const av = el("span", "av", a.name[0]); av.style.background = a.color;
  head.append(av, el("strong", "bt", title || a.name));
  if (vote && VOTE_LABEL[vote]) head.append(el("span", `vchip v-${vote}`, VOTE_LABEL[vote]));
  const p = el("p"); b.append(head, p); a.bubbleBox.append(b);
  let i = 0;
  a.typeTimer = setInterval(() => { p.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(a.typeTimer); }, 12);
  a.bubbleTimer = setTimeout(() => { b.classList.add("fade"); setTimeout(() => b.remove(), 400); }, Math.min(5000 + text.length * 40, 14000));
}

function floatCash(amount) {
  const a = AGENTS.sales;
  const f = el("div", "floatcash", `+${usd(amount)}`);
  a.ov.append(f); setTimeout(() => f.remove(), 2700);
}

function toast(text) {
  const t = el("div", "toast", text); document.body.append(t);
  setTimeout(() => t.classList.add("hide"), 4200); setTimeout(() => t.remove(), 4800);
}

// legend
for (const r of ROLES) {
  const c = el("span", "chip"); const sw = el("span", "sw"); sw.style.background = AGENTS[r].color;
  c.append(sw, document.createTextNode(`${AGENTS[r].name} · ${ROOMS[r].label}`)); $("legend").append(c);
}

// ======================================================================= meetings (live)
const meeting = { id: null, votes: {}, subject: "" };

function renderMeetCard(decision) {
  $("mc-dot").classList.toggle("on", !!meeting.id && !decision);
  $("mc-title").textContent = meeting.id ? (decision ? "Reunião encerrada" : "Reunião em andamento") : "Sem reunião agora";
  $("mc-sub").textContent = meeting.id ? meeting.subject + (decision ? ` → ${decision}` : "") :
    "Os agentes se reúnem na Board Room para votar cada ideia e revisar as metas semanais.";
  const rows = $("mc-rows"); rows.replaceChildren();
  if (!meeting.id) return;
  for (const r of ROLES) {
    const row = el("div", "mc-row"); const av = el("span", "av"); av.style.background = AGENTS[r].color;
    const v = meeting.votes[r];
    row.append(av, document.createTextNode(AGENTS[r].name), el("span", `st st-${v || "QUEUE"}`, v ? (VOTE_LABEL[v] || "—") : "…"));
    rows.append(row);
  }
}

function ataItem(agent, text, vote, at) {
  const a = AGENTS[agent] || { name: agent, color: "#5B6478" };
  const li = el("li"); const av = el("span", "av", a.name[0]); av.style.background = a.color;
  const body = el("div", "body"); const hdr = el("div", "hdr");
  hdr.append(el("strong", "", a.name), el("span", "time", hhmm(at || new Date().toISOString())));
  if (vote && VOTE_LABEL[vote]) hdr.append(el("span", `vchip v-${vote}`, VOTE_LABEL[vote]));
  body.append(hdr, el("p", "", text)); li.append(av, body);
  return li;
}

function pushAta(agent, text, vote) {
  const ul = $("ata"); const empty = ul.querySelector(".empty"); if (empty) empty.remove();
  ul.prepend(ataItem(agent, text, vote));
  while (ul.children.length > 12) ul.lastChild.remove();
}

// ======================================================================= WebSocket
let ws, backoff = 1000, refreshTimer;
function scheduleRefresh(ms = 1500) { clearTimeout(refreshTimer); refreshTimer = setTimeout(refreshAll, ms); }

function applyState(r, s) {
  if (!AGENTS[r] || !s) return;
  AGENTS[r].state = s.state || "idle"; AGENTS[r].text = s.text || ""; renderTag(r);
}

function onEvent(m) {
  switch (m.type) {
    case "hello":
      for (const [r, s] of Object.entries(m.agents || {})) applyState(r, s);
      if (m.meeting) { meeting.id = m.meeting.id; meeting.subject = m.meeting.subject; meeting.votes = m.meeting.votes || {}; ROLES.forEach((r) => moveAgent(r, "meet")); renderMeetCard(); }
      break;
    case "agent_status": applyState(m.agent, m); break;
    case "activity": appendFeed(m.agent, m.text, m.ts); scheduleRefresh(); break;
    case "meeting_start":
      meeting.id = m.id; meeting.subject = m.subject; meeting.votes = {};
      ROLES.forEach((r) => moveAgent(r, "meet")); renderMeetCard(); break;
    case "typing": if (AGENTS[m.agent]) showTyping(m.agent); break;
    case "meeting_message":
      meeting.votes[m.agent] = m.vote || ""; renderMeetCard();
      showBubble(m.agent, m.text, m.vote, `${AGENTS[m.agent].name} · ${meeting.subject.slice(0, 32)}`);
      pushAta(m.agent, m.text, m.vote); break;
    case "meeting_end":
      renderMeetCard(m.decision);
      // back to the desks, unless the next meeting already started meanwhile
      setTimeout(() => { if (meeting.id !== m.id) return; ROLES.forEach((r) => moveAgent(r, "home")); meeting.id = null; renderMeetCard(); }, 4000);
      scheduleRefresh(500); break;
    case "sale": floatCash(m.gross_usd); toast(`💰 Venda: ${m.product} — ${usd(m.gross_usd)} (${m.platform})`); scheduleRefresh(300); break;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { backoff = 1000; $("pill-conn").classList.remove("off"); $("conn-text").textContent = "Ao vivo"; };
  ws.onmessage = (e) => { try { onEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => { $("pill-conn").classList.add("off"); $("conn-text").textContent = "Reconectando…"; setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 15000); };
}

// ======================================================================= REST panels
async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function admin(path) {
  let token = store.get("studio_admin");
  if (!token) { token = prompt("ADMIN_TOKEN do escritório:") || ""; if (!token) return null; store.set("studio_admin", token); }
  const r = await fetch(path, { method: "POST", headers: { "X-Admin-Token": token } });
  if (r.status === 401) { store.set("studio_admin", ""); toast("Token inválido"); return null; }
  return r.json();
}

function setVal(id, v, signed) {
  const e = $(id); e.textContent = v;
  if (signed !== undefined) { e.classList.toggle("pos", signed > 0); e.classList.toggle("neg", signed < 0); }
}

function renderSummary(s) {
  setVal("h-rev", usd(s.month.net_revenue_usd)); setVal("h-profit", usd(s.month.profit_usd), s.month.profit_usd);
  setVal("h-rail", usd(s.monthly_costs.railway_usd));
  setVal("k-net", usd(s.all_time.net_revenue_usd)); $("k-gross").textContent = `bruto ${usd(s.all_time.gross_usd)} · taxas ${usd(s.all_time.fees_usd)}`;
  setVal("k-profit", usd(s.all_time.profit_usd), s.all_time.profit_usd);
  setVal("k-today", String(s.today.sales)); $("k-today-usd").textContent = usd(s.today.net_revenue_usd) + " líquido";
  setVal("k-live", String(s.active_products)); $("k-platforms").textContent = s.platforms.length ? s.platforms.join(" · ") : "nenhuma plataforma conectada";
  const auto = $("pill-auto"); auto.textContent = s.autopublish ? "AUTOPUBLICAR" : "APROVAÇÃO DO CEO"; auto.classList.toggle("live", s.autopublish);
  const llm = s.llm.providers.length ? s.llm.providers.map((p) => p.name).join(" → ") : "playbook interno (sem chave de IA)";
  $("scene-sub").textContent = `5 agentes · IA: ${llm} · ${s.llm.calls_today}/${s.llm.daily_budget} chamadas hoje`;
  $("upd").textContent = `atualizado ${new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })}`;
  for (const [r, st] of Object.entries(s.agents)) if (st.state && !ws?.readyState) applyState(r, st);

  const g = s.goals, gl = $("goals"); gl.replaceChildren();
  const goal = (label, cur, tgt, fmt, good) => {
    const w = el("div", "goal"); const top = el("div", "row-between");
    top.append(el("span", "muted-sm", label), el("span", "", `${fmt(cur)} / ${fmt(tgt)}`));
    const bar = el("div", `bar${good ? " good" : ""}`); const fill = el("span"); fill.style.width = `${Math.min(100, (cur / Math.max(tgt, 1e-9)) * 100)}%`;
    bar.append(fill); w.append(top, bar); gl.append(w);
  };
  goal("Receita líquida do mês", g.revenue.current, g.revenue.target, (v) => usd(v, 0), g.revenue.pct >= 1);
  goal("Produtos na semana", g.products_week.current, g.products_week.target, String, g.products_week.current >= g.products_week.target);
  goal("Ideias na semana", g.ideas_week.current, g.ideas_week.target, String, g.ideas_week.current >= g.ideas_week.target);
  const hint = el("div", "muted-xs", `Ritmo atual: ${usd(g.revenue.run_rate)} no mês · lucro do mês ${usd(g.profit_mtd)}`); gl.append(hint);
  $("g-be").textContent = g.break_even.monthly_cost_usd ? `ponto de equilíbrio ≈ ${g.break_even.sales_needed} vendas/mês` : "";

  const c = s.monthly_costs, box = $("costs"); box.replaceChildren();
  const line = (k, v, cls) => { const r = el("div", `row-between ${cls || ""}`); r.append(el("span", "", k), el("span", "", usd(v))); box.append(r); };
  line("Railway · web (FastAPI + agentes)", c.breakdown.web); line("Railway · PostgreSQL", c.breakdown.postgres);
  if (c.breakdown.redis) line("Railway · Redis", c.breakdown.redis);
  if (c.breakdown.n8n) line("Railway · n8n", c.breakdown.n8n);
  if (c.other_usd) line("Outros", c.other_usd);
  line("IAs (planos gratuitos)", c.llm_usd);
  line("Total por mês", c.railway_usd + c.other_usd, "total");
  box.append(el("div", "muted-xs", "Estimativas de ENV (RAILWAY_*_USD). Confira o consumo real em Railway → Usage. Custo acumulado por dia no P&L."));
}

function renderChart(rows) {
  const svg = $("chart-svg"), W = 340, H = 170, P = { l: 38, r: 6, t: 8, b: 20 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg", mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const max = Math.max(1, ...rows.map((r) => Math.max(r.revenue_usd, r.costs_usd)));
  const ih = H - P.t - P.b, iw = W - P.l - P.r, slot = iw / rows.length, bw = Math.max(1.5, (slot - 2) / 2 - 1);
  const y = (v) => P.t + ih - (v / max) * ih;
  for (const f of [0, 0.5, 1]) {
    svg.append(mk("line", { x1: P.l, x2: W - P.r, y1: y(max * f), y2: y(max * f), class: "grid" }));
    const t = mk("text", { x: P.l - 4, y: y(max * f) + 3, "text-anchor": "end", class: "axis" }); t.textContent = `$${(max * f).toFixed(max * f < 10 ? 2 : 0)}`; svg.append(t);
  }
  const bar = (x, v, color) => { const h = Math.max(0, (v / max) * ih); if (h <= 0) return; const r = Math.min(4, bw / 2, h);
    svg.append(mk("path", { d: `M${x},${P.t + ih} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z`, fill: color })); };
  const tip = $("tip");
  rows.forEach((r, i) => {
    const x0 = P.l + i * slot + 1;
    bar(x0, r.revenue_usd, "var(--series-rev)"); bar(x0 + bw + 2, r.costs_usd, "var(--series-cost)");
    const hit = mk("rect", { x: P.l + i * slot, y: P.t, width: slot, height: ih, class: "hit" });
    hit.addEventListener("mousemove", (e) => {
      tip.hidden = false; tip.replaceChildren(el("strong", "", r.day), el("br"), document.createTextNode(`Receita ${usd(r.revenue_usd)} · ${r.sales} vendas`), el("br"),
        document.createTextNode(`Custos ${usd(r.costs_usd)} · Lucro ${usd(r.profit_usd)}`));
      const box = $("chart").getBoundingClientRect();
      tip.style.left = `${Math.min(e.clientX - box.left + 10, box.width - 190)}px`; tip.style.top = `${e.clientY - box.top - 60}px`;
    });
    hit.addEventListener("mouseleave", () => { tip.hidden = true; });
    svg.append(hit);
  });
  [0, rows.length - 1].forEach((i) => { if (!rows[i]) return; const t = mk("text", { x: P.l + i * slot + slot / 2, y: H - 6, "text-anchor": i ? "end" : "start", class: "axis" }); t.textContent = rows[i].day.slice(5); svg.append(t); });
  const tbl = $("chart-table"); tbl.replaceChildren();
  const hr = el("tr"); ["Dia", "Vendas", "Receita", "Custos", "Lucro"].forEach((h) => hr.append(el("th", "", h))); tbl.append(hr);
  [...rows].reverse().forEach((r) => { const tr = el("tr"); tr.append(el("td", "", r.day), el("td", "num", String(r.sales)), el("td", "num", usd(r.revenue_usd)), el("td", "num", usd(r.costs_usd)), el("td", "num", usd(r.profit_usd))); tbl.append(tr); });
}

const feeds = {};
function appendFeed(agent, text, ts) {
  const ul = feeds[agent]; if (!ul) return;
  const li = el("li"); li.append(el("time", "", hhmm(ts || new Date().toISOString())), document.createTextNode(text));
  ul.prepend(li); while (ul.children.length > 20) ul.lastChild.remove();
}

function renderSectors(sec, summary, activity) {
  const root = $("sectors"); root.replaceChildren();
  const T = { product: "Pesquisa & validação", builder: "Construção & QA", growth: "Marketing & copy", sales: "Vendas & onboarding", cfo: "Financeiro & metas" };
  const mini = (pairs) => { const m = el("div", "mini"); for (const [k, v] of pairs) { const d = el("div"); d.append(el("span", "muted-sm", k), el("b", "", String(v))); m.append(d); } return m; };
  const pairs = {
    product: [["Sinais", Object.values(sec.product.signals_by_source).reduce((a, b) => a + b, 0)], ["Ideias", Object.values(sec.product.ideas_by_status).reduce((a, b) => a + b, 0)],
      ["🔥 Hair on fire", sec.product.hair_on_fire || 0], ["Descartadas", sec.product.discarded || 0]],
    builder: [["Construídos", sec.builder.built], ["QA ok", sec.builder.qa_pass], ["n8n", sec.builder.n8n], ["Python", sec.builder.python]],
    growth: [["Kits", sec.growth.kits], ["E-mails", sec.growth.emails], ["Posts", sec.growth.posts], ["Landing pages", sec.growth.kits]],
    sales: [["Visitas 14d", (sec.sales.funnel || []).reduce((t, f) => t + f.views, 0)], ["Cliques comprar", (sec.sales.funnel || []).reduce((t, f) => t + f.clicks, 0)],
      ["Visita→compra", (() => { const f = sec.sales.funnel || []; const v = f.reduce((t, x) => t + x.views, 0); const n = f.reduce((t, x) => t + x.sales, 0); return v ? `${(100 * n / v).toFixed(1)}%` : "—"; })()],
      ["Webhook Stripe", (sec.sales.setup || {}).stripe_webhook || "sem Stripe"], ["Apple/Google Pay", ((sec.sales.setup || {}).wallets || "—").replace("Apple Pay / Google Pay: ", "")],
      ["Anúncios ativos", sec.sales.listings_live], ["Manuais", sec.sales.listings_manual], ["Sem entrega", sec.sales.undelivered],
      ["Checkouts abandonados", (sec.sales.checkout || {}).abandoned || 0], ["E-mails de recuperação", (sec.sales.checkout || {}).recovery_emails || 0],
      ["Pagamentos recusados", (sec.sales.checkout || {}).failed || 0]],
    cfo: [["Lucro mês", usd(sec.cfo.profit_mtd)], ["Ritmo mês", usd(sec.cfo.revenue.run_rate)], ["Equilíbrio", `${sec.cfo.break_even.sales_needed} vendas`], ["IA hoje", `${sec.cfo.llm_calls_today}/${sec.cfo.llm_budget}`],
      ["Produtos parados 21d+", (sec.cfo.portfolio || []).filter((x) => ["reposition", "shutdown"].includes(x.health)).length],
      ["Margem 30d (top)", Object.values(sec.cfo.topic_margins || {}).length ? usd(Math.max(...Object.values(sec.cfo.topic_margins))) : "—"],
      ["Runway", !sec.cfo.runway ? "—" : !sec.cfo.runway.cash_set ? "aguardando o saldo do Stripe" : sec.cfo.runway.months !== null ? `${sec.cfo.runway.months} meses` : "sem queima"],
      ["Infra/mês", sec.cfo.infra && sec.cfo.infra.month_usd !== undefined ? `${usd(sec.cfo.infra.month_usd)}${sec.cfo.infra.measured ? " (medido)" : " (estimado)"}` : "—"],
      ["Austeridade", sec.cfo.infra && sec.cfo.infra.austerity ? "LIGADA" : "não"], ["LTV", usd((sec.cfo.ltv_cac || {}).ltv_usd || 0)],
      ["CAC", (sec.cfo.ltv_cac || {}).organic_only ? "orgânico ($0)" : usd((sec.cfo.ltv_cac || {}).cac_usd || 0)],
      ["Imposto+câmbio mês", usd(((sec.cfo.pnl_month || {}).tax_usd || 0) + ((sec.cfo.pnl_month || {}).fx_usd || 0))],
      ["Moeda de recebimento", `${((sec.cfo.params || {}).payout_currency || "usd").toUpperCase()} · ${(sec.cfo.params || {}).payout_source || ""}`],
      ["Imposto", `${(((sec.cfo.params || {}).tax_rate || 0) * 100).toFixed(1)}% · ${(sec.cfo.params || {}).tax_source || ""}`],
      ["Caixa", (sec.cfo.params || {}).cash_usd === null || (sec.cfo.params || {}).cash_usd === undefined ? "desconhecido (conecte o Stripe)" : `${usd(sec.cfo.params.cash_usd)} · ${sec.cfo.params.cash_source}`]],
  };
  for (const r of ROLES) {
    const a = AGENTS[r], info = summary.agents[r] || {};
    const c = el("section", "card sector"); c.style.setProperty("--c", a.color);
    const who = el("div", "who"); const av = el("span", "av", a.name[0]); const t = el("div");
    t.append(el("strong", "", `${a.name} · ${info.role || ""}`), el("span", "", `${ROOMS[r].label} — ${T[r]}`)); who.append(av, t);
    const now = el("div", "now", a.text || info.text || "Aguardando o próximo ciclo.");
    const ul = el("ul", "feed"); feeds[r] = ul;
    for (const e of activity.filter((x) => x.agent === r).slice(0, 8)) { const li = el("li"); li.append(el("time", "", hhmm(e.at)), document.createTextNode(e.text)); ul.append(li); }
    if (!ul.children.length) ul.append(el("li", "empty", "Sem atividade ainda."));
    const m = mini(pairs[r]); if (r === "sales" || r === "cfo") m.classList.add("one");
    c.append(who, now, m);
    if (r === "product") { const f = el("div", "focus", "Foco da semana: "); f.append(el("b", "", sec.product.focus.join(", ") || "—")); c.append(f); }
    c.append(ul); root.append(c);
  }
}

function renderKanban(ideas, products) {
  const cols = [["Ideias", ideas.filter((i) => i.status === "proposed")], ["Aprovadas", ideas.filter((i) => ["approved", "building"].includes(i.status))],
    ["Construídos", products.filter((p) => ["built", "marketing", "qa_failed"].includes(p.status))], ["Prontos (aguardando)", products.filter((p) => p.status === "ready")],
    ["Publicados", products.filter((p) => p.status === "published")]];
  const root = $("kanban"); root.replaceChildren();
  for (const [title, items] of cols) {
    const c = el("div", "kcol"); const h = el("h3"); h.append(el("span", "", title), el("span", "", String(items.length))); c.append(h);
    for (const it of items.slice(0, 8)) {
      const d = el("div", "item"); d.append(el("strong", "", it.title || it.name), el("span", "meta", `${it.type} · ${usd(it.price_usd, 0)}${it.assessment && URG[it.assessment.urgency] ? ` · ${URG[it.assessment.urgency][0]}` : ""}${it.score !== undefined ? ` · MVP ${Math.round(it.score)}` : ""}${it.sales ? ` · ${it.sales} vendas` : ""}`));
      c.append(d);
    }
    if (!items.length) c.append(el("div", "empty", "—"));
    root.append(c);
  }
}

function renderProducts(products, summary) {
  const root = $("plist"); root.replaceChildren();
  $("p-note").textContent = summary.autopublish ? "publicação automática ligada" : "produtos prontos aguardam sua aprovação (1 clique)";
  if (!products.length) { root.append(el("div", "empty", "O Theo ainda não construiu produtos. Eles aparecem aqui depois da primeira reunião de ideias.")); return; }
  for (const p of products) {
    const c = el("article", "pcard"); const top = el("div", "top"); top.append(el("h3", "", p.name), el("span", `badge b-${p.status}`, p.status)); c.append(top);
    c.append(el("p", "", p.headline || p.tagline), el("div", "muted-xs", `${p.type} · ${usd(p.price_usd, 0)} · ${p.size_kb} KB · ${p.sales} vendas · ${usd(p.revenue_usd)} · QA ${p.qa?.ok ? "ok" : "falhou"}`));
    const links = el("div", "links"); const lp = el("a", "", "Landing page"); lp.href = `/p/${p.slug}`; lp.target = "_blank"; links.append(lp);
    for (const l of p.listings) { if (l.url) { const a = el("a", "", `${l.platform} · ${l.status}`); a.href = l.url; a.target = "_blank"; a.rel = "noopener"; links.append(a); } else links.append(el("span", "", `${l.platform} · ${l.status}`)); }
    c.append(links);
    if (p.status === "ready" || p.status === "paused") {
      const b = el("button", "btn", "Aprovar e publicar");
      b.onclick = async () => { b.disabled = true; const r = await admin(`/api/products/${p.id}/approve`); if (r) toast(`Leo vai publicar ${p.name}`); else b.disabled = false; };
      c.append(b);
    }
    const det = el("details", "more"); det.append(el("summary", "", "Copy, e-mails e posts"));
    det.addEventListener("toggle", async () => {
      if (!det.open || det.dataset.loaded) return; det.dataset.loaded = "1";
      const d = await api(`/api/products/${p.id}`); det.append(el("pre", "", JSON.stringify(d.kit, null, 2)));
    });
    c.append(det); root.append(c);
  }
}

const URG = { hair_on_fire: ["🔥 Hair on fire", "u-fire"], important: ["Importante", "u-imp"], nice_to_have: ["Nice-to-have", "u-nice"] };

function renderIdeas(ideas) {
  const t = $("ideas"); t.replaceChildren();
  const cols = ["Ideia", "Urgência", "Opp. /20", "WTP /10", "TTV", "Fricção", "Paga?", "Moat", "Distrib.", "Valor/mês", "Preço", "MVP", "Evidências", "Status"];
  const hr = el("tr"); cols.forEach((h) => hr.append(el("th", "", h))); t.append(hr);
  if (!ideas.length) { const tr = el("tr"); const td = el("td", "empty", "Nenhuma ideia ainda: a primeira pesquisa roda quando o escritório liga."); td.colSpan = cols.length; tr.append(td); t.append(tr); return; }
  const n = (v, d = 1) => (v === undefined || v === null ? "—" : Number(v).toFixed(d));
  for (const i of ideas) {
    const a = i.assessment || {}, tr = el("tr"); const ev = el("td");
    for (const e of (i.evidence || []).slice(0, 3)) { if (e.url) { const l = el("a", "", e.source); l.href = e.url; l.target = "_blank"; l.rel = "noopener"; l.title = e.title; ev.append(l, document.createTextNode(" ")); } else ev.append(document.createTextNode(`${e.source} `)); }
    const name = el("td"); name.append(el("strong", "", i.title), el("br"), el("span", "muted-xs", a.job_to_be_done || i.problem));
    if (a.emotional_job) name.append(el("br"), el("span", "muted-xs", `❤ ${a.emotional_job}`));
    if (a.reasons && a.reasons.length && i.status === "rejected") name.append(el("br"), el("span", "muted-xs dec-rejected", a.reasons.join(" · ")));
    const [ul, uc] = URG[a.urgency] || ["—", ""];
    const ttv = el("td", `num ${a.ttv_minutes > 5 ? "dec-rejected" : ""}`, a.ttv_minutes ? `${a.ttv_minutes} min` : "—");
    const moat = el("td", `num ${a.commodity ? "dec-rejected" : ""}`, a.commodity ? "commodity" : n(a.moat, 0));
    const urg = el("td"); urg.append(el("span", `badge ${uc}`, ul));
    tr.append(name, urg, el("td", "num", n(a.opportunity)), el("td", "num", n(a.wtp)),
      ttv, el("td", `num ${a.friction > a.friction_tolerance ? "dec-rejected" : ""}`, a.friction !== undefined ? `${n(a.friction)}/${n(a.friction_tolerance)}` : "—"),
      el("td", `num ${a.payer_habit < -0.3 ? "dec-rejected" : ""}`, a.payer_habit !== undefined ? (a.payer_habit >= 0 ? "+" : "") + n(a.payer_habit, 2) : "—"),
      moat, el("td", "num", n(a.distribution, 0)),
      el("td", "num", a.value_month_low ? `${usd(a.value_month_low, 0)}–${usd(a.value_month_high, 0)}` : "—"),
      el("td", "num", usd(i.price_usd, 0)), el("td", "num", n(i.score, 0)), ev, el("td", `dec-${i.status}`, i.status));
    t.append(tr);
  }
}

function renderMeetings(ms) {
  const root = $("meetings"); root.replaceChildren();
  if (!ms.length) { root.append(el("div", "empty", "Nenhuma reunião ainda.")); return; }
  ms.forEach((m, idx) => {
    const d = el("details", "mtg"); if (idx === 0) d.open = true;
    const s = el("summary"); s.append(el("strong", "", m.subject), el("span", `dec-${m.decision}`, m.decision), el("span", "muted-xs", `${hhmm(m.started_at)} · ${m.summary}`));
    const ul = el("ul", "ata"); for (const x of m.messages) ul.append(ataItem(x.agent, x.text, x.vote, x.at));
    d.append(s, ul); root.append(d);
  });
  const ata = $("ata");
  if (ata.querySelector(".empty") && ms[0]) { ata.replaceChildren(); for (const x of [...ms[0].messages].reverse()) ata.append(ataItem(x.agent, x.text, x.vote, x.at)); }
}

async function refreshAll() {
  try {
    const [summary, daily, sec, products, ideas, meetings, activity] = await Promise.all([api("/api/dashboard/summary"), api("/api/dashboard/sales/daily?days=30"),
      api("/api/sectors"), api("/api/products"), api("/api/ideas?limit=60"), api("/api/meetings?limit=8"), api("/api/activity?limit=120")]);
    renderSummary(summary); renderChart(daily); renderSectors(sec, summary, activity); renderKanban(ideas, products);
    renderProducts(products, summary); renderIdeas(ideas); renderMeetings(meetings);
  } catch (e) { console.error(e); }
}

// ======================================================================= controls
let zoom = 1;
function applyZoom() { $("stage").style.transform = `scale(${zoom})`; $("stage").style.transformOrigin = "0 0"; $("z-label").textContent = `${Math.round(zoom * 100)}%`;
  $("scroller").style.height = `${760 * zoom}px`; }
$("z-in").onclick = () => { zoom = Math.min(1.6, zoom + 0.1); applyZoom(); };
$("z-out").onclick = () => { zoom = Math.max(0.5, zoom - 0.1); applyZoom(); };
$("btn-tick").onclick = async () => { const r = await admin("/api/run/tick"); if (r) toast("Ciclo iniciado: os agentes estão trabalhando"); };
const fit = () => { const w = $("scroller").clientWidth; zoom = Math.max(0.5, Math.min(1, w / 1260)); applyZoom(); };
window.addEventListener("resize", fit); fit();

connect();
refreshAll();
setInterval(refreshAll, 30000);
