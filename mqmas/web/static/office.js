/* MQ-MAS pixel-art office.
 *
 * Renders a 320x180 office on a canvas (scaled with nearest-neighbour) and
 * overlays DOM speech bubbles fed by the backend WebSocket (/ws).
 * Event types: hello, telemetry, agent_message, debate_start, decision,
 * position, risk, system — see README "Protocolo WebSocket" to plug the same
 * stream into another virtual office (e.g. a SoWork/WorkAdventure bridge).
 */
"use strict";

const W = 320, H = 180;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");

// ---------------------------------------------------------------- agents
const AGENTS = {
  chief_quant:    { name: "Ada",   title: "Chief Quant",  desk: { x: 62,  y: 78 },  seat: { x: 136, y: 104 }, hair: "#3b2a1a", skin: "#f1c27d", shirt: "#8a7dff", pants: "#2d2a4a" },
  econometrician: { name: "Gauss", title: "Econometrista", desk: { x: 258, y: 78 },  seat: { x: 184, y: 104 }, hair: "#d9d9d9", skin: "#e0ac69", shirt: "#3fb5a3", pants: "#263238" },
  cro:            { name: "Vera",  title: "CRO",          desk: { x: 62,  y: 150 }, seat: { x: 136, y: 140 }, hair: "#7a1f1f", skin: "#c68642", shirt: "#ff6b6b", pants: "#2b2b2b" },
  execution:      { name: "Hex",   title: "Execução",     desk: { x: 258, y: 150 }, seat: { x: 184, y: 140 }, hair: "#111111", skin: "#8d5524", shirt: "#ffc857", pants: "#1e3a5f" },
};
for (const a of Object.values(AGENTS)) {
  a.x = a.desk.x; a.y = a.desk.y - 14; a.tx = a.x; a.ty = a.y; a.vote = null; a.bubble = null;
}

const SPRITE = [
  "...hhhh...",
  "..hhhhhh..",
  "..hsssshh.",
  "..sesses..",
  "..ssssss..",
  "...ssss...",
  "..cccccc..",
  ".cccccccc.",
  ".sccccccs.",
  ".sccccccs.",
  "..pppppp..",
  "..pp..pp..",
  "..pp..pp..",
  "..kk..kk..",
];
const WALK = ["..pp...pp.", ".pp...pp..", ".kk....kk."];

function drawSprite(a, t) {
  const moving = Math.hypot(a.tx - a.x, a.ty - a.y) > 0.5;
  const bob = moving ? 0 : (Math.floor(t / 500) % 2);
  const rows = SPRITE.slice();
  if (moving && Math.floor(t / 150) % 2) { rows[11] = WALK[0]; rows[12] = WALK[1]; rows[13] = WALK[2]; }
  const pal = { h: a.hair, s: a.skin, e: "#1b1830", c: a.shirt, p: a.pants, k: "#111" };
  const ox = Math.round(a.x - 5), oy = Math.round(a.y - 14 + bob);
  ctx.fillStyle = "rgba(0,0,0,.25)";
  ctx.fillRect(ox + 1, Math.round(a.y) + 0, 8, 2);
  rows.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      if (ch === ".") return;
      ctx.fillStyle = pal[ch];
      ctx.fillRect(ox + x, oy + y, 1, 1);
    });
  });
}

// ---------------------------------------------------------------- scenery
const priceHistory = [];
let lastSnapshot = null;
let meeting = false;

function rect(c, x, y, w, h) { ctx.fillStyle = c; ctx.fillRect(x, y, w, h); }

function drawRoom(t) {
  // floor
  for (let y = 40; y < H; y += 8) for (let x = 0; x < W; x += 8)
    rect(((x + y) / 8) % 2 ? "#3a3350" : "#433b5c", x, y, 8, 8);
  // wall + skirting
  rect("#2a2440", 0, 0, W, 40); rect("#1b1730", 0, 38, W, 3);
  // windows with a slow day/night tint
  const sky = `hsl(${220 + 10 * Math.sin(t / 20000)}, 45%, ${18 + 6 * Math.sin(t / 15000)}%)`;
  for (const wx of [14, 262]) { rect("#14111f", wx - 1, 7, 46, 24); rect(sky, wx, 8, 44, 22); rect("#14111f", wx + 21, 8, 2, 22); }
  // wall screen with live price sparkline
  rect("#0d0b16", 98, 4, 124, 32); rect("#06140d", 100, 6, 120, 28);
  if (priceHistory.length > 1) {
    const min = Math.min(...priceHistory), max = Math.max(...priceHistory), span = max - min || 1;
    ctx.fillStyle = "#59d18c";
    priceHistory.forEach((p, i) => {
      const x = 101 + Math.round((i / (priceHistory.length - 1)) * 117);
      const y = 31 - Math.round(((p - min) / span) * 22);
      ctx.fillRect(x, y, 1, 1);
    });
  }
  if (lastSnapshot) {
    const z = lastSnapshot.zscore;
    rect(z <= -2 ? "#ff6b6b" : z <= -1.5 ? "#ffc857" : "#59d18c", 214, 8, 4, 4); // regime LED
  }
  // meeting table
  rect("#5b3d2a", 140, 112, 40, 22); rect("#7a5236", 141, 113, 38, 18);
  rect(meeting ? "#8a7dff" : "#3b2a1a", 158, 120, 4, 4);
  // plants
  for (const [px, py] of [[4, 44], [306, 44], [4, 166], [306, 166]]) {
    rect("#6b4a2e", px + 2, py + 6, 6, 5); rect("#2f8f4e", px, py, 10, 7); rect("#3fb36a", px + 2, py - 2, 6, 4);
  }
  // desks + monitors (monitor colour = last vote)
  for (const a of Object.values(AGENTS)) {
    const { x, y } = a.desk;
    rect("#6b4a2e", x - 16, y - 4, 32, 10); rect("#8a6040", x - 15, y - 3, 30, 3);
    const screen = a.vote === "APPROVE" ? "#59d18c" : a.vote === "REJECT" ? "#ff6b6b" : a.vote === "ABSTAIN" ? "#ffc857" : "#3a6ea5";
    rect("#111", x - 7, y - 13, 14, 9); rect(screen, x - 6, y - 12, 12, 7); rect("#111", x - 1, y - 4, 2, 1);
    if (Math.floor(t / 300) % 2) rect("#fff", x - 5, y - 11, 3, 1);
  }
}

// ---------------------------------------------------------------- overlay (tags + bubbles)
function pct(v, total) { return `${(v / total) * 100}%`; }

for (const [role, a] of Object.entries(AGENTS)) {
  a.tag = document.createElement("div");
  a.tag.className = "tag";
  a.tag.textContent = `${a.name} · ${a.title}`;
  overlay.appendChild(a.tag);
  a.role = role;
}

function placeOverlay() {
  for (const a of Object.values(AGENTS)) {
    a.tag.style.left = pct(a.x, W); a.tag.style.top = pct(a.y + 2, H);
    if (a.bubble) {
      // keep bubbles inside the office horizontally
      const x = Math.min(Math.max(a.x, 56), W - 56);
      a.bubble.style.left = pct(x, W); a.bubble.style.top = pct(a.y - 18, H);
    }
  }
}

function speak(role, name, text, vote) {
  const a = AGENTS[role];
  if (!a) return;
  if (a.bubble) a.bubble.remove();
  clearTimeout(a.bubbleTimer); clearInterval(a.typeTimer);
  const b = document.createElement("div");
  b.className = "bubble" + (vote ? ` vote-${vote}` : "");
  const who = document.createElement("span"); who.className = "who"; who.textContent = `${name}: `;
  const body = document.createElement("span");
  b.append(who, body);
  overlay.appendChild(b);
  a.bubble = b;
  let i = 0;
  a.typeTimer = setInterval(() => { body.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(a.typeTimer); }, 14);
  a.bubbleTimer = setTimeout(() => { b.classList.add("fade"); setTimeout(() => { b.remove(); if (a.bubble === b) a.bubble = null; }, 400); },
                             Math.min(6000 + text.length * 45, 16000));
  placeOverlay();
}

// ---------------------------------------------------------------- side panel
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });

function setText(id, text, cls) { const el = $(id); el.textContent = text; el.className = cls || ""; }

function onTelemetry(p) {
  const s = p.snapshot; lastSnapshot = s;
  if (p.spark && p.spark.length) { priceHistory.length = 0; priceHistory.push(...p.spark.slice(0, -1), s.price); }
  setText("m-price", fmt(s.price));
  setText("m-z", fmt(s.zscore), s.zscore <= -2 ? "neg" : s.zscore >= 2 ? "pos" : "");
  setText("m-vwap", fmt(s.vwap));
  setText("m-poc", fmt(s.volume_profile.poc));
  setText("m-imb", (s.orderbook.imbalance >= 0 ? "+" : "") + fmt(s.orderbook.imbalance, 3), s.orderbook.imbalance >= 0 ? "pos" : "neg");
  setText("m-atr", fmt(s.atr_pct * 100, 3) + "%");
  setText("m-fc", `${fmt(s.forecast.expected_price)} [${fmt(s.forecast.lower_95, 0)} – ${fmt(s.forecast.upper_95, 0)}]`);
  setText("m-eq", "US$ " + fmt(p.equity));
  $("pill-mode").textContent = p.mode.toUpperCase();
  $("pill-mode").className = "pill " + (p.mode === "live" ? "pill-bad" : "pill-warn");
  $("pill-lock").hidden = !p.lock_until;
}

function log(name, text, meta) {
  const li = document.createElement("li");
  const m = document.createElement("div"); m.className = "meta"; m.textContent = meta;
  const t = document.createElement("div"); t.textContent = `${name}: ${text}`;
  li.append(m, t);
  const ol = $("log"); ol.prepend(li);
  while (ol.children.length > 80) ol.lastChild.remove();
}

function renderVotes(votes) {
  const box = $("votes"); box.replaceChildren();
  for (const [role, v] of Object.entries(votes || {})) {
    const el = document.createElement("span");
    el.className = `vote ${v}`; el.textContent = `${AGENTS[role]?.name || role}: ${v}`;
    box.appendChild(el);
  }
}

function setMeeting(on) {
  meeting = on;
  for (const a of Object.values(AGENTS)) {
    const target = on ? a.seat : { x: a.desk.x, y: a.desk.y - 14 };
    a.tx = target.x; a.ty = target.y;
    if (on) a.vote = null;
  }
}

// ---------------------------------------------------------------- websocket
function handle(ev) {
  const p = ev.payload, time = new Date(ev.ts).toLocaleTimeString("pt-BR");
  switch (ev.type) {
    case "hello":
      $("pill-mode").textContent = p.mode.toUpperCase(); break;
    case "telemetry":
      onTelemetry(p); break;
    case "agent_message":
      if (p.role === "system") { log(p.agent_name, p.text, time); break; }
      if (p.vote && AGENTS[p.role]) AGENTS[p.role].vote = p.vote;
      speak(p.role, p.agent_name, p.text, p.vote);
      if (p.debate_id || p.provider !== "rules")
        log(p.agent_name, p.text, `${time} · ${p.vote || "fala"} · ${p.provider || ""}`);
      break;
    case "debate_start":
      setMeeting(true); renderVotes({}); break;
    case "decision":
      renderVotes(p.votes);
      setTimeout(() => setMeeting(false), 9000); break;
    case "position": {
      const pos = p.position;
      $("m-pos").textContent = pos.status === "open" ? `${fmt(pos.quantity, 5)} @ ${fmt(pos.entry_price)}` : "—";
      if (p.event === "closed") log("Mesa", `Posição encerrada (${pos.exit_reason}) PnL US$ ${fmt(pos.pnl_usdt, 4)}`, time);
      break;
    }
    case "risk":
      log("Risco", JSON.stringify(p), time);
      if (p.event === "circuit_breaker") $("pill-lock").hidden = false;
      break;
  }
}

let backoff = 1000;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  const conn = $("pill-conn");
  ws.onopen = () => { backoff = 1000; conn.textContent = "online"; conn.className = "pill pill-good"; };
  ws.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) { console.error(e); } };
  ws.onclose = () => {
    conn.textContent = "reconectando…"; conn.className = "pill pill-bad";
    setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 15000);
  };
  const ping = setInterval(() => { if (ws.readyState === 1) ws.send("ping"); else clearInterval(ping); }, 20000);
}

// ---------------------------------------------------------------- main loop
function frame(t) {
  for (const a of Object.values(AGENTS)) {
    a.x += Math.sign(a.tx - a.x) * Math.min(Math.abs(a.tx - a.x), 0.6);
    a.y += Math.sign(a.ty - a.y) * Math.min(Math.abs(a.ty - a.y), 0.6);
  }
  ctx.clearRect(0, 0, W, H);
  drawRoom(t);
  Object.values(AGENTS).sort((a, b) => a.y - b.y).forEach((a) => drawSprite(a, t));
  placeOverlay();
  requestAnimationFrame(frame);
}

connect();
requestAnimationFrame(frame);
