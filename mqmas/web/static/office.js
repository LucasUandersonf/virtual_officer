/* MQ-MAS virtual office — isometric 2.5D scene in a cosy "virtual HQ" style.
 *
 * Everything is drawn procedurally on a canvas (no external assets): the
 * "HQ de Vidro" scene (glass facades, LED trading wall + stage, bleachers,
 * glass focus rooms, standing desks, Fuel Café, game zone) and four chibi
 * agents with floating portrait cards.
 * Speech bubbles are DOM elements positioned over the canvas.
 *
 * WebSocket events: hello, telemetry, agent_message, debate_start, decision,
 * position, risk, system — see README "Protocolo WebSocket".
 */
"use strict";

const W = 1280, H = 720;
const TW = 64, TH = 32;            // tile size (2:1 isometric)
const N = 13;                      // room is N x N tiles
const OX = 640, OY = 168;          // screen position of the back corner
const WALL = 170;                  // wall height in px

const canvas = document.getElementById("canvas");
const mainCtx = canvas.getContext("2d");
let ctx = mainCtx;  // helpers draw into `ctx`; paintStatic() points it at the background layer
const overlay = document.getElementById("overlay");
const DPR = Math.min(window.devicePixelRatio || 1, 2);
canvas.width = W * DPR; canvas.height = H * DPR;

const iso = (gx, gy, z = 0) => [OX + (gx - gy) * TW / 2, OY + (gx + gy) * TH / 2 - z];

function poly(pts, fill, stroke) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
}

function shade(hex, f) {
  const n = parseInt(hex.slice(1), 16);
  const c = (v) => Math.max(0, Math.min(255, Math.round(v * f)));
  return `rgb(${c(n >> 16)},${c((n >> 8) & 255)},${c(n & 255)})`;
}

/** Isometric box: (gx,gy) back corner, w along x, d along y, h height, z0 elevation. */
function box(gx, gy, w, d, h, color, z0 = 0) {
  const t = z0 + h;
  poly([iso(gx, gy + d, z0), iso(gx + w, gy + d, z0), iso(gx + w, gy + d, t), iso(gx, gy + d, t)], shade(color, 0.78));
  poly([iso(gx + w, gy, z0), iso(gx + w, gy + d, z0), iso(gx + w, gy + d, t), iso(gx + w, gy, t)], shade(color, 0.62));
  poly([iso(gx, gy, t), iso(gx + w, gy, t), iso(gx + w, gy + d, t), iso(gx, gy + d, t)], color);
}

function ellipse(x, y, rx, ry, fill) {
  ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill();
}

function roundRect(x, y, w, h, r, fill, stroke) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
}

// ===================================================================== agents
const AGENTS = {
  chief_quant:    { name: "Ada",   title: "Chief Quant",   home: [1.3, 4.75],  seat: [9.8, 0.95], skin: "#f2c9a0", hair: "#5a3825", style: "long",  shirt: "#8a7dff", pants: "#2f2a55", bg: "#dcd6ff" },
  econometrician: { name: "Gauss", title: "Econometrista", home: [2.7, 4.75],  seat: [8.5, 0.9],  skin: "#e8b48a", hair: "#e6e6e6", style: "short", shirt: "#2fb59a", pants: "#24323d", bg: "#cdf1e8", glasses: true },
  cro:            { name: "Vera",  title: "CRO",           home: [4.1, 4.75],  seat: [11.1, 1.1], skin: "#c98e62", hair: "#b8323a", style: "bob",   shirt: "#ff6b6b", pants: "#3a2a2a", bg: "#ffd9d6" },
  execution:      { name: "Hex",   title: "Execução",      home: [3.0, 9.0],   seat: [7.3, 1.1],  skin: "#8d5a3b", hair: "#1c1c1c", style: "cap",   shirt: "#ffc04d", pants: "#1e3a5f", bg: "#ffeec6" },
};
for (const [role, a] of Object.entries(AGENTS)) {
  a.role = role; a.gx = a.home[0]; a.gy = a.home[1]; a.tx = a.gx; a.ty = a.gy;
  a.vote = null; a.talkingUntil = 0; a.bubble = null; a.facing = 1; a.path = [];
  a.portrait = makePortrait(a);
}

function drawHair(c, a, x, y, r, back) {
  c.fillStyle = a.hair;
  if (back) {
    if (a.style === "long") { c.beginPath(); c.roundRect(x - r * 1.05, y - r * 0.2, r * 2.1, r * 1.9, r * 0.6); c.fill(); }
    if (a.style === "bob") { c.beginPath(); c.roundRect(x - r * 1.1, y - r * 0.3, r * 2.2, r * 1.35, r * 0.5); c.fill(); }
    return;
  }
  c.beginPath(); c.arc(x, y - r * 0.15, r * 1.03, Math.PI * 1.02, Math.PI * 1.98); c.fill();
  if (a.style === "long" || a.style === "bob") {
    c.beginPath(); c.ellipse(x - r * 0.35, y - r * 0.55, r * 0.75, r * 0.38, -0.3, 0, Math.PI * 2); c.fill();
  }
  if (a.style === "short") {
    c.beginPath(); c.ellipse(x, y - r * 0.75, r * 0.95, r * 0.4, 0, 0, Math.PI * 2); c.fill();
  }
  if (a.style === "cap") {
    c.fillStyle = "#2b59c3";
    c.beginPath(); c.arc(x, y - r * 0.25, r * 1.04, Math.PI, Math.PI * 2); c.fill();
    c.beginPath(); c.ellipse(x + r * 0.7, y - r * 0.28, r * 0.75, r * 0.2, 0, 0, Math.PI * 2); c.fill();
  }
}

function drawFace(c, a, x, y, r, blink) {
  c.fillStyle = "#2a2233";
  if (blink) { c.fillRect(x - r * 0.45, y + r * 0.08, r * 0.22, 2); c.fillRect(x + r * 0.23, y + r * 0.08, r * 0.22, 2); }
  else { c.beginPath(); c.arc(x - r * 0.34, y + r * 0.1, r * 0.11, 0, 7); c.arc(x + r * 0.34, y + r * 0.1, r * 0.11, 0, 7); c.fill(); }
  c.fillStyle = "rgba(255,120,120,.35)";
  c.beginPath(); c.arc(x - r * 0.55, y + r * 0.38, r * 0.15, 0, 7); c.arc(x + r * 0.55, y + r * 0.38, r * 0.15, 0, 7); c.fill();
  c.strokeStyle = "#7a3b3b"; c.lineWidth = Math.max(1.2, r * 0.07);
  c.beginPath(); c.arc(x, y + r * 0.35, r * 0.22, 0.15 * Math.PI, 0.85 * Math.PI); c.stroke();
  if (a.glasses) {
    c.strokeStyle = "#333"; c.lineWidth = Math.max(1.2, r * 0.06);
    c.beginPath(); c.arc(x - r * 0.34, y + r * 0.1, r * 0.22, 0, 7); c.moveTo(x + r * 0.56, y + r * 0.1);
    c.arc(x + r * 0.34, y + r * 0.1, r * 0.22, 0, 7); c.moveTo(x - r * 0.12, y + r * 0.1); c.lineTo(x + r * 0.12, y + r * 0.1); c.stroke();
  }
}

/** Off-screen "webcam" portrait used on the floating card. */
function makePortrait(a) {
  const s = 2, w = 60 * s, h = 46 * s;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, h); grad.addColorStop(0, a.bg); grad.addColorStop(1, "#ffffff");
  g.fillStyle = grad; g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(0,0,0,.06)"; g.fillRect(w * 0.62, h * 0.12, w * 0.3, h * 0.42); // shelf/frame in the back
  g.fillStyle = a.shirt; g.beginPath(); g.ellipse(w / 2, h * 1.08, w * 0.36, h * 0.38, 0, 0, 7); g.fill();
  g.fillStyle = a.skin; g.fillRect(w / 2 - 6 * s, h * 0.6, 12 * s, 8 * s);
  const r = 13 * s, hx = w / 2, hy = h * 0.47;
  drawHair(g, a, hx, hy, r, true);
  g.fillStyle = a.skin; g.beginPath(); g.arc(hx, hy, r, 0, 7); g.fill();
  drawHair(g, a, hx, hy, r, false);
  drawFace(g, a, hx, hy, r, false);
  return c;
}

const onStage = (a) => a.gx > STAGE.gx0 && a.gx < STAGE.gx1 && a.gy < STAGE.gy1;

function drawAgent(a, t) {
  const [x, y] = iso(a.gx, a.gy, onStage(a) ? STAGE.h : 0);
  const moving = Math.hypot(a.tx - a.gx, a.ty - a.gy) > 0.02;
  const step = moving ? Math.sin(t / 90) : 0;
  const bob = moving ? Math.abs(step) * 2 : Math.sin(t / 600 + a.gx) * 1.2;
  ellipse(x, y, 15, 6, "rgba(0,0,0,.22)");
  // legs
  roundRect(x - 8, y - 16 + step * 2, 7, 15 - step * 2, 3, a.pants);
  roundRect(x + 1, y - 16 - step * 2, 7, 15 + step * 2, 3, a.pants);
  ctx.fillStyle = "#2a2233"; ctx.fillRect(x - 9, y - 3 + step * 2, 9, 3); ctx.fillRect(x + 1, y - 3 - step * 2, 9, 3);
  // body + arms
  const by = y - 36 - bob;
  roundRect(x - 13, by, 26, 24, 9, a.shirt);
  roundRect(x - 17, by + 3 + step * 2, 6, 17, 3, shade(a.shirt, 0.85));
  roundRect(x + 11, by + 3 - step * 2, 6, 17, 3, shade(a.shirt, 0.85));
  // head
  const hx = x, hy = by - 15, r = 17;
  drawHair(ctx, a, hx, hy, r, true);
  ellipse(hx, hy, r, r, a.skin);
  drawHair(ctx, a, hx, hy, r, false);
  drawFace(ctx, a, hx, hy, r, (t + a.gx * 900) % 4200 < 140);
  a.sx = x; a.sy = hy - r;  // anchor for card + bubble
}

function drawCard(a, t) {
  const cw = 64, ch = 52, x = a.sx - cw / 2, y = a.sy - ch - 16;
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,.35)"; ctx.shadowBlur = 10; ctx.shadowOffsetY = 3;
  roundRect(x, y, cw, ch, 8, "#ffffff");
  ctx.restore();
  ctx.save(); ctx.beginPath(); ctx.roundRect(x + 3, y + 3, cw - 6, ch - 6, 6); ctx.clip();
  ctx.drawImage(a.portrait, x + 3, y + 3, cw - 6, ch - 6); ctx.restore();
  // pointer
  poly([[a.sx - 6, y + ch], [a.sx + 6, y + ch], [a.sx, y + ch + 7]], "#ffffff");
  // name pill
  ctx.font = "700 11px Nunito, system-ui, sans-serif"; ctx.textAlign = "center";
  const label = a.name, lw = ctx.measureText(label).width + 12;
  roundRect(a.sx - lw / 2, y + ch - 13, lw, 15, 7, "rgba(30,26,50,.85)");
  ctx.fillStyle = "#fff"; ctx.fillText(label, a.sx, y + ch - 2);
  // status chip: typing dots while talking, else vote badge
  const cx = x + cw + 2, cy = y + 4;
  if (t < a.talkingUntil) {
    roundRect(cx, cy, 24, 16, 6, "#1e1a32");
    for (let i = 0; i < 3; i++) {
      const up = Math.sin(t / 150 - i) > 0.3 ? 2 : 0;
      ellipse(cx + 6 + i * 6, cy + 8 - up, 2, 2, "#fff");
    }
  } else if (a.vote) {
    const col = a.vote === "APPROVE" ? "#22a35a" : a.vote === "REJECT" ? "#e0443e" : "#f0a92b";
    const sym = a.vote === "APPROVE" ? "✓" : a.vote === "REJECT" ? "✕" : "?";
    roundRect(cx, cy, 18, 18, 6, col);
    ctx.fillStyle = "#fff"; ctx.font = "800 12px Nunito, system-ui"; ctx.fillText(sym, cx + 9, cy + 13);
  }
}

// ===================================================================== scenery: "HQ de Vidro"
// Glass HQ: floor-to-ceiling glass with the city behind, an LED trading wall
// with a stage for the debates, bleachers for all-hands, glass focus rooms,
// a standing-desk row, the Fuel Café and a game zone.
const priceHistory = [];
let lastSnapshot = null;
let meeting = false;

const BLUE = "#2F6BFF", AMBER = "#FFB547", CORAL = "#FF6474", MINT = "#33D08A";
const INK = "#1A1D24", WOODHQ = "#C9A27A", EPOXY = "#E4E7EC";
const AISLE_GX = 5.6;                 // agents walk along the blue wayfinding strip
const STAGE = { gx0: 6.3, gx1: 12.4, gy0: 0.12, gy1: 1.9, h: 6 };
const bg = document.createElement("canvas"); bg.width = W * DPR; bg.height = H * DPR;

const quadR = (u0, u1, z0, z1, fill, stroke) => poly([iso(u0, 0, z0), iso(u1, 0, z0), iso(u1, 0, z1), iso(u0, 0, z1)], fill, stroke);
const quadL = (v0, v1, z0, z1, fill, stroke) => poly([iso(0, v0, z0), iso(0, v1, z0), iso(0, v1, z1), iso(0, v0, z1)], fill, stroke);
const floorQuad = (gx, gy, w, d, fill, z = 0) => poly([iso(gx, gy, z), iso(gx + w, gy, z), iso(gx + w, gy + d, z), iso(gx, gy + d, z)], fill);

/** Text laid on a plane: "R" = right wall (gy=0), "L" = left wall (gx=0), "F" = floor. */
function planeText(c, plane, at, text, font, color) {
  const [x, y] = at;
  c.save();
  const m = plane === "R" ? [1, 0.5, 0, 1] : plane === "L" ? [1, -0.5, 0, 1] : [1, 0.5, -1, 0.5];
  c.setTransform(DPR * m[0], DPR * m[1], DPR * m[2], DPR * m[3], x * DPR, y * DPR);
  c.font = font; c.fillStyle = color; c.textAlign = "left"; c.textBaseline = "middle"; c.fillText(text, 0, 0);
  c.restore();
}

/** Glass partition on plane gy=c (axis "x") or gx=c (axis "y"), from a to b. */
function glassPane(axis, c, a, b, h) {
  const pts = axis === "x" ? [iso(a, c, 0), iso(b, c, 0), iso(b, c, h), iso(a, c, h)]
                           : [iso(c, a, 0), iso(c, b, 0), iso(c, b, h), iso(c, a, h)];
  poly(pts, "rgba(190,215,255,.24)", "rgba(255,255,255,.9)");
  const band = axis === "x" ? [iso(a, c, 30), iso(b, c, 30), iso(b, c, 42), iso(a, c, 42)]
                            : [iso(c, a, 30), iso(c, b, 30), iso(c, b, 42), iso(c, a, 42)];
  poly(band, "rgba(47,107,255,.30)");
}

function skyline(quad, len) {
  // daylight city seen through the glass: sky, haze, two layers of towers
  quad(0, len, 0, WALL, "#CFE4FA");
  quad(0, len, 0, 40, "#E2EEFA");
  const far = [[0.2, 0.9, 92], [1.1, 1.6, 120], [1.8, 2.7, 80], [3.0, 3.5, 132], [3.8, 4.7, 98], [5.0, 5.4, 70],
               [5.7, 6.6, 116], [6.9, 7.4, 86], [7.7, 8.6, 128], [8.9, 9.6, 76], [9.9, 10.4, 110], [10.7, 11.6, 94], [11.9, 12.8, 122]];
  for (const [a, b, h] of far) if (a < len) quad(a, Math.min(b, len), 0, h, "#B9CDE3");
  const near = [[0.5, 1.4, 64], [2.2, 3.0, 74], [4.2, 5.1, 58], [6.0, 6.8, 70], [8.0, 8.9, 62], [10.1, 11.0, 72], [11.6, 12.6, 56]];
  for (const [a, b, h] of near) if (a < len) quad(a, Math.min(b, len), 0, h, "#A3BAD4");
  // reflections
  for (let u = 0.6; u < len; u += 3.1) quad(u, Math.min(u + 0.5, len), 20, WALL - 20, "rgba(255,255,255,.22)");
  // mullions + sill + head
  for (let u = 0; u <= len + 0.01; u += 2) quad(Math.max(u - 0.04, 0), Math.min(u + 0.04, len), 0, WALL, "#FFFFFF");
  quad(0, len, 0, 6, "#FFFFFF"); quad(0, len, WALL - 8, WALL, "#FFFFFF");
}

function paintStatic() {
  const b = bg.getContext("2d"); b.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx = b;
  const g = b.createLinearGradient(0, 0, 0, H); g.addColorStop(0, "#1d2336"); g.addColorStop(1, "#12151f");
  b.fillStyle = g; b.fillRect(0, 0, W, H);

  // epoxy floor with a soft grid and sheen
  floorQuad(0, 0, N, N, EPOXY);
  b.strokeStyle = "rgba(150,160,180,.35)"; b.lineWidth = 1;
  for (let k = 2.5; k < N; k += 2.5) {
    let [x1, y1] = iso(k, 0), [x2, y2] = iso(k, N); b.beginPath(); b.moveTo(x1, y1); b.lineTo(x2, y2); b.stroke();
    [x1, y1] = iso(0, k); [x2, y2] = iso(N, k); b.beginPath(); b.moveTo(x1, y1); b.lineTo(x2, y2); b.stroke();
  }
  floorQuad(0.4, 0, 1.4, N, "rgba(255,255,255,.18)"); floorQuad(0, 0.4, N, 1.2, "rgba(255,255,255,.16)");
  poly([iso(0, N), iso(N, N), iso(N, N, -14), iso(0, N, -14)], "#BEC4CF");
  poly([iso(N, 0), iso(N, N), iso(N, N, -14), iso(N, 0, -14)], "#A6ADBA");

  // glass facades with the city behind
  skyline(quadL, N);
  skyline(quadR, N);
  poly([iso(0, 0, WALL), iso(0, N, WALL), iso(-0.2, N, WALL), iso(-0.2, -0.2, WALL)], "#FFFFFF");
  poly([iso(0, 0, WALL), iso(N, 0, WALL), iso(N, -0.2, WALL), iso(-0.2, -0.2, WALL)], "#F1F3F7");
  // sunlight through the glass
  floorQuad(0.1, 0.1, 6, 2.2, "rgba(255,255,255,.28)");

  // Fuel Café menu board on the left facade
  quadL(7.4, 11.8, 74, 122, "#1A1D24");
  planeText(b, "L", iso(0, 11.5, 108), "FUEL", "800 22px Manrope, Nunito, system-ui", AMBER);
  planeText(b, "L", iso(0, 11.5, 90), "FLAT WHITE · COLD BREW", "700 10px Manrope, Nunito, system-ui", "#E4E7EC");
  planeText(b, "L", iso(0, 11.5, 80), "KOMBUCHA · MATCHA", "700 10px Manrope, Nunito, system-ui", "#E4E7EC");

  // wayfinding strip (the agents' walking route) and zone labels
  floorQuad(AISLE_GX - 0.25, 0, 0.5, N, "rgba(47,107,255,.20)");
  planeText(b, "F", iso(0.5, 6.0), "STANDING DESKS", "800 12px Manrope, Nunito, system-ui", "#3B4560");
  planeText(b, "F", iso(6.9, 5.6), "ALL-HANDS · TRADING FLOOR", "800 12px Manrope, Nunito, system-ui", BLUE);
  planeText(b, "F", iso(0.6, 12.6), "FUEL CAFÉ", "800 12px Manrope, Nunito, system-ui", "#9A6A2E");
  planeText(b, "F", iso(7.2, 12.6), "GAME ZONE", "800 12px Manrope, Nunito, system-ui", "#3B4560");

  // glass focus rooms (Alpha / Beta) against the back facade
  floorQuad(0.4, 0.4, 2.3, 2.4, "#D4DCEB"); floorQuad(2.9, 0.4, 2.3, 2.4, "#D4DCEB");
  quadR(0.9, 2.1, 78, 104, "#151A24"); quadR(3.4, 4.6, 78, 104, "#151A24");
  poly([iso(1.0, 0, 90), iso(1.5, 0, 96), iso(2.0, 0, 86)], null, BLUE);
  box(1.1, 1.1, 1.2, 0.9, 3, "#FFFFFF", 22); box(1.6, 1.45, 0.15, 0.15, 22, INK);
  box(0.75, 1.3, 0.35, 0.35, 16, AMBER); box(2.35, 1.3, 0.35, 0.35, 16, BLUE); box(1.5, 2.15, 0.35, 0.35, 16, BLUE);
  const [rx, ry] = iso(4.05, 1.5, 22); ctx.fillStyle = INK; ctx.fillRect(rx - 3, ry, 6, 22);
  ellipse(rx, ry, 30, 15, "#FFFFFF"); box(3.3, 1.2, 0.35, 0.35, 16, MINT); box(4.5, 1.6, 0.35, 0.35, 16, AMBER); box(3.9, 2.2, 0.35, 0.35, 16, CORAL);
  glassPane("y", 2.7, 0.4, 2.8, 84); glassPane("y", 5.2, 0.4, 2.8, 84);
  glassPane("x", 2.8, 0.4, 2.7, 84); glassPane("x", 2.8, 2.9, 5.2, 84);
  planeText(b, "R", iso(1.0, 2.8, 60), "SALA ALPHA", "800 10px Manrope, Nunito, system-ui", BLUE);
  planeText(b, "R", iso(3.5, 2.8, 60), "SALA BETA", "800 10px Manrope, Nunito, system-ui", BLUE);

  // stage in front of the LED wall
  box(STAGE.gx0, STAGE.gy0, STAGE.gx1 - STAGE.gx0, STAGE.gy1 - STAGE.gy0, STAGE.h, "#2A2F3B");
  poly([iso(STAGE.gx0, STAGE.gy1, 0), iso(STAGE.gx1, STAGE.gy1, 0), iso(STAGE.gx1, STAGE.gy1, 2), iso(STAGE.gx0, STAGE.gy1, 2)], BLUE);

  // bleachers (back to front, each step taller)
  const cushions = [[[7.2, BLUE], [10.0, AMBER]], [[8.3, CORAL], [10.9, MINT]], [[7.6, AMBER], [10.4, BLUE]]];
  [[2.6, 10], [3.3, 20], [4.0, 30]].forEach(([gy, h], i) => {
    box(6.8, gy, 5.0, 0.7, h, WOODHQ);
    for (const [gx, col] of cushions[i]) box(gx, gy + 0.12, 1.2, 0.46, 4, col, h);
  });
  ctx = mainCtx;
}

/** LED trading wall on the right facade, fed by live telemetry. */
function drawLedWall(t) {
  const u0 = 6.3, u1 = 12.3, z0 = 36, z1 = 150;
  quadR(u0 - 0.1, u1 + 0.1, z0 - 5, z1 + 5, "#1F2430");
  quadR(u0, u1, z0, z1, "#0E1118");
  const s = lastSnapshot;
  planeText(ctx, "R", iso(u0 + 0.2, 0, z1 - 12), "MQ-MAS · TRADING FLOOR", "800 9px Manrope, Nunito, system-ui", "#7FA6FF");
  if (s) {
    planeText(ctx, "R", iso(u0 + 0.2, 0, z1 - 32), Math.round(s.price).toLocaleString("pt-BR"), "700 18px 'JetBrains Mono', monospace", "#FFFFFF");
    planeText(ctx, "R", iso(u0 + 0.2, 0, z1 - 50), `Z ${s.zscore.toFixed(2)}σ`, "700 11px 'JetBrains Mono', monospace",
              s.zscore <= -2 ? "#FF8A95" : s.zscore <= -1.5 ? AMBER : "#7EE2B0");
  }
  // agent vote chips
  Object.values(AGENTS).forEach((a, i) => {
    const cu = u0 + 0.2 + (i % 2) * 0.95, cz = z0 + 30 - Math.floor(i / 2) * 16;
    const col = a.vote === "APPROVE" ? "#1F7A4D" : a.vote === "REJECT" ? "#B42335" : a.vote === "ABSTAIN" ? "#9A6A10" : "#3B4560";
    quadR(cu, cu + 0.85, cz - 6, cz + 6, col);
    planeText(ctx, "R", iso(cu + 0.08, 0, cz), a.name.toUpperCase(), "800 7px Manrope, Nunito, system-ui", "#FFFFFF");
  });
  // price chart with VWAP reference
  const c0 = u0 + 2.4, c1 = u1 - 0.2, cz0 = z0 + 10, cz1 = z1 - 30;
  if (priceHistory.length > 1) {
    const vals = s ? [...priceHistory, s.vwap] : priceHistory;
    const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
    const zOf = (p) => cz0 + ((p - min) / span) * (cz1 - cz0);
    if (s) {
      const [x1, y1] = iso(c0, 0, zOf(s.vwap)), [x2, y2] = iso(c1, 0, zOf(s.vwap));
      ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = AMBER; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
    }
    ctx.beginPath();
    priceHistory.forEach((p, i) => {
      const [x, y] = iso(c0 + (i / (priceHistory.length - 1)) * (c1 - c0), 0, zOf(p));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = MINT; ctx.lineWidth = 2; ctx.stroke();
  }
  if (meeting && Math.floor(t / 600) % 2) {
    const [x, y] = iso(u1 - 0.3, 0, z1 - 10); ellipse(x, y, 3, 3, CORAL);
  }
}

function plant(gx, gy, s = 1) {
  box(gx - 0.3, gy - 0.3, 0.6, 0.6, 18 * s, "#FFFFFF");
  const [x, y] = iso(gx, gy, 18 * s);
  for (const [dx, dy, r, c] of [[0, -16, 13, "#2F8B57"], [-10, -7, 11, "#4DB079"], [10, -8, 11, "#3E9F68"], [2, -28, 10, "#5CC287"]])
    ellipse(x + dx * s, y + dy * s, r * s, r * s * 0.85, c);
}

function pouf(gx, gy, color) {
  const [x, y] = iso(gx, gy);
  ellipse(x, y - 6, 19, 10, shade(color, 0.72)); ctx.fillStyle = shade(color, 0.72); ctx.fillRect(x - 19, y - 16, 38, 10);
  ellipse(x, y - 16, 19, 10, color);
}

// furniture: [depthKey, drawFn]
function furniture(t) {
  const items = [];
  const add = (gx, gy, w, d, fn) => items.push({ depth: gx + w / 2 + gy + d / 2, fn });

  // standing-desk row: one segment per agent, monitor colour = that agent's last vote
  const seg = [[0.6, "chief_quant"], [2.0, "econometrician"], [3.4, "cro"]];
  for (const [gx, role] of seg) {
    add(gx, 3.5, 1.4, 0.7, () => {
      box(gx + 0.05, 3.55, 0.1, 0.6, 34, INK); box(gx + 1.25, 3.55, 0.1, 0.6, 34, INK);
      box(gx, 3.5, 1.4, 0.7, 4, WOODHQ, 34);
      const a = AGENTS[role];
      const scr = a.vote === "APPROVE" ? "#4be08f" : a.vote === "REJECT" ? "#ff6b6b" : a.vote === "ABSTAIN" ? "#ffc857" : "#5aa9ff";
      box(gx + 0.62, 3.6, 0.08, 0.08, 10, "#9AA0AE", 38);
      box(gx + 0.25, 3.56, 0.9, 0.08, 24, INK, 48);
      poly([iso(gx + 0.3, 3.64, 51), iso(gx + 1.1, 3.64, 51), iso(gx + 1.1, 3.64, 69), iso(gx + 0.3, 3.64, 69)],
           (Math.floor(t / 700 + gx) % 6) ? scr : shade("#5aa9ff", 0.8));
      box(gx + 0.35, 3.9, 0.6, 0.18, 2, "#FFFFFF", 38);
      box(gx + 1.12, 3.95, 0.12, 0.12, 6, "#FFFFFF", 38);
    });
  }

  // Fuel Café: back counter, barista counter, communal high table, stools
  add(0, 7.4, 0.7, 4.6, () => {
    box(0, 7.4, 0.7, 4.6, 36, EPOXY); floorQuad(0, 7.4, 0.7, 4.6, "#FFFFFF", 36);
    box(0.1, 8.0, 0.5, 0.6, 24, INK, 36);
    box(0.15, 10.4, 0.45, 0.8, 30, "#DCE3EC", 36);
  });
  add(1.4, 7.6, 3.6, 0.7, () => {
    box(1.4, 7.6, 3.6, 0.7, 40, WOODHQ); floorQuad(1.4, 7.6, 3.6, 0.7, "#FFFFFF", 40);
    box(1.8, 7.7, 0.9, 0.6, 22, "#D9DDE3", 40);
    poly([iso(1.8, 8.3, 40), iso(2.7, 8.3, 40), iso(2.7, 8.3, 45), iso(1.8, 8.3, 45)], BLUE);
    box(3.0, 7.75, 0.4, 0.4, 20, INK, 40);
    ctx.globalAlpha = 0.75; box(3.6, 7.7, 1.2, 0.55, 16, "#DCEBF8", 40); ctx.globalAlpha = 1;
    for (const [gx, c] of [[3.8, "#E2A65C"], [4.2, "#D9934A"], [4.55, "#6B4226"]]) { const [x, y] = iso(gx, 8.1, 46); ellipse(x, y, 5, 3, c); }
  });
  add(1.6, 10.0, 3.1, 0.9, () => {
    box(1.8, 10.1, 0.12, 0.7, 30, INK); box(4.4, 10.1, 0.12, 0.7, 30, INK);
    box(1.6, 10.0, 3.1, 0.9, 4, WOODHQ, 30);
    box(2.2, 10.3, 0.22, 0.22, 7, "#FFFFFF", 34);
    box(3.0, 10.25, 0.7, 0.45, 2, "#C9CED8", 34);
    box(4.0, 10.3, 0.35, 0.35, 3, "#FFFFFF", 34);
  });
  for (const [gx, col] of [[2.1, BLUE], [3.2, AMBER], [4.3, BLUE]]) {
    add(gx - 0.2, 11.3, 0.4, 0.4, () => {
      const [x, y] = iso(gx, 11.5); ctx.fillStyle = INK; ctx.fillRect(x - 2, y - 22, 4, 22);
      ellipse(x, y - 23, 11, 6, shade(col, 0.8)); ellipse(x, y - 24, 11, 6, col);
    });
  }

  // game zone: two arcade cabinets, foosball, poufs
  for (const [gx, body, top] of [[9.4, BLUE, AMBER], [10.5, CORAL, INK]]) {
    add(gx, 7.2, 0.9, 1.0, () => {
      box(gx, 7.2, 0.9, 1.0, 80, body);
      floorQuad(gx, 7.2, 0.9, 1.0, top, 80);
      poly([iso(gx + 0.12, 8.2, 66), iso(gx + 0.78, 8.2, 66), iso(gx + 0.78, 8.2, 74), iso(gx + 0.12, 8.2, 74)], top);
      poly([iso(gx + 0.12, 8.2, 40), iso(gx + 0.78, 8.2, 40), iso(gx + 0.78, 8.2, 62), iso(gx + 0.12, 8.2, 62)],
           (Math.floor(t / 400 + gx) % 3) ? "#0E1118" : "#18233F");
      const [bx, by] = iso(gx + 0.3, 8.2, 34); ellipse(bx, by, 2.5, 2.5, MINT); ellipse(bx + 10, by + 5, 2.5, 2.5, CORAL);
    });
  }
  add(7.0, 9.3, 2.0, 1.1, () => {
    box(7.2, 9.45, 0.15, 0.8, 22, INK); box(8.65, 9.45, 0.15, 0.8, 22, INK);
    box(7.0, 9.3, 2.0, 1.1, 10, WOODHQ, 22);
    floorQuad(7.1, 9.4, 1.8, 0.9, "#33A86A", 32);
    for (const gx of [7.4, 7.8, 8.2, 8.6]) {
      const [x1, y1] = iso(gx, 9.1, 37), [x2, y2] = iso(gx, 10.6, 37);
      ctx.strokeStyle = "#9AA0AE"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
  });
  add(10.8, 10.0, 0.6, 0.6, () => pouf(11.1, 10.3, BLUE));
  add(11.8, 9.3, 0.6, 0.6, () => pouf(12.1, 9.6, AMBER));

  // plants
  for (const [gx, gy, s] of [[12.65, 2.4, 1.1], [12.4, 12.3, 1.2], [0.5, 6.4, 1], [6.4, 6.6, 0.9]]) {
    add(gx - 0.3, gy - 0.3, 0.6, 0.6, () => plant(gx, gy, s));
  }
  return items;
}

// ===================================================================== overlay bubbles
function speak(role, name, text, vote) {
  const a = AGENTS[role];
  if (!a) return;
  if (a.bubble) a.bubble.remove();
  clearTimeout(a.bubbleTimer); clearInterval(a.typeTimer);
  const b = document.createElement("div");
  b.className = "bubble" + (vote ? ` vote-${vote}` : "");
  const who = document.createElement("span"); who.className = "who"; who.textContent = `${name}`;
  const body = document.createElement("span"); body.className = "txt";
  b.append(who, body);
  overlay.appendChild(b);
  a.bubble = b;
  const dur = Math.min(6000 + text.length * 45, 16000);
  a.talkingUntil = performance.now() + Math.min(text.length * 14, 4000);
  let i = 0;
  a.typeTimer = setInterval(() => { body.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(a.typeTimer); }, 14);
  a.bubbleTimer = setTimeout(() => { b.classList.add("fade"); setTimeout(() => { b.remove(); if (a.bubble === b) a.bubble = null; }, 400); }, dur);
}

function placeOverlay() {
  for (const a of Object.values(AGENTS)) {
    if (!a.bubble || a.sx === undefined) continue;
    const x = Math.min(Math.max(a.sx, 170), W - 170);
    a.bubble.style.left = `${(x / W) * 100}%`;
    a.bubble.style.top = `${((a.sy - 76) / H) * 100}%`;
  }
}

// ===================================================================== side panel
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
  const box_ = $("votes"); box_.replaceChildren();
  for (const [role, v] of Object.entries(votes || {})) {
    const el = document.createElement("span");
    el.className = `vote ${v}`; el.textContent = `${AGENTS[role]?.name || role}: ${v}`;
    box_.appendChild(el);
  }
}

function setMeeting(on) {
  meeting = on;
  // Route via the blue wayfinding strip so nobody walks through the bleachers.
  for (const a of Object.values(AGENTS)) {
    const laneY = Math.min(Math.max(a.home[1], 2.1), 9.0);
    a.path = on ? [[AISLE_GX, laneY], [AISLE_GX, 2.1], a.seat] : [[AISLE_GX, 2.1], [AISLE_GX, laneY], a.home];
    if (on) a.vote = null;
  }
}

// ===================================================================== websocket
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

// ===================================================================== main loop
function frame(t) {
  for (const a of Object.values(AGENTS)) {
    if (a.path.length) [a.tx, a.ty] = a.path[0];
    const dx = a.tx - a.gx, dy = a.ty - a.gy, dist = Math.hypot(dx, dy);
    if (dist > 0.02) { const sp = Math.min(dist, 0.035); a.gx += (dx / dist) * sp; a.gy += (dy / dist) * sp; }
    else if (a.path.length) a.path.shift();
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(bg, 0, 0);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  drawLedWall(t);
  const items = furniture(t);
  for (const a of Object.values(AGENTS)) items.push({ depth: a.gx + a.gy + 0.2, fn: () => drawAgent(a, t) });
  items.sort((p, q) => p.depth - q.depth).forEach((it) => it.fn());
  for (const a of Object.values(AGENTS)) drawCard(a, t);
  placeOverlay();
  requestAnimationFrame(frame);
}

document.fonts.ready.then(() => {
  paintStatic();
  connect();
  requestAnimationFrame(frame);
});
