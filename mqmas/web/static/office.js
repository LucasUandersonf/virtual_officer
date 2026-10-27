/* MQ-MAS — Quant Floor.
 *
 * The isometric office in index.html is plain DOM (CSS-projected boxes). This
 * script brings it to life from the backend WebSocket (/ws): agents walk from
 * their desks to the debate room, speak in bubbles (Tree of Thoughts bars,
 * pre-mortem risks, hard-gate counts), and the side panels mirror telemetry,
 * the order plan and the debate minutes.
 *
 * Events: hello, telemetry, agent_message, debate_start, decision, position,
 * risk, system — see README "Protocolo WebSocket".
 */
"use strict";

const ROLES = ["chief_quant", "econometrician", "cro", "execution"];
const WALK_MS = 2400;

const AGENTS = {
  chief_quant: {
    name: "Ada", tag: "Ada", sub: "Tree of Thoughts", color: "#7C6CFF", side: "to-left",
    home: [101, 118], meet: [377, 131],
    vars: "--skin:#F1C27D;--hair:#3B2A1A;--shirt:#6C5CE7;--pants:#2D2A4A",
    body: '<span class="sd"></span><span class="hb"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="hd"><span class="hr" style="height: 9px; border-radius: 0 0 60% 20%"></span><span class="ey e1"></span><span class="ey e2"></span><span class="ck c1"></span><span class="ck c2"></span><span class="mo"></span></span>',
  },
  econometrician: {
    name: "Gauss", tag: "Gauss", sub: "Econometria", color: "#19B394", side: "to-left",
    home: [225, 118], meet: [463, 72],
    vars: "--skin:#E0AC69;--hair:#C3C8D2;--shirt:#138F77;--pants:#3A4152",
    body: '<span class="sd"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="er r1"></span><span class="er r2"></span><span class="hd"><span class="hr" style="height: 5px"></span><span class="ey e1"></span><span class="ey e2"></span><span style="left: 3px; top: 9px; width: 7px; height: 6px; box-sizing: border-box; border: 1.5px solid #1E2230; border-radius: 2px"></span><span style="left: 12px; top: 9px; width: 7px; height: 6px; box-sizing: border-box; border: 1.5px solid #1E2230; border-radius: 2px"></span><span style="left: 2px; top: 15px; width: 18px; height: 10px; border-radius: 0 0 9px 9px; background: #C3C8D2"></span><span class="mo" style="background: #6B3A2E"></span></span>',
  },
  cro: {
    name: "Vera", tag: "Vera · CRO", sub: "Pre-mortem", color: "#FF6B5B", side: "to-right",
    home: [111, 356], meet: [469, 183],
    vars: "--skin:#C68642;--hair:#4A1414;--shirt:#E8574A;--pants:#2B2B35;--under:#FFF4EE",
    body: '<span class="sd"></span><span style="left: 15px; bottom: calc(var(--lh) + 51px); width: 10px; height: 10px; border-radius: 50%; background: var(--hair)"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="er r1"></span><span class="er r2"></span><span class="hd"><span class="hr" style="height: 8px; border-radius: 0 0 40% 40%"></span><span class="ey e1"></span><span class="ey e2"></span><span class="ck c1"></span><span class="ck c2"></span><span class="mo"></span></span>',
  },
  execution: {
    name: "Hex", tag: "Hex", sub: "Execução", color: "#F5A524", side: "to-left",
    home: [451, 370], meet: [556, 131],
    vars: "--skin:#8D5524;--hair:#141414;--shirt:#E99A16;--pants:#1E3A5F;--under:#C77F0A",
    body: '<span class="sd"></span><span class="sho s1"></span><span class="sho s2"></span><span class="lg l1"></span><span class="lg l2"></span><span class="lap"></span><span class="arm a1"></span><span class="arm a2"></span><span class="tor"></span><span class="col"></span><span class="hn h1"></span><span class="hn h2"></span><span class="nk"></span><span class="hd"><span class="hr" style="height: 7px"></span><span class="ey e1"></span><span class="ey e2"></span><span class="mo"></span></span><span style="left: 7px; bottom: calc(var(--lh) + 42px); width: 26px; height: 15px; box-sizing: border-box; border: 3px solid #2B3145; border-bottom: 0; border-radius: 14px 14px 0 0"></span><span style="left: 5px; bottom: calc(var(--lh) + 35px); width: 6px; height: 10px; border-radius: 3px; background: #2B3145"></span><span style="left: 29px; bottom: calc(var(--lh) + 35px); width: 6px; height: 10px; border-radius: 3px; background: #2B3145"></span>',
  },
};

const VOTE_LABEL = { APPROVE: "APROVA", REJECT: "REJEITA", ABSTAIN: "ABSTÉM", THINK: "ANALISANDO", QUEUE: "NA FILA", SKIP: "—" };
const STATUS_TEXT = { APPROVE: "Aprova", REJECT: "Rejeita", ABSTAIN: "Abstém", THINK: "Analisando…", QUEUE: "Na fila", SKIP: "Não consultado" };
const BAR_COLOR = { APPROVE: "#22B07D", REJECT: "#E5484D", ABSTAIN: "#F5A524", THINK: "#7C6CFF", QUEUE: "#263052", SKIP: "#263052" };

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 2) => `${v >= 0 ? "+" : "−"}${fmt(Math.abs(v) * 100, d)}%`;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const stripIcon = (t) => (t || "").replace(/^[^\p{L}\p{N}]{1,3}\s*/u, "");

const world = $("world");
const overlayLayer = $("overlay-layer");
const debate = { id: null, active: false, start: 0, end: 0, status: {}, result: null, limit: null };

// ===================================================================== agents in the scene
function setPos(node, [x, y]) { node.style.setProperty("--x", `${x}px`); node.style.setProperty("--y", `${y}px`); }

for (const role of ROLES) {
  const a = AGENTS[role];
  a.where = "home"; a.lastVote = null;

  a.node = el("div", "a agent");
  a.node.style.setProperty("--z", "4px");
  const pp = el("div", "pp seated"); pp.setAttribute("style", a.vars); pp.innerHTML = a.body;
  a.node.append(pp);
  setPos(a.node, a.home);
  world.querySelector(`.slot[data-slot="home-${role}"]`).after(a.node);

  a.ov = el("div", "a agent-ov");
  a.ov.style.setProperty("--z", "4px");
  setPos(a.ov, a.home);
  a.tagEl = el("div", "tag");
  a.bubbleBox = el("div");
  a.ov.append(a.tagEl, a.bubbleBox);
  overlayLayer.append(a.ov);
  renderTag(role);
}

/** Walk to the desk ("home") or the debate room ("meet"). Painter order is fixed up on arrival. */
function moveAgent(role, where) {
  const a = AGENTS[role];
  if (a.where === where) return;
  a.where = where;
  const target = where === "meet" ? a.meet : a.home;
  setPos(a.node, target); setPos(a.ov, target);
  clearTimeout(a.walkTimer);
  a.walkTimer = setTimeout(() => {
    world.querySelector(`.slot[data-slot="${where}-${role}"]`).after(a.node);
  }, WALK_MS + 50);
}

function renderTag(role) {
  const a = AGENTS[role];
  const status = debate.id ? debate.status[role] : null;
  const shown = status && status !== "SKIP" ? status : a.lastVote;
  a.tagEl.replaceChildren();
  const dot = el("span", "dot"); dot.style.background = a.color;
  a.tagEl.append(dot, document.createTextNode(a.tag));
  if (shown) a.tagEl.append(el("span", `vchip v-${shown}`, VOTE_LABEL[shown]));
}

function setTalking(role, ms) {
  const a = AGENTS[role];
  a.tagEl.classList.add("talking");
  clearTimeout(a.talkTimer);
  a.talkTimer = setTimeout(() => a.tagEl.classList.remove("talking"), ms);
}

function bubbleSide(a) {
  const [x, y] = a.where === "meet" ? a.meet : a.home;
  return x - y < 0 ? "to-right" : a.side;
}

function clearBubble(a) {
  clearTimeout(a.bubbleTimer); clearInterval(a.typeTimer);
  a.bubbleBox.replaceChildren();
}

function showTyping(role) {
  const a = AGENTS[role];
  clearBubble(a);
  const b = el("div", `bubble typing ${bubbleSide(a)}`);
  b.append(el("span", "tdot"), el("span", "tdot"), el("span", "tdot"));
  a.bubbleBox.append(b);
}

function showBubble(role, m) {
  const a = AGENTS[role];
  // one full bubble at a time; "typing…" indicators of other agents stay
  for (const r of ROLES) if (r !== role && !AGENTS[r].bubbleBox.querySelector(".typing")) clearBubble(AGENTS[r]);
  clearBubble(a);
  const b = el("div", `bubble ${bubbleSide(a)}`);
  const head = el("div", "bh");
  const av = el("span", "av", a.name[0]); av.style.background = a.color;
  head.append(av, el("strong", "bt", `${a.name} · ${m.debate_id ? a.sub : "status"}`));
  if (m.vote) head.append(el("span", `vchip v-${m.vote}`, VOTE_LABEL[m.vote]));
  else if (m.provider) head.append(el("span", "meta", m.provider));
  b.append(head);

  const data = m.data || {};
  if (Array.isArray(data.scenarios) && data.scenarios.length) {
    const by = Object.fromEntries(data.scenarios.map((s) => [s.name, s.probability]));
    const bar = el("div", "tot");
    for (const [k, c] of [["bullish", "#22B07D"], ["neutral", "#B8C0D6"], ["bearish", "#E5484D"]]) {
      const s = el("span"); s.style.width = `${(by[k] || 0) * 100}%`; s.style.background = c; bar.append(s);
    }
    const lab = el("div", "tot-l");
    for (const [k, txt, c] of [["bullish", "Bullish", "#0E7A55"], ["neutral", "Neutro", "#5B6478"], ["bearish", "Bearish", "#B42318"]]) {
      const s = el("span", "", `${txt} ${Math.round((by[k] || 0) * 100)}%`); s.style.color = c; lab.append(s);
    }
    b.append(bar, lab);
  }
  if (Array.isArray(data.pre_mortem) && data.pre_mortem.length) {
    const box = el("div", "risks");
    const lk = { low: "BAIXO", medium: "MÉDIO", high: "ALTO" };
    for (const f of data.pre_mortem.slice(0, 3)) {
      const row = el("div", "risk");
      row.append(el("span", `lk lk-${f.likelihood}`, lk[f.likelihood] || f.likelihood), document.createTextNode(f.risk));
      box.append(row);
    }
    b.append(box);
  }
  const text = stripIcon(m.text);
  const p = el("p"); b.append(p);
  if (Array.isArray(data.gates) && data.gates.length) {
    const ok = data.gates.filter((g) => g.passed).length;
    const g = el("div", "gates", "Travas rígidas: ");
    const s = el("strong", "", `${ok}/${data.gates.length} OK`); s.style.color = ok === data.gates.length ? "#0E7A55" : "#B42318";
    g.append(s); b.append(g);
  }
  a.bubbleBox.append(b);

  let i = 0;
  a.typeTimer = setInterval(() => { p.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(a.typeTimer); }, 14);
  const dur = Math.min(6000 + text.length * 45, 16000);
  setTalking(role, Math.min(text.length * 14 + 800, 4500));
  a.bubbleTimer = setTimeout(() => { b.classList.add("fade"); setTimeout(() => b.remove(), 400); }, dur);
}

// ===================================================================== debate card
function renderDebateCard() {
  $("dc-dot").classList.toggle("on", debate.active);
  $("dc-title").textContent = debate.id ? `Debate #${debate.id.slice(0, 6)}` : "Sem debate ativo";
  const rows = $("dc-rows"); rows.replaceChildren();
  const bar = $("dc-bar"); bar.replaceChildren();
  for (const role of ROLES) {
    const a = AGENTS[role], st = debate.id ? debate.status[role] || "QUEUE" : null;
    const row = el("div", "dc-row");
    const av = el("span", "av"); av.style.background = a.color;
    row.append(av, document.createTextNode(a.name), el("span", `st st-${st || "QUEUE"}`, st ? STATUS_TEXT[st] : "Na mesa"));
    rows.append(row);
    const seg = el("span"); seg.style.background = BAR_COLOR[st || "QUEUE"]; bar.append(seg);
  }
  const approvals = ROLES.filter((r) => debate.status[r] === "APPROVE").length;
  $("dc-foot").textContent = !debate.id ? "Aguardando o pré-filtro estatístico"
    : debate.active ? `Unanimidade exigida · ${approvals} de 4`
    : debate.result || `Encerrado · ${approvals} de 4`;
  $("scene-sub").textContent = `4 agentes · ${debate.active ? "1 reunião ativa" : "nenhuma reunião ativa"}`;
}

function tickTimer() {
  if (!debate.id) { $("dc-timer").textContent = "—"; return; }
  const s = Math.max(0, Math.floor(((debate.active ? Date.now() : debate.end) - debate.start) / 1000));
  $("dc-timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function nextThinker() {
  const next = ROLES.find((r) => debate.status[r] === "QUEUE");
  if (next) { debate.status[next] = "THINK"; showTyping(next); renderTag(next); }
}

// ===================================================================== side panels
let lastTelemetryAt = 0;

function onTelemetry(p) {
  lastTelemetryAt = Date.now();
  const s = p.snapshot, spark = p.spark || [];
  $("h-price").textContent = fmt(s.price);
  if (spark.length > 1) {
    const ch = (s.price - spark[0]) / spark[0];
    $("h-change").textContent = pct(ch); $("h-change").className = `mono small ${ch >= 0 ? "pos" : "neg"}`;
    $("h-change").title = "Variação nas últimas 24h";
  }
  $("h-equity").textContent = `US$ ${fmt(p.equity)}`;
  setMode(p.mode);
  $("pill-lock").hidden = !p.lock_until;

  $("t-z").textContent = (s.zscore >= 0 ? "+" : "−") + fmt(Math.abs(s.zscore));
  $("t-z").className = `mono big ${s.zscore <= -2 ? "neg" : s.zscore >= 2 ? "pos" : ""}`;
  const zDot = $("t-z-dot");
  zDot.style.left = `${clamp((s.zscore + 3) / 6, 0, 1) * 100}%`;
  zDot.style.background = s.zscore <= -2 ? "#FF6B5B" : s.zscore <= -1.5 ? "#FBC15E" : "#7C6CFF";

  const imb = s.orderbook.imbalance;
  $("t-imb").textContent = (imb >= 0 ? "+" : "−") + fmt(Math.abs(imb));
  $("t-imb").className = `mono big ${imb >= 0 ? "pos" : "neg"}`;
  $("t-imb-bid").style.width = `${((1 + imb) / 2) * 100}%`;
  $("t-imb-ask").style.width = `${((1 - imb) / 2) * 100}%`;

  $("t-vwap").textContent = fmt(s.vwap);
  const dv = (s.price - s.vwap) / s.vwap;
  $("t-vwap-sub").textContent = `preço ${fmt(Math.abs(dv) * 100)}% ${dv < 0 ? "abaixo" : "acima"}`;
  $("t-vwap-sub").className = `muted-xs ${dv < 0 ? "neg" : "pos"}`;
  $("t-poc").textContent = fmt(s.volume_profile.poc);
  $("t-va").textContent = `VA ${fmt(s.volume_profile.value_area_low, 0)} – ${fmt(s.volume_profile.value_area_high, 0)}`;

  const f = s.forecast;
  $("t-fc-label").textContent = `Projeção ${f.horizon}×15m · IC 95%`;
  $("t-fc").textContent = fmt(f.expected_price);
  const lo = Math.min(f.lower_95, s.price), hi = Math.max(f.upper_95, s.price), pad = (hi - lo) * 0.15 || 1;
  const pos = (v) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100;
  $("t-ci-band").style.left = `${pos(f.lower_95)}%`; $("t-ci-band").style.right = `${100 - pos(f.upper_95)}%`;
  $("t-ci-mark").style.left = `${pos(f.expected_price)}%`;
  $("t-ci-price").style.left = `${pos(s.price)}%`;
  $("t-ci-lo").textContent = fmt(f.lower_95, 0); $("t-ci-hi").textContent = fmt(f.upper_95, 0);

  drawTv(spark, s);
}

/** Wall screen in the debate room: last 24h of closes + VWAP. */
function drawTv(spark, s) {
  const line = $("tv-line"), vw = $("tv-vwap");
  if (!line || spark.length < 2) return;
  const vals = [...spark, s.vwap], min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const yOf = (v) => 40 - ((v - min) / span) * 36;
  line.setAttribute("points", spark.map((v, i) => `${((i / (spark.length - 1)) * 214).toFixed(1)},${yOf(v).toFixed(1)}`).join(" "));
  const y = yOf(s.vwap).toFixed(1);
  vw.setAttribute("y1", y); vw.setAttribute("y2", y);
}

function tickAge() {
  if (!lastTelemetryAt) return;
  const s = Math.round((Date.now() - lastTelemetryAt) / 1000);
  $("t-age").textContent = s < 2 ? "atualizado agora" : `atualizado há ${s}s`;
}

function setMode(mode) {
  if (!mode) return;
  $("pill-mode").textContent = mode.toUpperCase();
  $("pill-mode").classList.toggle("live", mode === "live");
}

function renderPlan(plan, status, limit) {
  if (!plan) return;
  $("p-status").textContent = status;
  $("p-qty").textContent = `${fmt(plan.quantity, 5)} BTC`;
  $("p-limit").textContent = fmt(limit ?? plan.entry_price);
  $("p-stop-label").textContent = `Stop (−${fmt(plan.stop_pct * 100)}%)`;
  $("p-stop").textContent = fmt(plan.stop_price);
  $("p-tp-label").textContent = `Take profit (+${fmt(plan.take_profit_pct * 100)}%)`;
  $("p-tp").textContent = fmt(plan.take_profit_price);
  $("p-risk").textContent = `US$ ${fmt(plan.max_loss_usdt)}`;
}

function renderPosition(pos) {
  $("p-status").textContent = pos.status === "open" ? "posição aberta" : `posição ${pos.status}`;
  $("p-qty").textContent = `${fmt(pos.quantity, 5)} BTC`;
  $("p-limit").textContent = fmt(pos.entry_price);
  $("p-stop").textContent = fmt(pos.stop_price);
  $("p-tp").textContent = fmt(pos.take_profit_price);
}

const CLOCK_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path></svg>';

function addMinute(role, name, text, ts, vote) {
  const list = $("ata-list");
  list.querySelector(".empty")?.remove();
  const li = el("li");
  const a = AGENTS[role];
  const av = el("span", "av", a ? a.name[0] : "");
  if (a) av.style.background = a.color; else { av.style.background = "#263052"; av.style.color = "#C9CFE0"; av.innerHTML = CLOCK_SVG; }
  const body = el("div", "body");
  const hdr = el("div", "hdr");
  hdr.append(el("strong", "", name), el("span", "time", new Date(ts).toLocaleTimeString("pt-BR")));
  if (vote) hdr.append(el("span", `chip6 vchip v-${vote}`, VOTE_LABEL[vote]));
  body.append(hdr, el("p", "", stripIcon(text)));
  li.append(av, body);
  list.prepend(li);
  while (list.children.length > 80) list.lastChild.remove();
}

// ===================================================================== setup league
function renderLeague(report) {
  const list = $("lg-list");
  if (!report) return;
  const ok = report.entries.filter((e) => e.approved).length;
  const end = report.end ? new Date(report.end) : null;
  const days = report.start && end ? Math.round((end - new Date(report.start)) / 86400000) : 0;
  $("lg-sub").textContent = `${ok}/${report.entries.length} aprovados · ${days} dias de histórico`;
  list.replaceChildren();
  for (const e of report.entries) {
    const li = el("li", `lg-row${e.approved ? " ok" : ""}`);
    li.title = e.approved ? e.thesis : `${e.thesis}\nReprovado: ${e.reasons.join("; ")}`;
    const name = el("div", "lg-name");
    const title = el("strong", "", e.name);
    title.append(el("span", `lg-badge ${e.approved ? "ok" : "no"}`, e.approved ? "ATIVO" : "REPROVADO"));
    name.append(title,
      el("span", "", `${e.family} · alvo ${(e.profile.tp_pct * 100).toFixed(1)}% · ${e.oos.trades} trades OOS`));
    const pf = e.oos.profit_factor >= 99 ? "∞" : e.oos.profit_factor.toFixed(2);
    const st = el("div", "lg-stats mono");
    st.append(el("span", e.oos.expectancy_pct > 0 ? "pos" : "neg", `${pct(e.oos.expectancy_pct, 3)}/trade`),
      el("span", "muted-xs", `acerto ${(e.oos.win_rate * 100).toFixed(0)}% · PF ${pf}`));
    li.append(el("span", "lg-rank mono", `${e.rank}`), name, st);
    list.append(li);
  }
}

async function loadLeague() {
  try {
    const r = await fetch("/api/league");
    if (!r.ok) return;
    const data = await r.json();
    if (data.report) renderLeague(data.report);
    else setTimeout(loadLeague, 20000);
  } catch (e) { setTimeout(loadLeague, 20000); }
}

// ===================================================================== strategy book
function renderBook(b) {
  if (!b || b.enabled === false) { $("bk-sub").textContent = "desativada"; return; }
  const regime = b.btc_regime_ok === null || b.btc_regime_ok === undefined ? "regime BTC —"
    : (b.btc_regime_ok ? "regime BTC ✔ favorável" : "regime BTC ✖ (em caixa)");
  $("bk-sub").textContent = `${(b.universe || []).length} moedas no radar · ${regime}`;
  if (b.strategies) $("bk-strats").textContent = `Setups ativos: ${b.strategies.join(", ")}`;
  const sigs = b.last_signals || [];
  const when = b.last_run ? new Date(b.last_run).toLocaleString("pt-BR") : "—";
  $("bk-signals").textContent = sigs.length
    ? `Últimos sinais (${when}): ${sigs.map((x) => x.symbol).join(", ")}`
    : `Última análise: ${when} · nenhum sinal${(b.notes || []).length ? " · " + b.notes.join("; ") : ""}`;
  const conf = b.confluence || [];
  if (conf.length) {
    const cl = $("bk-conf");
    cl.replaceChildren();
    for (const c of conf.slice(0, 6)) {
      const cls = c.decisao === "EXECUTAR" ? "ok" : (c.score >= 60 ? "mid" : "no");
      const li = el("li", `lg-row ${cls}`);
      const name = el("div", "lg-name");
      const title = el("strong", "", `${c.symbol} · ${c.tf}`);
      title.append(el("span", `lg-badge ${cls}`, c.decisao));
      const o = c.ordem || {};
      name.append(title, el("span", "", `${c.motivo} · entrada ${fmt(o.entrada, 4)} · stop ${fmt(o.stop, 4)} · alvo ${fmt(o.alvo, 4)}`));
      li.append(el("span", "lg-rank mono", String(Math.round(c.score))), name);
      cl.append(li);
    }
  }
  if (!b.open_positions) return;
  const list = $("bk-pos");
  list.replaceChildren();
  if (!b.open_positions.length) { list.append(el("li", "muted-sm", "Sem posições abertas.")); return; }
  for (const p of b.open_positions) {
    const li = el("li", "lg-row ok");
    const name = el("div", "lg-name");
    const title = el("strong", "", p.symbol);
    title.append(el("span", "lg-badge ok", "ABERTA"));
    name.append(title, el("span", "", `entrada ${fmt(p.entry_price, 4)} · ${new Date(p.opened_at).toLocaleDateString("pt-BR")}`));
    const st = el("div", "lg-stats mono");
    st.append(el("span", "neg", `stop ${fmt(p.stop_price, 4)}`), el("span", "pos", `alvo ${fmt(p.take_profit_price, 4)}`));
    li.append(el("span", "lg-rank mono", "•"), name, st);
    list.append(li);
  }
}

async function loadBook() {
  try { const r = await fetch("/api/book"); if (r.ok) renderBook(await r.json()); } catch (e) { /* retry on next event */ }
}

// ===================================================================== technical analysis
const VERDICT_CLASS = { "funciona": "ok", "funciona ao contrário": "mid", "direção certa, não paga custos": "mid", "sem efeito": "no" };

function renderTA(r) {
  const t = r.tendencia, m = r.momento, v = r.volatilidade, s = r.estrutura;
  $("ta-sub").textContent = `${new Date().toLocaleTimeString("pt-BR")} · saldo de confluência ${r.confluencia.saldo >= 0 ? "+" : ""}${r.confluencia.saldo}`;
  const cells = [
    ["Tendência", `${t.vies} · ADX ${t.adx ?? "—"}`], ["Estrutura", s.tendencia_smc], ["RSI", m.rsi ?? "—"],
    ["MACD hist", m.macd_hist === null ? "—" : (m.macd_hist >= 0 ? "positivo" : "negativo")],
    ["Bollinger", `${v.posicao_bollinger}${v.squeeze_ttm ? " · squeeze" : ""}`], ["ATR", `${v.atr_pct ?? "—"}%`],
  ];
  const grid = $("ta-grid");
  grid.replaceChildren(...cells.map(([k, val]) => { const d = el("div", "ta-cell"); d.append(el("span", "", k), el("strong", "", String(val))); return d; }));
  const list = $("ta-pats");
  const pats = [...r.padroes_recentes.alta.map((p) => [p, "alta"]), ...r.padroes_recentes.baixa.map((p) => [p, "baixa"])];
  list.replaceChildren();
  if (!pats.length) { list.append(el("li", "muted-sm", "Nenhum padrão nos últimos 3 candles de 4h.")); return; }
  for (const [p, side] of pats) {
    const li = el("li", "lg-row");
    const name = el("div", "lg-name");
    const title = el("strong", "", p.padrao.replaceAll("_", " "));
    const h = p.historico;
    title.append(el("span", `lg-badge ${h ? VERDICT_CLASS[h.veredito] || "no" : "mid"}`, h ? h.veredito : "sem estudo"));
    name.append(title, el("span", "", `${p.categoria} · viés de ${side}` + (h ? ` · acerto histórico ${(h.acerto * 100).toFixed(0)}%` : "")));
    li.append(el("span", `lg-rank mono ${side === "alta" ? "pos" : "neg"}`, side === "alta" ? "▲" : "▼"), name);
    list.append(li);
  }
}

function renderLevels(m) {
  const box = $("lv-map");
  const row = (z, cls) => {
    const r = el("div", "lv-row");
    r.append(el("strong", `mono ${cls}`, fmt(z.preco, z.preco < 10 ? 4 : 2)),
      el("span", "src", `${"●".repeat(Math.min(z.forca, 5))} ${z.fontes.join(", ")}`),
      el("span", `mono muted-xs`, `${z.distancia_pct >= 0 ? "+" : ""}${z.distancia_pct.toFixed(1)}%`));
    return r;
  };
  const nodes = [el("div", "lv-head", "Resistências")];
  (m.resistencias || []).slice(0, 4).reverse().forEach((z) => nodes.push(row(z, "neg")));
  nodes.push(el("div", "lv-head", `Preço ${fmt(m.preco, m.preco < 10 ? 4 : 2)} · Suportes`));
  (m.suportes || []).slice(0, 4).forEach((z) => nodes.push(row(z, "pos")));
  const p = m.plano_compra || {};
  const plan = el("div", "lv-plan");
  if (p.entrada) {
    plan.append(el("strong", "", p.valido ? "Plano de compra (válido)" : "Plano de compra (aguardar)"),
      el("span", "mono", `Entrada ${fmt(p.entrada, 2)} · Stop ${fmt(p.stop, 2)} (${p.risco_pct}%)`),
      el("span", "mono", `Alvos ${p.alvos.map((x) => fmt(x, 2)).join(" / ")} · R:R ${p.risco_retorno.join(" / ")}`),
      el("span", "muted-xs", p.motivo));
  } else {
    plan.append(el("span", "muted-sm", "Sem zona de compra forte perto do preço: nenhum plano agora."));
  }
  nodes.push(plan);
  box.replaceChildren(...nodes);
}

async function loadLevels() {
  try { const r = await fetch("/api/levels/BTC?tf=4h"); if (r.ok) renderLevels(await r.json()); } catch (e) { /* next tick */ }
}

async function loadTA() {
  try { const r = await fetch("/api/ta/BTC?tf=4h"); if (r.ok) renderTA(await r.json()); } catch (e) { /* next tick */ }
}

// ===================================================================== websocket
function handle(ev) {
  const p = ev.payload;
  switch (ev.type) {
    case "hello":
      setMode(p.mode); break;
    case "telemetry":
      onTelemetry(p); break;
    case "debate_start":
      Object.assign(debate, { id: p.debate_id, active: true, start: Date.now(), end: 0, result: null, limit: null });
      debate.status = Object.fromEntries(ROLES.map((r) => [r, "QUEUE"]));
      for (const r of ROLES) { moveAgent(r, "meet"); clearBubble(AGENTS[r]); }
      nextThinker();
      ROLES.forEach(renderTag); renderDebateCard(); followDebate();
      break;
    case "agent_message": {
      if (p.role === "system" || !AGENTS[p.role]) { addMinute(null, p.agent_name, p.text, ev.ts); break; }
      const inDebate = p.debate_id && p.debate_id === debate.id && debate.active;
      if (inDebate) {
        debate.status[p.role] = p.vote || "ABSTAIN";
        if (p.data && p.data.limit_price) debate.limit = p.data.limit_price;
        AGENTS[p.role].lastVote = p.vote || "ABSTAIN";
      }
      // during a debate, routine status lines must not cover the deliberation bubbles
      if (inDebate || !debate.active) showBubble(p.role, p);
      if (inDebate && debate.status[p.role] === "APPROVE") nextThinker();
      renderTag(p.role); renderDebateCard();
      if (p.debate_id || p.provider !== "rules") addMinute(p.role, AGENTS[p.role].name, p.text, ev.ts, p.vote);
      break;
    }
    case "decision": {
      debate.active = false; debate.end = Date.now();
      for (const r of ROLES) {
        if (["QUEUE", "THINK"].includes(debate.status[r])) debate.status[r] = "SKIP";
        if (AGENTS[r].bubbleBox.querySelector(".typing")) clearBubble(AGENTS[r]);
      }
      debate.result = p.approved ? "Aprovado por unanimidade" : `Sem operação · ${p.reason}`;
      if (p.sizing) renderPlan(p.sizing, p.approved ? "aprovado · ordem enviada" : "último debate · não executado", debate.limit);
      ROLES.forEach(renderTag); renderDebateCard(); tickTimer();
      setTimeout(() => { if (!debate.active) { ROLES.forEach((r) => moveAgent(r, "home")); if (follow) centerView(); } }, 9000);
      break;
    }
    case "position": {
      const pos = p.position;
      loadBook();
      if (!(pos.setup || "").startsWith("book:")) renderPosition(pos);
      if (p.event === "closed") addMinute(null, "Mesa", `Posição ${pos.symbol} encerrada (${pos.exit_reason}) · PnL US$ ${fmt(pos.pnl_usdt, 4)}`, ev.ts);
      else if (p.event === "opened") addMinute(null, "Mesa", `Posição aberta: ${fmt(pos.quantity, 5)} ${pos.symbol.split("/")[0]} @ ${fmt(pos.entry_price, 4)}`, ev.ts);
      break;
    }
    case "league":
      renderLeague(p); break;
    case "book":
      renderBook(p); break;
    case "risk":
      if (p.event === "circuit_breaker") { $("pill-lock").hidden = false; addMinute(null, "Risco", `Circuit breaker armado até ${new Date(p.lock_until).toLocaleString("pt-BR")}`, ev.ts); }
      else if (p.event === "kill_switch") { setKill(p.engaged); addMinute(null, "Risco", p.engaged ? "Kill switch ACIONADO" : "Kill switch liberado", ev.ts); }
      else addMinute(null, "Risco", JSON.stringify(p), ev.ts);
      break;
  }
}

let backoff = 1000;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { backoff = 1000; $("pill-conn").classList.remove("off"); $("conn-text").textContent = "Ao vivo"; };
  ws.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) { console.error(e); } };
  ws.onclose = () => {
    $("pill-conn").classList.add("off"); $("conn-text").textContent = "Reconectando…";
    setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 15000);
  };
  const ping = setInterval(() => { if (ws.readyState === 1) ws.send("ping"); else clearInterval(ping); }, 20000);
}

// ===================================================================== dock
let zoom = 1, follow = true, killEngaged = false;
const stage = $("stage"), scroller = $("scroller");

function applyZoom() {
  stage.style.zoom = zoom;
  $("zoom-label").textContent = `${Math.round(zoom * 100)}%`;
  if (debate.active) followDebate(); else centerView();
}
$("zoom-in").addEventListener("click", () => { zoom = clamp(Math.round((zoom + 0.1) * 10) / 10, 0.6, 1.5); applyZoom(); });
$("zoom-out").addEventListener("click", () => { zoom = clamp(Math.round((zoom - 0.1) * 10) / 10, 0.6, 1.5); applyZoom(); });

function centerView() {
  scroller.scrollLeft = Math.max(0, (1100 * zoom - scroller.clientWidth) / 2);
}

function followDebate() {
  if (!follow) return;
  // the debate room sits around x = 800px of the 1100px stage
  scroller.scrollLeft = Math.max(0, 800 * zoom - scroller.clientWidth / 2);
}
$("follow").addEventListener("click", () => {
  follow = !follow;
  $("follow").setAttribute("aria-pressed", String(follow));
  if (follow && debate.active) followDebate();
});

function setKill(engaged) {
  killEngaged = engaged;
  $("kill").classList.toggle("engaged", engaged);
  $("kill-text").textContent = engaged ? "Kill switch ATIVO" : "Kill switch";
}
$("kill").addEventListener("click", async () => {
  const engage = !killEngaged;
  const ok = window.confirm(engage
    ? "Acionar o kill switch? Novas entradas ficam bloqueadas e posições abertas são zeradas a mercado."
    : "Liberar o kill switch e permitir novas entradas?");
  if (!ok) return;
  const token = window.prompt("ADMIN_TOKEN:");
  if (!token) return;
  try {
    const r = await fetch(`/api/kill-switch?engage=${engage}`, { method: "POST", headers: { "X-Admin-Token": token } });
    if (!r.ok) throw new Error(r.status === 401 ? "token inválido" : `HTTP ${r.status}`);
    setKill(engage);
  } catch (e) {
    addMinute(null, "Sistema", `Kill switch não alterado: ${e.message}`, Date.now());
  }
});

// ===================================================================== boot
async function boot() {
  $("ata-list").append(el("li", "empty", "As falas dos agentes aparecem aqui."));
  renderDebateCard();
  centerView();
  window.addEventListener("resize", () => (debate.active ? followDebate() : centerView()));
  try {
    const st = await (await fetch("/api/state")).json();
    setMode(st.mode);
    $("h-equity").textContent = `US$ ${fmt(st.equity)}`;
    setKill(Boolean(st.kill_switch));
    $("pill-lock").hidden = !st.lock_until;
    if (st.last_decision && st.last_decision.sizing) renderPlan(st.last_decision.sizing, "último debate", null);
    const desk = (st.open_positions || []).filter((x) => !(x.setup || "").startsWith("book:"));
    if (desk.length) renderPosition(desk[0]);
  } catch (e) { console.warn("state unavailable", e); }
  connect();
  loadLeague();
  loadBook();
  loadTA();
  loadLevels();
  setInterval(() => { loadTA(); loadLevels(); }, 5 * 60 * 1000);
  setInterval(() => { tickTimer(); tickAge(); }, 1000);
}
boot();
