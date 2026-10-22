"use strict";
// Vanilla JS chat client. All text is inserted with textContent or escaped first, never as raw HTML from the model.

const log = document.getElementById("log");          // the scrolling area
const thread = document.getElementById("log-inner");  // the centered column that holds the messages
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");

let busy = false;

// ---------- small DOM helpers ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const PLANE_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" ' +
  'd="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z"/></svg>';

function scrollToBottom() {
  log.scrollTop = log.scrollHeight;
}

// ---------- minimal safe markdown (headings, lists, tables, bold, italic, code) ----------

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, "$1<em>$2</em>");
}

function splitRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function renderMarkdown(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = Math.min(Math.max(heading[1].length + 2, 3), 5);
      out.push(`<h${level} dir="auto">${inline(heading[2])}</h${level}>`);
      i++; continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { out.push("<hr>"); i++; continue; }

    if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(splitRow(lines[i++]));
      out.push("<table><thead><tr>" + head.map((c) => `<th dir="auto">${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td dir="auto">${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }

    const listMatch = line.match(/^\s*([-*•]|\d+[.)])\s+/);
    if (listMatch) {
      const ordered = /\d/.test(listMatch[1]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
        if (!m) break;
        items.push(`<li dir="auto">${inline(m[2])}</li>`);
        i++;
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*([-*•]|\d+[.)])\s|\s*\|)/.test(lines[i])) para.push(inline(lines[i++]));
    if (!para.length) { para.push(inline(lines[i++])); }
    out.push(`<p dir="auto">${para.join("<br>")}</p>`);
  }
  return out.join("");
}

// ---------- messages ----------

function addUserMessage(text) {
  const node = el("div", "msg user", text);
  node.dir = "auto";
  thread.appendChild(node);
  scrollToBottom();
}

function addWorkingMessage() {
  // Animated dots with elapsed seconds while the agent works (answers take about 10 seconds).
  const node = el("div", "msg agent");
  const avatar = el("div", "avatar");
  avatar.innerHTML = PLANE_SVG;  // a constant icon, not model text
  const bubble = el("div", "bubble");
  const status = el("div", "status");
  const dots = el("span", "dots");
  dots.append(el("i"), el("i"), el("i"));
  const secs = el("span", "secs");
  status.append(dots, el("span", "", "Running tools and computing scores"), secs);
  bubble.appendChild(status);
  node.append(avatar, bubble);
  thread.appendChild(node);
  scrollToBottom();

  const started = Date.now();
  const tick = () => { secs.textContent = `${Math.round((Date.now() - started) / 1000)}s`; };
  tick();
  const timer = setInterval(tick, 500);
  return { node, bubble, stop() { clearInterval(timer); } };
}

function labelRow(label, value) {
  const row = el("div", "row");
  row.append(el("span", "label", label), el("span", "", value));
  return row;
}

function listBlock(label, items, className) {
  const wrap = el("div", `row ${className || ""}`.trim());
  wrap.appendChild(el("div", "label", label));
  const ul = el("ul");
  items.forEach((t) => ul.appendChild(el("li", "", t)));
  wrap.appendChild(ul);
  return wrap;
}

function renderTool(t) {
  const box = el("div", "tool");
  const result = t.result || {};
  const head = el("h4");
  head.append(el("code", "", t.name));
  if (result.error) {
    head.append(el("span", "badge error", "error"));
  } else if (result.confidence) {
    head.append(el("span", `badge ${result.confidence}`, `${result.confidence} confidence`));
  }
  box.appendChild(head);
  const args = el("div", "row");
  args.append(el("span", "label", "Arguments"), el("code", "", JSON.stringify(t.arguments)));
  box.appendChild(args);

  if (result.error) {
    box.appendChild(labelRow("Error", result.error));
    if (result.valid_options && result.valid_options.length) box.appendChild(labelRow("Valid options", result.valid_options.join(", ")));
  } else {
    if (result.method) box.appendChild(labelRow("Method", result.method));
    if (result.coverage) box.appendChild(labelRow("Coverage", result.coverage));
    if (result.warnings && result.warnings.length) box.appendChild(listBlock("Warnings", result.warnings, "warnings"));
    if (result.assumptions && result.assumptions.length) box.appendChild(listBlock("Assumptions", result.assumptions));
    if (result.sources && result.sources.length) {
      box.appendChild(listBlock("Sources", result.sources.map((s) => `${s.dataset}, period ${s.period}, fetched ${s.fetched}`)));
    }
  }
  const raw = el("details");
  raw.appendChild(el("summary", "", "Raw tool result (JSON)"));
  raw.appendChild(el("pre", "", JSON.stringify(result, null, 2)));
  box.appendChild(raw);
  return box;
}

function renderWork(answer) {
  if (!answer.trace.length) return null;
  const details = el("details", "work");
  const warnCount = answer.trace.reduce((n, t) => n + ((t.result && t.result.warnings) || []).length, 0);
  const summary = el("summary", "", `Show work · ${answer.trace.length} tool call${answer.trace.length > 1 ? "s" : ""}`);
  if (warnCount) summary.appendChild(el("span", "warn-count", `${warnCount} warning${warnCount > 1 ? "s" : ""}`));
  details.appendChild(summary);
  answer.trace.forEach((t) => details.appendChild(renderTool(t)));
  const u = answer.usage || {};
  const models = (answer.models || []).join(" → ");  // two names mean the fallback model took over
  details.appendChild(el("div", "meta",
    `${answer.elapsed_s}s · ${u.llm_calls || 0} model call${u.llm_calls === 1 ? "" : "s"} · ${u.input_tokens || 0} input / ${u.output_tokens || 0} output tokens` +
    (models ? ` · ${models}` : "")));
  return details;
}

function showAnswer(working, answer) {
  working.stop();
  working.bubble.replaceChildren();
  if (!answer.ok) working.node.classList.add("failed");
  const body = el("div", "md");
  body.innerHTML = renderMarkdown(answer.text || "");
  working.bubble.appendChild(body);
  const work = renderWork(answer);
  if (work) working.bubble.appendChild(work);
  working.node.scrollIntoView({ block: "start" });
}

function showError(working, message) {
  working.stop();
  working.bubble.replaceChildren(el("div", "", message));
  working.node.classList.add("failed");
  scrollToBottom();
}

// ---------- talking to the server ----------

function setBusy(value) {
  busy = value;
  sendBtn.disabled = value;
  input.disabled = value;
  newChatBtn.disabled = value;
  if (!value) input.focus();
}

async function ask(question) {
  addUserMessage(question);
  const working = addWorkingMessage();
  setBusy(true);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      let message = `The server answered ${res.status}.`;
      try {
        const body = await res.json();
        if (typeof body.error === "string") message = body.error;
        else if (res.status === 422) message = "Please type a question of up to 1000 characters.";
      } catch (_) { /* keep the generic message */ }
      showError(working, message);
      return;
    }
    showAnswer(working, await res.json());
  } catch (err) {
    showError(working, "Could not reach the server. Please check that it is running and try again.");
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
  input.value = "";
  autosize();
  ask(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  input.style.overflowY = input.scrollHeight > 160 ? "auto" : "hidden";  // no scrollbar until the box is full
}
input.addEventListener("input", autosize);

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.textContent.trim();
    form.requestSubmit();
  });
});

newChatBtn.addEventListener("click", async () => {
  if (busy) return;
  try { await fetch("/api/reset", { method: "POST" }); } catch (_) { /* a new page load also starts fresh */ }
  window.location.reload();
});

// ---------- voice input (browser Web Speech API; feature-detected) ----------

(function setupVoice() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return;
  const mic = document.getElementById("mic");
  const langSelect = document.getElementById("voice-lang");
  const note = document.getElementById("voice-note");
  mic.hidden = langSelect.hidden = false;
  if ((navigator.language || "").toLowerCase().startsWith("he")) langSelect.value = "he-IL";

  let recognition = null;
  let baseText = "";

  function stop() { if (recognition) recognition.stop(); }

  mic.addEventListener("click", () => {
    if (recognition) { stop(); return; }
    recognition = new Recognition();
    recognition.lang = langSelect.value;
    recognition.interimResults = true;
    recognition.continuous = false;
    baseText = input.value ? input.value.trimEnd() + " " : "";
    recognition.onstart = () => { mic.setAttribute("aria-pressed", "true"); note.hidden = false; };
    recognition.onresult = (event) => {
      let transcript = "";
      for (const result of event.results) transcript += result[0].transcript;
      input.value = baseText + transcript;
      autosize();
    };
    recognition.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        note.textContent = "Microphone access was blocked. Allow it in the browser to dictate.";
      } else if (event.error !== "no-speech" && event.error !== "aborted") {
        note.textContent = `Voice input failed (${event.error}). You can still type.`;
      }
      note.hidden = false;
    };
    recognition.onend = () => { recognition = null; mic.setAttribute("aria-pressed", "false"); input.focus(); };
    recognition.start();
  });
  langSelect.addEventListener("change", stop);
})();

input.focus();
