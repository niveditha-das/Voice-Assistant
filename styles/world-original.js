// =====================================================================================
//  Offline Voice Assistant - a small language model + Whisper running in the browser
//  Personal project - runs entirely in the browser, no server
//  Author: <YOUR FULL NAME>            (replace before submitting)
// =====================================================================================
//
//  WHAT IT DOES
//  You speak (or type). Whisper (speech-to-text) turns your speech into text, a small
//  language model (SLM) writes a reply, and the browser reads the reply aloud.
//  Both neural networks are DOWNLOADED to the browser and RUN LOCALLY - no AI API is
//  called. After the models have loaded you can switch off Wi-Fi and it still works.
//
//  CREDITS (code, libraries, models)
//  - Transformers.js (Hugging Face) - runs the models in the browser on WebGPU or WASM.
//      https://huggingface.co/docs/transformers.js
//      https://github.com/huggingface/transformers.js
//      Loaded from the jsDelivr CDN: https://www.jsdelivr.com/package/npm/@huggingface/transformers
//  - ONNX Runtime Web (Microsoft) - the inference engine used inside Transformers.js.
//      https://onnxruntime.ai/docs/tutorials/web/
//  - Models, downloaded from the Hugging Face Hub (https://huggingface.co):
//      SmolLM2 135M / 360M Instruct (Hugging Face)  https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
//      Qwen2.5 0.5B Instruct (Alibaba Qwen team)     https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct
//      Llama 3.2 1B Instruct (Meta)                  https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-q4f16
//      Whisper tiny.en / base (OpenAI)               https://huggingface.co/onnx-community/whisper-tiny.en
//                                                    https://github.com/openai/whisper
//  - Streaming / interrupt pattern adapted from the official Transformers.js WebGPU chat
//    examples: https://github.com/huggingface/transformers.js-examples
//  - Web APIs (MDN): MediaRecorder https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder
//                    Web Audio     https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
//                    speechSynthesis https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis
//                    WebGPU        https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
//                    Web Workers   https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
//  - Ancient Brain: https://ancientbrain.com  ("Recognise any image" World used as starting point:
//                   https://ancientbrain.com/world.php?world=6692470813)
//  - Developed with help from Claude (Anthropic AI assistant) https://claude.ai
//
//  DESIGN NOTES
//  - We deliberately do NOT use the browser's SpeechRecognition API, because in Chrome it
//    sends audio to Google's servers. Speech recognition here is the local Whisper model.
//  - speechSynthesis uses voices installed on the user's own machine (local).
//  - The models run in a Web Worker (created from a Blob) so the page never freezes.
// =====================================================================================


(function () {
  "use strict";

  // ---------------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------------

  const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

  // Language models. dtype chosen at runtime from what the GPU supports.
  const LLMS = [
    { id: "HuggingFaceTB/SmolLM2-135M-Instruct", name: "SmolLM2 135M (fastest, ~120 MB)", params: 135 },
    { id: "HuggingFaceTB/SmolLM2-360M-Instruct", name: "SmolLM2 360M (balanced, ~270 MB)", params: 360 },
    { id: "onnx-community/Qwen2.5-0.5B-Instruct", name: "Qwen2.5 0.5B (smarter, ~480 MB)", params: 500 },
    { id: "onnx-community/Llama-3.2-1B-Instruct-q4f16", name: "Llama 3.2 1B (best, ~1.2 GB, WebGPU only)", params: 1000, gpuOnly: true },
  ];

  // Speech-to-text models
  const ASRS = [
    { id: "onnx-community/whisper-tiny.en", name: "Whisper tiny (English, ~45 MB)", multilingual: false },
    { id: "onnx-community/whisper-base", name: "Whisper base (multilingual, ~80 MB)", multilingual: true },
  ];

  // Personas = system prompts. Gives the assistant a "character".
  const PERSONAS = {
    assistant: "You are a friendly, helpful voice assistant. Keep answers short (2-4 sentences) because they will be read aloud.",
    tutor: "You are a patient tutor for AI topics (search, neural networks, deep learning, LLMs). " +
           "Explain clearly and simply, give a small example, and finish by asking the learner one short question to check understanding. Keep it short.",
    pirate: "You are a cheerful pirate captain. Answer every question correctly but in pirate speak. Keep answers short.",
  };

  // ---------------------------------------------------------------------------------
  // The Web Worker. All the AI runs here, off the main thread.
  // This function is turned into a string and run as a worker from a Blob URL,
  // so the whole World stays in ONE file on Ancient Brain.
  // ---------------------------------------------------------------------------------

  function workerMain() {
    let T = null;          // the Transformers.js library
    let llm = null;        // text-generation pipeline
    let asr = null;        // speech-recognition pipeline
    let stopper = null;    // lets the user interrupt generation

    const post = (msg) => self.postMessage(msg);

    function progress(which) {
      return (p) => post({ type: "progress", which, status: p.status, file: p.file || "", loaded: p.loaded || 0, total: p.total || 0 });
    }

    self.onmessage = async (e) => {
      const m = e.data;
      try {
        if (!T) {
          T = await import(m.tjs);
          T.env.allowLocalModels = false;        // always fetch from the Hub (then cached by the browser)
        }

        if (m.type === "loadLLM") {
          if (llm) { try { await llm.dispose(); } catch (err) {} llm = null; }
          const t0 = performance.now();
          llm = await T.pipeline("text-generation", m.id, { device: m.device, dtype: m.dtype, progress_callback: progress("llm") });
          // warm-up: the first run compiles GPU shaders, so do it now not on the user's first question
          await llm([{ role: "user", content: "Hi" }], { max_new_tokens: 1 });
          post({ type: "loaded", which: "llm", ms: performance.now() - t0 });
        }

        else if (m.type === "loadASR") {
          if (asr) { try { await asr.dispose(); } catch (err) {} asr = null; }
          const t0 = performance.now();
          asr = await T.pipeline("automatic-speech-recognition", m.id, { device: m.device, dtype: m.dtype, progress_callback: progress("asr") });
          await asr(new Float32Array(16000));      // warm-up with 1 second of silence
          post({ type: "loaded", which: "asr", ms: performance.now() - t0 });
        }

        else if (m.type === "transcribe") {
          const t0 = performance.now();
          const opts = m.multilingual ? { task: "transcribe", language: m.language || null } : {};
          const out = await asr(m.audio, opts);
          post({ type: "transcript", text: (out.text || "").trim(), ms: performance.now() - t0, seconds: m.audio.length / 16000 });
        }

        else if (m.type === "generate") {
          stopper = new T.InterruptableStoppingCriteria();
          let nTokens = 0, tFirst = null;
          const t0 = performance.now();
          const streamer = new T.TextStreamer(llm.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text) => post({ type: "token", text }),
            token_callback_function: () => { nTokens++; if (tFirst === null) tFirst = performance.now(); },
          });
          const out = await llm(m.messages, {
            max_new_tokens: m.maxTokens,
            do_sample: m.temperature > 0,
            temperature: m.temperature > 0 ? m.temperature : undefined,
            top_k: 40,
            repetition_penalty: 1.1,
            streamer,
            stopping_criteria: stopper,
          });
          const t1 = performance.now();
          const gen = out[0].generated_text;
          const text = Array.isArray(gen) ? gen[gen.length - 1].content : String(gen);
          const genSecs = tFirst ? (t1 - tFirst) / 1000 : 0;
          post({
            type: "done", text, tokens: nTokens,
            ttft: tFirst ? tFirst - t0 : 0,
            tps: genSecs > 0 ? (nTokens - 1) / genSecs : 0,
            total: t1 - t0,
          });
        }

        else if (m.type === "interrupt") {
          if (stopper) stopper.interrupt();
        }
      } catch (err) {
        post({ type: "error", where: m.type, message: (err && err.message) ? err.message : String(err) });
      }
    };
  }

  // ---------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------

  const S = {
    worker: null,
    device: "wasm",        // "webgpu" or "wasm"
    hasF16: false,
    llmReady: false,
    asrReady: false,
    busy: false,           // generating
    history: [],           // [{role, content}]
    results: [],           // experiment log rows
    llmLoadMs: 0, asrLoadMs: 0,
    rec: null,             // recording state
    speakBuffer: "",
    files: {},             // download progress per file
    conversation: false,   // hands-free mode
  };

  const $id = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------------

  const CSS = `
  #va-root{position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:#0f1222;color:#e8e9f3;font-size:15px}
  #va-root *{box-sizing:border-box}
  #va-head{display:flex;align-items:center;gap:12px;padding:10px 18px;background:#171a30;border-bottom:1px solid #2a2e4d}
  #va-head h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.2px}
  .va-badge{font-size:12px;padding:3px 9px;border-radius:999px;background:#2a2e4d;color:#c9cbe0;white-space:nowrap}
  .va-badge.ok{background:#173f2c;color:#7ee2a8}.va-badge.warn{background:#4a3a12;color:#ffd479}.va-badge.bad{background:#4a1820;color:#ff9aa8}
  #va-body{flex:1;display:flex;min-height:0}
  #va-side{width:320px;padding:16px;overflow-y:auto;background:#141729;border-right:1px solid #2a2e4d}
  #va-side h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#8c90b5;margin:18px 0 8px}
  #va-side h2:first-child{margin-top:0}
  #va-side label{display:block;font-size:13px;color:#b8bbd6;margin:8px 0 4px}
  #va-side select,#va-side input[type=range]{width:100%}
  #va-side select{background:#1e2240;color:#e8e9f3;border:1px solid #343a63;border-radius:8px;padding:7px}
  .va-row{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px;color:#c9cbe0}
  .va-btn{border:0;border-radius:10px;padding:9px 14px;font-weight:600;cursor:pointer;background:#5865f2;color:#fff;font-size:14px}
  .va-btn:hover{filter:brightness(1.1)}.va-btn:disabled{opacity:.45;cursor:not-allowed;filter:none}
  .va-btn.sec{background:#2a2e4d;color:#e8e9f3}.va-btn.red{background:#d9425a}
  .va-bar{height:6px;background:#262a48;border-radius:4px;overflow:hidden;margin:4px 0 2px}
  .va-bar>div{height:100%;width:0;background:linear-gradient(90deg,#5865f2,#9b6dff);transition:width .2s}
  .va-small{font-size:12px;color:#8c90b5}
  table.va-stats{width:100%;font-size:12.5px;border-collapse:collapse}
  table.va-stats td{padding:3px 0;color:#b8bbd6}table.va-stats td:last-child{text-align:right;color:#fff;font-variant-numeric:tabular-nums}
  #va-main{flex:1;display:flex;flex-direction:column;min-width:0}
  #va-chat{flex:1;overflow-y:auto;padding:22px 26px;display:flex;flex-direction:column;gap:12px}
  .va-msg{max-width:760px;padding:11px 15px;border-radius:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
  .va-msg.user{align-self:flex-end;background:#5865f2;color:#fff;border-bottom-right-radius:4px}
  .va-msg.assistant{align-self:flex-start;background:#1e2240;border:1px solid #2d3260;border-bottom-left-radius:4px}
  .va-msg.system{align-self:center;background:transparent;color:#8c90b5;font-size:13px;text-align:center;max-width:620px}
  .va-msg .meta{display:block;font-size:11px;color:#8c90b5;margin-top:6px}
  .va-msg.user .meta{color:#d6d9ff}
  .va-msg code{background:#0f1222;padding:1px 5px;border-radius:4px}
  #va-input{display:flex;gap:10px;padding:14px 22px;border-top:1px solid #2a2e4d;background:#141729;align-items:flex-end}
  #va-text{flex:1;resize:none;min-height:44px;max-height:140px;background:#1e2240;color:#e8e9f3;border:1px solid #343a63;border-radius:12px;padding:11px 13px;font:inherit}
  #va-mic{width:54px;height:44px;border-radius:12px;font-size:20px;padding:0}
  #va-mic.rec{background:#d9425a;animation:vapulse 1.2s infinite}
  @keyframes vapulse{0%{box-shadow:0 0 0 0 rgba(217,66,90,.6)}70%{box-shadow:0 0 0 12px rgba(217,66,90,0)}100%{box-shadow:0 0 0 0 rgba(217,66,90,0)}}
  #va-level{width:54px;height:4px;background:#262a48;border-radius:3px;overflow:hidden;margin-top:4px}
  #va-level>div{height:100%;width:0;background:#7ee2a8}
  .va-cursor{display:inline-block;width:8px;height:15px;background:#9b6dff;vertical-align:-2px;margin-left:2px;animation:vablink 1s steps(2) infinite}
  @keyframes vablink{50%{opacity:0}}
  `;

  function buildUI() {
    if ($id("va-root")) $id("va-root").remove();

    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.id = "va-root";
    root.innerHTML = `
      <div id="va-head">
        <h1>🎙️ Offline Voice Assistant</h1>
        <span class="va-small">Whisper + small language model, running in your browser</span>
        <span style="flex:1"></span>
        <span id="va-backend" class="va-badge">checking GPU…</span>
        <span id="va-online" class="va-badge">online</span>
      </div>
      <div id="va-body">
        <div id="va-side">
          <h2>1. Models</h2>
          <label for="va-llm">Language model</label>
          <select id="va-llm"></select>
          <label for="va-asr">Speech-to-text model</label>
          <select id="va-asr"></select>
          <label for="va-dev">Run on</label>
          <select id="va-dev">
            <option value="webgpu">GPU (WebGPU)</option>
            <option value="wasm">CPU (WebAssembly)</option>
          </select>
          <div style="display:flex;gap:8px;margin-top:12px">
            <button id="va-load" class="va-btn" style="flex:1">Load models</button>
          </div>
          <div id="va-prog" style="margin-top:10px"></div>

          <h2>2. Assistant</h2>
          <label for="va-persona">Persona</label>
          <select id="va-persona">
            <option value="assistant">Helpful assistant</option>
            <option value="tutor">AI module revision tutor</option>
            <option value="pirate">Pirate captain</option>
          </select>
          <label for="va-temp">Creativity (temperature): <b id="va-tempv">0.3</b></label>
          <input id="va-temp" type="range" min="0" max="1.2" step="0.1" value="0.3">
          <label for="va-max">Max reply length (tokens): <b id="va-maxv">200</b></label>
          <input id="va-max" type="range" min="32" max="512" step="16" value="200">

          <h2>3. Voice</h2>
          <div class="va-row"><input type="checkbox" id="va-speak" checked><label for="va-speak" style="margin:0">Read replies aloud</label></div>
          <div class="va-row"><input type="checkbox" id="va-auto" checked><label for="va-auto" style="margin:0">Auto-stop recording when I stop talking</label></div>
          <div class="va-row"><input type="checkbox" id="va-conv"><label for="va-conv" style="margin:0">Hands-free conversation mode</label></div>
          <label for="va-voice">Voice</label>
          <select id="va-voice"></select>
          <label for="va-file">…or ask with an audio file (wav / mp3 / webm)</label>
          <input id="va-file" type="file" accept="audio/*" disabled style="font-size:12px;color:#b8bbd6;width:100%">

          <h2>4. Performance (live)</h2>
          <table class="va-stats">
            <tr><td>Backend</td><td id="st-dev">–</td></tr>
            <tr><td>LLM load time</td><td id="st-llmload">–</td></tr>
            <tr><td>Whisper load time</td><td id="st-asrload">–</td></tr>
            <tr><td>Time to first token</td><td id="st-ttft">–</td></tr>
            <tr><td>Generation speed</td><td id="st-tps">–</td></tr>
            <tr><td>Tokens in last reply</td><td id="st-tok">–</td></tr>
            <tr><td>Transcription time</td><td id="st-asr">–</td></tr>
            <tr><td>Real-time factor</td><td id="st-rtf">–</td></tr>
          </table>
          <div style="display:flex;gap:8px;margin-top:10px">
            <button id="va-csv" class="va-btn sec" style="flex:1">Copy results (CSV)</button>
            <button id="va-clear" class="va-btn sec" style="flex:1">Clear chat</button>
          </div>
          <p class="va-small" style="margin-top:14px">Tip: once loaded, turn off Wi-Fi. It keeps working, because the models run on your machine.</p>
        </div>

        <div id="va-main">
          <div id="va-chat"></div>
          <div id="va-input">
            <textarea id="va-text" rows="1" placeholder="Load the models, then type here or press the mic…" disabled></textarea>
            <div>
              <button id="va-mic" class="va-btn" title="Click to talk (or hold Space)" disabled>🎤</button>
              <div id="va-level"><div></div></div>
            </div>
            <button id="va-send" class="va-btn" disabled>Send</button>
            <button id="va-stop" class="va-btn red" style="display:none">Stop</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(root);

    // fill selects
    const llmSel = $id("va-llm");
    LLMS.forEach((m, i) => llmSel.add(new Option(m.name, i)));
    llmSel.value = "1";
    const asrSel = $id("va-asr");
    ASRS.forEach((m, i) => asrSel.add(new Option(m.name, i)));

    // events
    $id("va-load").onclick = loadModels;
    $id("va-send").onclick = () => sendText();
    $id("va-stop").onclick = () => { S.worker.postMessage({ type: "interrupt" }); speechSynthesis.cancel(); };
    $id("va-mic").onclick = toggleRecording;
    $id("va-clear").onclick = () => { S.history = []; $id("va-chat").innerHTML = ""; sysMsg("Chat cleared."); };
    $id("va-csv").onclick = copyCSV;
    $id("va-temp").oninput = (e) => ($id("va-tempv").textContent = e.target.value);
    $id("va-max").oninput = (e) => ($id("va-maxv").textContent = e.target.value);
    $id("va-persona").onchange = () => sysMsg("Persona changed. It applies from your next message.");
    $id("va-conv").onchange = (e) => {
      S.conversation = e.target.checked;
      if (S.conversation) { $id("va-speak").checked = true; $id("va-auto").checked = true; sysMsg("Hands-free mode: press the mic once. After each spoken reply it listens again automatically."); }
    };
    $id("va-speak").onchange = (e) => { if (!e.target.checked) speechSynthesis.cancel(); };
    $id("va-file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) transcribeBlob(f); };

    const ta = $id("va-text");
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
    });
    ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 140) + "px"; });

    // hold Space (when not typing) = push-to-talk
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && document.activeElement !== ta && !e.repeat && !$id("va-mic").disabled && !S.rec) { e.preventDefault(); startRecording(true); }
    });
    document.addEventListener("keyup", (e) => {
      if (e.code === "Space" && S.rec && S.rec.pushToTalk) { e.preventDefault(); stopRecording(); }
    });

    // online / offline badge
    const upd = () => {
      const b = $id("va-online");
      if (navigator.onLine) { b.textContent = "online"; b.className = "va-badge"; }
      else { b.textContent = "OFFLINE ✓ still running locally"; b.className = "va-badge ok"; }
    };
    window.addEventListener("online", upd); window.addEventListener("offline", upd); upd();

    // voices
    const fillVoices = () => {
      const sel = $id("va-voice"); const keep = sel.value; sel.innerHTML = "";
      // English voices only, minus the macOS novelty voices
      const silly = /^(Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Fred|Good News|Jester|Junior|Organ|Ralph|Superstar|Trinoids|Whisper|Wobble|Zarvox)\b/;
      const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang) && !silly.test(v.name));
      voices.forEach((v) => sel.add(new Option(v.name + (v.localService ? "" : " (online voice)"), v.name)));
      if (keep) sel.value = keep;
      // prefer a local voice
      if (!keep) { const local = voices.find((v) => v.localService); if (local) sel.value = local.name; }
    };
    if ("speechSynthesis" in window) { fillVoices(); speechSynthesis.onvoiceschanged = fillVoices; }

    sysMsg("Welcome! This assistant runs <b>two neural networks inside your browser</b>: Whisper (OpenAI) for speech-to-text " +
           "and a small language model for replies. Nothing is sent to an AI server.<br><br>Choose models on the left and press <b>Load models</b>. " +
           "The first load downloads the model files (then the browser caches them).");
  }

  // ---------------------------------------------------------------------------------
  // Chat display helpers
  // ---------------------------------------------------------------------------------

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function mini_md(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  }
  function addMsg(role, html) {
    const d = document.createElement("div");
    d.className = "va-msg " + role;
    d.innerHTML = html;
    $id("va-chat").appendChild(d);
    scrollDown();
    return d;
  }
  function sysMsg(html) { return addMsg("system", html); }
  function scrollDown() { const c = $id("va-chat"); c.scrollTop = c.scrollHeight; }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms"; }

  // ---------------------------------------------------------------------------------
  // GPU detection and worker start
  // ---------------------------------------------------------------------------------

  async function detectBackend() {
    const b = $id("va-backend");
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          S.device = "webgpu";
          S.hasF16 = adapter.features.has("shader-f16");
        }
      }
    } catch (e) { console.warn("WebGPU check failed:", e); }

    S.hasGPU = S.device === "webgpu";
    const devSel = $id("va-dev");
    if (S.hasGPU) {
      b.textContent = "WebGPU available" + (S.hasF16 ? " · fp16" : ""); b.className = "va-badge ok";
    } else {
      b.textContent = "No WebGPU · using CPU (slower)"; b.className = "va-badge warn";
      devSel.options[0].disabled = true;
    }
    devSel.value = S.device;
    devSel.onchange = () => { S.device = devSel.value; applyDevice(); sysMsg("Backend set to <b>" + S.device + "</b>. Press <b>Load models</b> to reload on it."); };
    applyDevice();
  }

  function applyDevice() {
    // the 1B model only ships a GPU fp16 (q4f16) version
    const ok1B = S.device === "webgpu" && S.hasF16;
    [...$id("va-llm").options].forEach((o) => { if (LLMS[o.value].gpuOnly) o.disabled = !ok1B; });
    if (!ok1B && LLMS[$id("va-llm").value].gpuOnly) $id("va-llm").value = "1";
    if (!S.hasGPU && $id("va-llm").value === "1") $id("va-llm").value = "0";
    $id("st-dev").textContent = S.device === "webgpu" ? "WebGPU" : "WASM (CPU)";
  }

  function startWorker() {
    const src = "(" + workerMain.toString() + ")();";
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    S.worker = new Worker(url, { type: "module" });
    S.worker.onmessage = onWorkerMessage;
    S.worker.onerror = (e) => { console.error("Worker error:", e.message || e); sysMsg("⚠️ Worker error: " + esc(String(e.message || e))); };
  }

  // choose the quantisation ("dtype") for each model + device
  function llmDtype(m) {
    if (S.device === "webgpu") return S.hasF16 ? "q4f16" : "q4";
    return "q8";
  }
  function asrDtype() {
    // NB: the fp16 merged decoder fails in ONNX Runtime ("Subgraph output (logits) is an outer scope value"),
    // so we use the 4-bit decoder, as in the official Whisper WebGPU demo.
    if (S.device === "webgpu") return { encoder_model: "fp32", decoder_model_merged: "q4" };
    return { encoder_model: "q8", decoder_model_merged: "q8" };
  }

  // ---------------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------------

  let loadPending = 0;

  function loadModels() {
    const llm = LLMS[$id("va-llm").value];
    const asr = ASRS[$id("va-asr").value];
    S.llmReady = S.asrReady = false;
    S.files = {};
    setInputs(false);
    $id("va-load").disabled = true;
    $id("va-load").textContent = "Loading…";
    $id("va-prog").innerHTML =
      `<div class="va-small">Language model: <span id="pl-llm">starting…</span></div><div class="va-bar"><div id="pb-llm"></div></div>
       <div class="va-small" style="margin-top:6px">Whisper: <span id="pl-asr">waiting…</span></div><div class="va-bar"><div id="pb-asr"></div></div>`;
    sysMsg(`Loading <b>${esc(llm.name)}</b> and <b>${esc(asr.name)}</b> on <b>${S.device}</b>…`);
    loadPending = 2;
    // load one after the other (less memory pressure / bandwidth contention)
    S.worker.postMessage({ type: "loadLLM", tjs: TJS_URL, id: llm.id, device: S.device, dtype: llmDtype(llm) });
  }

  function updateProgress(which, msg) {
    const key = which + ":" + msg.file;
    if (msg.status === "progress" || msg.status === "done") {
      if (msg.total) S.files[key] = { loaded: msg.status === "done" ? msg.total : msg.loaded, total: msg.total, which };
    }
    let loaded = 0, total = 0;
    for (const k in S.files) if (S.files[k].which === which) { loaded += S.files[k].loaded; total += S.files[k].total; }
    const bar = $id("pb-" + which), label = $id("pl-" + which);
    if (!bar) return;
    if (total > 0) {
      bar.style.width = (100 * loaded / total).toFixed(1) + "%";
      label.textContent = (loaded / 1e6).toFixed(0) + " / " + (total / 1e6).toFixed(0) + " MB";
    }
    if (msg.status === "ready") label.textContent = "compiling on " + S.device + "…";
  }

  // ---------------------------------------------------------------------------------
  // Messages from the worker
  // ---------------------------------------------------------------------------------

  let current = null;   // the assistant message being streamed

  function onWorkerMessage(e) {
    const m = e.data;
    switch (m.type) {
      case "progress": updateProgress(m.which, m); break;

      case "loaded":
        if (m.which === "llm") {
          S.llmReady = true; S.llmLoadMs = m.ms;
          $id("pb-llm").style.width = "100%"; $id("pl-llm").textContent = "ready ✓";
          $id("st-llmload").textContent = fmtMs(m.ms);
          // now Whisper
          const asr = ASRS[$id("va-asr").value];
          $id("pl-asr").textContent = "starting…";
          S.worker.postMessage({ type: "loadASR", tjs: TJS_URL, id: asr.id, device: S.device, dtype: asrDtype() });
        } else {
          S.asrReady = true; S.asrLoadMs = m.ms;
          $id("pb-asr").style.width = "100%"; $id("pl-asr").textContent = "ready ✓";
          $id("st-asrload").textContent = fmtMs(m.ms);
        }
        if (--loadPending === 0) {
          $id("va-load").disabled = false; $id("va-load").textContent = "Reload models";
          setInputs(true);
          sysMsg(`✅ Both models loaded (LLM ${fmtMs(S.llmLoadMs)}, Whisper ${fmtMs(S.asrLoadMs)}). Type, click 🎤, or hold <b>Space</b> to talk.`);
          $id("va-text").focus();
        }
        break;

      case "transcript":
        $id("st-asr").textContent = fmtMs(m.ms) + " for " + m.seconds.toFixed(1) + " s";
        $id("st-rtf").textContent = (m.ms / 1000 / m.seconds).toFixed(2) + "×";
        S.lastAsr = { ms: m.ms, seconds: m.seconds };
        if (S.pendingVoice) { S.pendingVoice.remove(); S.pendingVoice = null; }
        if (!m.text || /^\[.*\]$|^\(.*\)$/.test(m.text)) {   // Whisper outputs e.g. "[BLANK_AUDIO]" for silence
          sysMsg("Didn't catch that - try again.");
          setInputs(true);
          if (S.conversation) setTimeout(() => startRecording(false), 600);
          return;
        }
        sendText(m.text, true);
        break;

      case "token":
        if (!current) return;
        current.text += m.text;
        current.body.innerHTML = mini_md(current.text) + '<span class="va-cursor"></span>';
        scrollDown();
        speakStream(m.text);
        break;

      case "done":
        finishReply(m);
        break;

      case "error":
        console.error("Model error (" + m.where + "):", m.message);
        sysMsg("⚠️ Error during <b>" + esc(m.where) + "</b>: " + esc(m.message));
        S.busy = false;
        $id("va-load").disabled = false; $id("va-load").textContent = "Load models";
        $id("va-stop").style.display = "none"; $id("va-send").style.display = "";
        setInputs(S.llmReady);
        break;
    }
  }

  function setInputs(on) {
    $id("va-text").disabled = !on;
    $id("va-send").disabled = !on;
    $id("va-mic").disabled = !(on && S.asrReady);
    $id("va-file").disabled = !(on && S.asrReady);
  }

  // ---------------------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------------------

  function sendText(text, fromVoice) {
    if (S.busy || !S.llmReady) return;
    const ta = $id("va-text");
    text = (text !== undefined ? text : ta.value).trim();
    if (!text) return;
    ta.value = ""; ta.style.height = "auto";

    const meta = fromVoice && S.lastAsr ? `<span class="meta">🎤 heard by Whisper in ${fmtMs(S.lastAsr.ms)}</span>` : "";
    addMsg("user", esc(text) + meta);
    S.history.push({ role: "user", content: text });

    // keep only the last few turns: small models have small context windows and slow down with long prompts
    const recent = S.history.slice(-8);
    const messages = [{ role: "system", content: PERSONAS[$id("va-persona").value] }].concat(recent);

    const el = addMsg("assistant", '<span class="va-cursor"></span>');
    const body = document.createElement("span"); el.innerHTML = ""; el.appendChild(body);
    body.innerHTML = '<span class="va-cursor"></span>';
    current = { el, body, text: "" };

    S.busy = true;
    S.speakBuffer = "";
    speechSynthesis.cancel();
    setInputs(false);
    $id("va-send").style.display = "none"; $id("va-stop").style.display = "";

    S.worker.postMessage({
      type: "generate", messages,
      temperature: parseFloat($id("va-temp").value),
      maxTokens: parseInt($id("va-max").value, 10),
    });
  }

  function finishReply(m) {
    const llm = LLMS[$id("va-llm").value];
    if (current) {
      const text = current.text.trim() || m.text.trim();
      current.body.innerHTML = mini_md(text);
      const meta = document.createElement("span"); meta.className = "meta";
      meta.textContent = `${m.tokens} tokens · ${m.tps.toFixed(1)} tok/s · first token ${fmtMs(m.ttft)}`;
      current.el.appendChild(meta);
      S.history.push({ role: "assistant", content: text });
      current = null;
    }
    flushSpeech();

    $id("st-ttft").textContent = fmtMs(m.ttft);
    $id("st-tps").textContent = m.tps.toFixed(1) + " tok/s";
    $id("st-tok").textContent = m.tokens;

    S.results.push({
      time: new Date().toISOString(), model: llm.id, device: S.device,
      llm_load_ms: Math.round(S.llmLoadMs), ttft_ms: Math.round(m.ttft), tokens: m.tokens,
      tokens_per_s: m.tps.toFixed(2), total_ms: Math.round(m.total),
      asr_model: ASRS[$id("va-asr").value].id,
      asr_ms: S.lastAsr ? Math.round(S.lastAsr.ms) : "", audio_s: S.lastAsr ? S.lastAsr.seconds.toFixed(2) : "",
    });
    S.lastAsr = null;

    S.busy = false;
    $id("va-stop").style.display = "none"; $id("va-send").style.display = "";
    setInputs(true);
    if (!S.conversation) $id("va-text").focus();
  }

  function copyCSV() {
    if (!S.results.length) { sysMsg("No results yet - ask a few questions first."); return; }
    const cols = Object.keys(S.results[0]);
    const csv = [cols.join(",")].concat(S.results.map((r) => cols.map((c) => r[c]).join(","))).join("\n");
    navigator.clipboard.writeText(csv).then(
      () => sysMsg(`Copied ${S.results.length} rows of results to the clipboard. Paste into Excel / Sheets for your report.`),
      () => { console.log(csv); sysMsg("Clipboard blocked - results printed to the console instead."); }
    );
  }

  // ---------------------------------------------------------------------------------
  // Text-to-speech: speak each sentence as soon as it is complete (lower latency)
  // ---------------------------------------------------------------------------------

  function speakStream(chunk) {
    if (!$id("va-speak").checked || !("speechSynthesis" in window)) return;
    S.speakBuffer += chunk;
    const match = S.speakBuffer.match(/^([\s\S]*?[.!?…:\n])(\s+)([\s\S]*)$/);
    if (match && match[1].trim().length > 2) {
      say(match[1]);
      S.speakBuffer = match[3];
    }
  }
  function flushSpeech() {
    if ($id("va-speak").checked && S.speakBuffer.trim()) say(S.speakBuffer);
    S.speakBuffer = "";
    if (S.conversation) {
      // listen again once all speech has finished
      const wait = () => {
        if (speechSynthesis.speaking || speechSynthesis.pending) return setTimeout(wait, 250);
        if (S.conversation && !S.rec && !S.busy) startRecording(false);
      };
      setTimeout(wait, 400);
    }
  }
  function say(text) {
    const clean = text.replace(/[*_`#>]/g, "").trim();
    if (!clean) return;
    const u = new SpeechSynthesisUtterance(clean);
    const v = speechSynthesis.getVoices().find((x) => x.name === $id("va-voice").value);
    if (v) u.voice = v;
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }

  // ---------------------------------------------------------------------------------
  // Recording from the microphone -> 16 kHz mono Float32Array for Whisper
  // ---------------------------------------------------------------------------------

  function toggleRecording() {
    if (S.rec) stopRecording(); else startRecording(false);
  }

  async function startRecording(pushToTalk) {
    if (S.rec || S.busy || !S.asrReady) return;
    speechSynthesis.cancel();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    } catch (err) {
      console.error("Microphone error:", err);
      sysMsg("⚠️ Microphone not available (" + esc(err.name || String(err)) + "). Check the browser's mic permission.");
      S.conversation = false; $id("va-conv").checked = false;
      return;
    }

    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };

    // level meter + silence detection
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const an = ctx.createAnalyser(); an.fftSize = 1024;
    src.connect(an);
    const buf = new Float32Array(an.fftSize);
    const started = performance.now();
    let heardSpeech = false, lastLoud = performance.now();

    const rec = { recorder, stream, ctx, chunks, pushToTalk, raf: 0 };
    S.rec = rec;

    const tick = () => {
      an.getFloatTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      $id("va-level").firstChild.style.width = Math.min(100, rms * 600) + "%";
      const now = performance.now();
      if (rms > 0.02) { heardSpeech = true; lastLoud = now; }
      const auto = $id("va-auto").checked && !pushToTalk;
      if (auto && heardSpeech && now - lastLoud > 1300) return stopRecording();     // 1.3 s of silence after speech
      if (auto && !heardSpeech && now - started > 8000) return stopRecording(true); // nobody spoke
      if (now - started > 30000) return stopRecording();                            // Whisper window = 30 s
      rec.raf = requestAnimationFrame(tick);
    };

    recorder.onstop = async () => {
      cancelAnimationFrame(rec.raf);
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      $id("va-level").firstChild.style.width = "0";
      if (rec.discard) { setInputs(true); return; }
      transcribeBlob(new Blob(chunks, { type: recorder.mimeType }));
    };

    recorder.start();
    $id("va-mic").classList.add("rec"); $id("va-mic").textContent = "⏹";
    $id("va-text").disabled = true; $id("va-send").disabled = true;
    tick();
  }

  function stopRecording(discard) {
    const rec = S.rec;
    if (!rec) return;
    S.rec = null;
    rec.discard = !!discard;
    if (discard) sysMsg("No speech heard - stopped listening.");
    $id("va-mic").classList.remove("rec"); $id("va-mic").textContent = "🎤";
    $id("va-mic").disabled = true;
    if (rec.recorder.state !== "inactive") rec.recorder.stop();
  }

  // Recording or uploaded file -> Whisper
  async function transcribeBlob(blob) {
    if (S.busy || !S.asrReady) return;
    setInputs(false);
    try {
      let audio = await blobTo16k(blob);
      if (audio.length < 16000 * 0.3) { sysMsg("Recording too short."); setInputs(true); return; }
      if (audio.length > 16000 * 30) { audio = audio.slice(0, 16000 * 30); sysMsg("Audio longer than 30 s - using the first 30 s."); }
      S.pendingVoice = sysMsg("🎧 Transcribing with Whisper…");
      const asr = ASRS[$id("va-asr").value];
      S.worker.postMessage({ type: "transcribe", audio, multilingual: asr.multilingual }, [audio.buffer]);
    } catch (err) {
      console.error("Audio decode error:", err);
      sysMsg("⚠️ Could not decode the audio: " + esc(String(err.message || err)));
      setInputs(true);
    }
  }

  // Decode the compressed recording and resample to 16 kHz mono (what Whisper expects)
  async function blobTo16k(blob) {
    const arr = await blob.arrayBuffer();
    const ctx = new AudioContext({ sampleRate: 16000 });   // browser resamples for us
    const decoded = await ctx.decodeAudioData(arr);
    ctx.close();
    if (decoded.numberOfChannels === 1) return new Float32Array(decoded.getChannelData(0));
    const a = decoded.getChannelData(0), b = decoded.getChannelData(1);
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
    return out;
  }

  // ---------------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------------

  async function start() {
    buildUI();
    startWorker();
    await detectBackend();
  }

  if (typeof AB !== "undefined" && AB.world) {
    // Running on Ancient Brain
    AB.drawRunControls = false;
    AB.world.newRun = function () { start(); };
  } else {
    // Running stand-alone (local testing)
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
  }
})();
