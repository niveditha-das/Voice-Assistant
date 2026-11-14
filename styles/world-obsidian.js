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
//  - Visual design: black-and-crimson style based on the "ThoughtLab" style reference on Refero
//      https://styles.refero.design/style/82d52a5f-b1bb-4a69-91a3-15a7eb8bbe99
//    Font: Space Grotesk by Florian Karsten, via Google Fonts https://fonts.google.com/specimen/Space+Grotesk
//    Obsidian-sphere shader: raymarched signed distance field, technique by Inigo Quilez
//      https://iquilezles.org/articles/distfunctions/
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
    loading: false, transcribing: false, level: 0, tpsHist: [], progressText: "",
    chats: [], chatId: null,
  };

  const $id = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------------

  // Design: "obsidian monument with crimson signal" - pure black canvas, ash (#ccc) body, frost (#fff) display,
  // graphite (#4c4c4c) meta, ONE crimson pill for the primary action, transparent components, hairlines, no shadows.
  // A dark obsidian liquid sphere is the only decoration.
  // Style reference: https://styles.refero.design/style/82d52a5f-b1bb-4a69-91a3-15a7eb8bbe99
  // Font: Space Grotesk (substitute for Sui) from Google Fonts https://fonts.google.com/specimen/Space+Grotesk
  const CSS = `
  #va-root{--void:#000;--ash:#ccc;--frost:#fff;--graphite:#4c4c4c;--crimson:#fc1c46;--hair:rgba(255,255,255,.14);
    --ease:cubic-bezier(0.19,1,0.22,1);--sideW:280px;
    position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;overflow:hidden;background:var(--void);color:var(--ash);
    font-family:"Space Grotesk","Sui",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;font-weight:400;line-height:1.15;
    -webkit-font-smoothing:antialiased}
  #va-root *{box-sizing:border-box}
  #va-root ::selection{background:var(--crimson);color:#fff}
  #va-root ::-webkit-scrollbar{width:6px}#va-root ::-webkit-scrollbar-thumb{background:var(--graphite);border-radius:9999px}

  #va-liquid{position:absolute;inset:0;width:100%;height:100%;display:block;z-index:0}
  #va-body{position:relative;z-index:1;flex:1;display:flex;min-height:0}

  /* pills only. Ghost by default; ONE crimson fill for the primary action in view */
  .va-btn{border:1px solid var(--hair);border-radius:9999px;padding:0 22px;height:40px;background:transparent;color:var(--frost);font:inherit;font-size:15px;
    text-transform:uppercase;letter-spacing:-.2px;cursor:pointer;white-space:nowrap;transition:border-color .6s var(--ease),opacity .4s ease,background .6s var(--ease)}
  .va-btn:hover{border-color:var(--frost)}
  .va-btn:disabled{opacity:.25;cursor:not-allowed;border-color:var(--hair)}
  .va-btn.solid,#va-send:not(:disabled){background:var(--crimson);border-color:var(--crimson);color:#fff;padding:0 30px}
  .va-btn.solid:hover,#va-send:not(:disabled):hover{border-color:var(--crimson);opacity:.88}
  .va-btn.sec{font-size:12px;height:34px;padding:0 16px}
  .va-btn.red{border-color:var(--frost)}
  .va-link{background:none;border:0;padding:0;font:inherit;font-size:14px;color:var(--frost);cursor:pointer;text-align:left;
    text-decoration:underline;text-decoration-color:transparent;text-underline-offset:4px;transition:text-decoration-color .4s ease}
  .va-link:hover{text-decoration-color:var(--frost)}

  /* ---- collapsible sidebar ---- */
  #va-side{width:var(--sideW);flex:none;display:flex;flex-direction:column;border-right:1px solid var(--hair);background:rgba(0,0,0,.6);transition:margin-left 1s var(--ease)}
  #va-root.side-hidden #va-side{margin-left:calc(-1 * var(--sideW) - 1px)}
  #va-brand{height:72px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-size:15px;color:var(--frost);font-weight:500}
  #va-brand span{display:flex;align-items:center;gap:9px}
  #va-brand span::before{content:"";width:10px;height:10px;border-radius:9999px;background:var(--crimson)}
  #va-collapse{font-size:14px;color:var(--graphite)}
  #va-collapse:hover{color:var(--frost)}
  #va-new{margin:9px 22px 36px;width:calc(100% - 44px)}
  .va-label{font-size:10px;text-transform:uppercase;color:var(--graphite);padding:0 22px 9px;letter-spacing:.3px}
  #va-hist{flex:1;overflow-y:auto;padding:0 0 12px}
  .va-h{display:flex;align-items:center;gap:9px;padding:9px 22px;cursor:pointer;color:var(--ash);transition:color .4s ease}
  .va-h:hover{color:var(--frost)}
  .va-h.on{color:var(--frost)}
  .va-h.on span::before{content:"— "}
  .va-h span{flex:1;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .va-h button{opacity:0;font-size:16px;line-height:1;color:var(--graphite);text-decoration:none}
  .va-h:hover button{opacity:1}
  .va-empty{font-size:14px;color:var(--graphite);padding:0 22px}
  #va-side-foot{border-top:1px solid var(--hair);padding:22px 22px 9px;display:flex;justify-content:space-between;align-items:center}
  #va-status{font-size:10px;text-transform:uppercase;color:var(--graphite);display:flex;align-items:center;gap:7px;letter-spacing:.3px}
  #va-status::before{content:"";width:6px;height:6px;border-radius:9999px;border:1px solid var(--graphite)}
  #va-status.ok{color:var(--ash)}#va-status.ok::before{background:var(--frost);border-color:var(--frost)}
  #va-credit{padding:0 22px 22px;font-size:10px;color:var(--graphite);line-height:1.25}
  #va-vlabel{display:none}

  /* ---- main ---- */
  #va-main{position:relative;flex:1;display:flex;flex-direction:column;min-width:0}
  #va-hero{flex:none;height:72px;display:flex;align-items:center;gap:22px;padding:0 72px}
  #va-menu{display:none;color:var(--frost)}
  #va-root.side-hidden #va-menu{display:inline-block}
  #va-state{font-size:15px;text-transform:uppercase;color:var(--ash);display:flex;align-items:baseline;gap:9px}
  #va-state small{font-size:10px;color:var(--graphite);font-variant-numeric:tabular-nums}

  #va-chat{flex:1;overflow-y:auto;padding:36px 72px 22px;display:flex;flex-direction:column;gap:36px}
  .va-msg{max-width:680px;line-height:1.25;white-space:pre-wrap;word-wrap:break-word;font-size:18px;color:var(--ash);animation:vain 1s var(--ease)}
  @keyframes vain{from{opacity:0;transform:translateY(12px)}}
  .va-msg.user{align-self:flex-end;text-align:left;color:var(--frost);font-size:27px;line-height:1.2;max-width:620px}
  .va-msg.user::before,.va-msg.assistant::before{display:block;font-size:10px;text-transform:uppercase;color:var(--graphite);margin-bottom:9px;line-height:1;letter-spacing:.3px}
  .va-msg.user::before{content:"You"}
  .va-msg.assistant{align-self:flex-start;width:min(680px,100%)}
  .va-msg.assistant::before{content:"Assistant"}
  .va-msg.system{align-self:flex-start;font-size:14px;color:var(--graphite)}
  .va-msg code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;color:var(--frost)}
  .va-msg b{font-weight:700;color:var(--frost)}

  #va-welcome{max-width:900px;margin:auto 0;padding-bottom:65px;animation:vain 1s var(--ease)}
  #va-welcome h3{font-size:91px;font-weight:700;line-height:.92;letter-spacing:-1.82px;text-transform:uppercase;color:var(--frost);margin:0 0 36px}
  #va-welcome p{font-size:18px;line-height:1.1;margin:0 0 36px;max-width:440px}
  .va-chips{display:flex;flex-wrap:wrap;gap:9px}
  .va-chip{font:inherit;font-size:15px;text-transform:uppercase;color:var(--frost);height:40px;padding:0 22px;border-radius:9999px;background:transparent;border:1px solid var(--hair);cursor:pointer;
    transition:border-color .6s var(--ease)}
  .va-chip:hover{border-color:var(--frost)}
  .va-small{font-size:10px;text-transform:uppercase;color:var(--graphite)}
  .va-bar{height:1px;background:var(--hair);overflow:hidden;max-width:440px;margin-top:9px}
  .va-bar>div{height:100%;width:0;background:var(--frost);transition:width 1s var(--ease)}

  /* input: hairline only */
  #va-input{display:flex;gap:9px;padding:22px 72px 36px;align-items:center}
  #va-text{flex:1;resize:none;min-height:44px;max-height:140px;background:transparent;color:var(--frost);border:0;border-bottom:1px solid var(--frost);border-radius:0;outline:none;
    padding:9px 2px;font:inherit;font-size:18px;line-height:1.25;margin-right:13px}
  #va-text::placeholder{color:var(--graphite)}
  #va-text:disabled{border-bottom-color:var(--graphite)}
  #va-mic{position:relative;width:40px;height:40px;padding:0;font-size:0;flex:none}
  #va-mic::before{content:"";position:absolute;left:50%;top:50%;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;border-radius:9999px;background:var(--frost);transition:transform .6s var(--ease)}
  #va-mic:hover::before{transform:scale(1.3)}
  #va-mic.rec{border-color:var(--frost)}
  #va-mic.rec::before{background:var(--crimson);transform:scale(1.2)}
  #va-mic.rec::after{content:"";position:absolute;inset:-1px;border-radius:9999px;border:1px solid var(--frost);animation:varing 1.6s var(--ease) infinite}
  @keyframes varing{from{transform:scale(1);opacity:1}to{transform:scale(1.8);opacity:0}}
  #va-level{display:none}
  .va-cursor{display:inline-block;width:8px;height:17px;background:var(--frost);vertical-align:-3px;margin-left:3px;animation:vablink 1s steps(2) infinite}
  @keyframes vablink{50%{opacity:0}}
  .va-dots span{display:inline-block;width:6px;height:6px;margin-right:6px;border-radius:9999px;background:var(--frost);animation:vadot 1.2s var(--ease) infinite}
  .va-dots span:nth-child(2){animation-delay:.15s}.va-dots span:nth-child(3){animation-delay:.3s}
  @keyframes vadot{0%,80%,100%{opacity:.2}40%{opacity:1}}

  /* ---- settings: full-height overlay panel on the void ---- */
  #va-scrim{position:absolute;inset:0;z-index:20;background:rgba(0,0,0,.6);opacity:0;pointer-events:none;transition:opacity .8s var(--ease)}
  #va-settings{position:absolute;z-index:21;top:0;right:0;bottom:0;width:420px;max-width:100%;background:#000;border-left:1px solid var(--hair);overflow-y:auto;
    transform:translateX(101%);transition:transform 1s var(--ease)}
  #va-root.set-open #va-scrim{opacity:1;pointer-events:auto}
  #va-root.set-open #va-settings{transform:none}
  #va-set-head{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 36px;position:sticky;top:0;background:#000;z-index:1;
    font-size:27px;font-weight:700;letter-spacing:-.5px;text-transform:uppercase;color:var(--frost)}
  .va-card{padding:22px 36px 36px;border-top:1px solid var(--hair)}
  .va-card h2{display:flex;gap:9px;font-size:15px;font-weight:400;text-transform:uppercase;color:var(--ash);margin:0 0 22px}
  .va-card h2 b{font-weight:400;color:var(--graphite)}
  #va-settings label{display:block;font-size:10px;text-transform:uppercase;color:var(--graphite);margin:22px 0 7px;letter-spacing:.3px}
  #va-settings h2 + label{margin-top:0}
  #va-settings select{width:100%;appearance:none;-webkit-appearance:none;border:0;border-bottom:1px solid var(--frost);border-radius:0;padding:7px 22px 9px 2px;
    background:transparent url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23fff' fill='none'/%3E%3C/svg%3E") no-repeat right 2px center;
    font:inherit;font-size:15px;color:var(--frost);cursor:pointer;text-overflow:ellipsis}
  #va-settings select:focus{outline:none}
  #va-settings select option{background:#000;color:var(--frost)}
  #va-settings input[type=range]{width:100%;-webkit-appearance:none;appearance:none;height:1px;background:var(--frost);outline:none;margin:12px 0 4px}
  #va-settings input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:9999px;background:var(--frost);cursor:pointer}
  #va-settings .va-lbl{display:flex;justify-content:space-between}#va-settings .va-lbl b{font-weight:400;color:var(--frost)}
  #va-settings .va-switch{display:flex;align-items:center;justify-content:space-between;gap:9px;margin:0;padding:13px 0;border-bottom:1px solid var(--hair);
    font-size:15px;color:var(--ash);cursor:pointer}
  #va-settings .va-switch input{display:none}
  #va-settings .va-switch i{flex:none;position:relative;width:34px;height:18px;border-radius:9999px;border:1px solid var(--frost);transition:background .6s var(--ease)}
  #va-settings .va-switch i::after{content:"";position:absolute;left:3px;top:3px;width:10px;height:10px;border-radius:9999px;background:var(--frost);transition:transform .6s var(--ease),background .6s var(--ease)}
  #va-settings .va-switch input:checked + i{background:var(--frost)}
  #va-settings .va-switch input:checked + i::after{transform:translateX(16px);background:#000}
  #va-reload{width:100%;margin-top:22px}
  #va-file{font-size:12px;color:var(--graphite);width:100%}
  #va-file::file-selector-button{font:inherit;font-size:12px;text-transform:uppercase;color:var(--frost);background:transparent;border:1px solid var(--hair);border-radius:9999px;padding:7px 16px;margin-right:12px;cursor:pointer}
  .va-stats{display:grid;grid-template-columns:1fr 1fr;column-gap:22px}
  .va-stat{padding:12px 0;border-bottom:1px solid var(--hair)}
  .va-stat .k{font-size:10px;text-transform:uppercase;color:var(--graphite);letter-spacing:.3px}
  .va-stat .v{font-size:27px;font-weight:700;letter-spacing:-.5px;color:var(--frost);margin-top:7px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #va-spark{width:100%;height:52px;display:block;margin-top:22px}

  #va-toast{position:absolute;z-index:30;left:50%;bottom:110px;transform:translate(-50%,16px);opacity:0;pointer-events:none;background:#000;color:var(--frost);border:1px solid var(--hair);
    font-size:14px;padding:11px 22px;border-radius:9999px;transition:opacity .8s var(--ease),transform .8s var(--ease);max-width:80%;text-align:center}
  #va-toast.on{opacity:1;transform:translate(-50%,0)}

  @media (max-width:1200px){ #va-welcome h3{font-size:72px;letter-spacing:-1.4px} }
  @media (max-width:900px){
    #va-side{position:absolute;z-index:15;left:0;top:0;bottom:0;background:#000}
    #va-menu{display:inline-block}
    #va-welcome h3{font-size:54px;letter-spacing:-1px}
    #va-chat,#va-input{padding-left:22px;padding-right:22px}#va-hero{padding:0 22px}
  }
  `;

  function renderWelcome() {
    const loaded = S.llmReady && S.asrReady;
    const body = loaded
      ? `<div class="va-chips">
           <button class="va-chip" data-q="Explain what a neural network is, simply.">What is a neural network?</button>
           <button class="va-chip" data-q="Why is ReLU better than sigmoid for deep networks?">ReLU vs sigmoid</button>
           <button class="va-chip" data-q="Tell me a short joke about computers.">Tell me a joke</button>
         </div>`
      : S.loading
        ? `<div class="va-small" id="va-wprog">Downloading the models…</div><div class="va-bar"><div id="va-wbar"></div></div>`
        : `<button id="va-load" class="va-btn solid">Load models</button>`;
    $id("va-chat").innerHTML = `
      <div id="va-welcome">
        <h3>${loaded ? "Ask me anything." : "An AI that lives in your browser."}</h3>
        <p>${loaded ? "Type below, or press the mic and speak."
                    : "Whisper (speech-to-text) and a small language model are downloaded to this page and run on your own device. No AI server is used."}</p>
        ${body}
      </div>`;
    const lb = $id("va-load"); if (lb) lb.onclick = loadModels;
  }

  function buildUI() {
    if ($id("va-root")) $id("va-root").remove();

    if (!document.getElementById("va-font")) {   // Space Grotesk, as a substitute for Sui (falls back to system font offline)
      const lk = document.createElement("link"); lk.id = "va-font"; lk.rel = "stylesheet";
      lk.href = "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&display=swap";
      document.head.appendChild(lk);
    }
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.id = "va-root";
    root.innerHTML = `
      <canvas id="va-liquid"></canvas>
      <div id="va-body">
        <div id="va-side">
          <div id="va-brand"><span>Offline voice assistant</span><button id="va-collapse" class="va-link" title="Hide sidebar">Hide</button></div>
          <button id="va-new" class="va-btn">New chat</button>
          <div class="va-label">Chats</div>
          <div id="va-hist"></div>
          <div id="va-side-foot">
            <button id="va-open-settings" class="va-link">Settings</button>
            <span id="va-status">not loaded</span>
          </div>
          <div id="va-credit">Built with Transformers.js, Whisper and SmolLM2. Runs on this device.</div>
        </div>

        <div id="va-main">
          <div id="va-hero">
            <button id="va-menu" class="va-link" title="Show sidebar">Chats</button>
            <div id="va-state">Ready to load</div>
          </div>
          <div id="va-vlabel">on-device AI · 2 models</div>
          <div id="va-chat"></div>
          <div id="va-input">
            <textarea id="va-text" rows="1" placeholder="Load the models to start…" disabled></textarea>
            <button id="va-send" class="va-btn" disabled>Send</button>
            <button id="va-stop" class="va-btn red" style="display:none">Stop</button>
            <button id="va-mic" class="va-btn" title="Click to talk (or hold Space)" disabled>Mic</button>
            <div id="va-level"><div></div></div>
          </div>
        </div>
      </div>

      <div id="va-scrim"></div>
      <div id="va-settings">
        <div id="va-set-head"><span>Settings</span><button id="va-close-settings" class="va-link">Close</button></div>

        <div class="va-card">
          <h2><b>01</b> Models</h2>
          <label for="va-llm">Language model</label>
          <select id="va-llm"></select>
          <label for="va-asr">Speech-to-text</label>
          <select id="va-asr"></select>
          <label for="va-dev">Run on <span id="va-backend"></span></label>
          <select id="va-dev">
            <option value="webgpu">GPU (WebGPU)</option>
            <option value="wasm">CPU (WebAssembly)</option>
          </select>
          <button id="va-reload" class="va-btn solid">Load models</button>
        </div>

        <div class="va-card">
          <h2><b>02</b> Assistant</h2>
          <label for="va-persona">Persona</label>
          <select id="va-persona">
            <option value="assistant">Helpful assistant</option>
            <option value="tutor">AI module revision tutor</option>
            <option value="pirate">Pirate captain</option>
          </select>
          <label for="va-temp" class="va-lbl"><span>Creativity (temperature)</span><b id="va-tempv">0.3</b></label>
          <input id="va-temp" type="range" min="0" max="1.2" step="0.1" value="0.3">
          <label for="va-max" class="va-lbl"><span>Max reply length (tokens)</span><b id="va-maxv">200</b></label>
          <input id="va-max" type="range" min="32" max="512" step="16" value="200">
        </div>

        <div class="va-card">
          <h2><b>03</b> Voice</h2>
          <label class="va-switch"><span>Read replies aloud</span><input type="checkbox" id="va-speak" checked><i></i></label>
          <label class="va-switch"><span>Auto-stop when I pause</span><input type="checkbox" id="va-auto" checked><i></i></label>
          <label class="va-switch"><span>Hands-free conversation</span><input type="checkbox" id="va-conv"><i></i></label>
          <label for="va-voice">Speaking voice</label>
          <select id="va-voice"></select>
          <label for="va-file">Ask with an audio file</label>
          <input id="va-file" type="file" accept="audio/*" disabled>
        </div>

        <div class="va-card" style="border:0">
          <h2><b>04</b> Performance</h2>
          <div class="va-stats">
            <div class="va-stat"><div class="k">Backend</div><div class="v" id="st-dev">–</div></div>
            <div class="va-stat"><div class="k">Tokens / s</div><div class="v" id="st-tps">–</div></div>
            <div class="va-stat"><div class="k">First token</div><div class="v" id="st-ttft">–</div></div>
            <div class="va-stat"><div class="k">Reply tokens</div><div class="v" id="st-tok">–</div></div>
            <div class="va-stat"><div class="k">LLM load</div><div class="v" id="st-llmload">–</div></div>
            <div class="va-stat"><div class="k">Whisper load</div><div class="v" id="st-asrload">–</div></div>
            <div class="va-stat"><div class="k">Transcription</div><div class="v" id="st-asr">–</div></div>
            <div class="va-stat"><div class="k">Real-time factor</div><div class="v" id="st-rtf">–</div></div>
          </div>
          <canvas id="va-spark" width="688" height="104" title="Tokens per second for each reply"></canvas>
          <div style="display:flex;gap:8px;margin-top:24px">
            <button id="va-csv" class="va-btn sec" style="flex:1">Copy results (CSV)</button>
            <button id="va-clear" class="va-btn sec" style="flex:1">Delete all chats</button>
          </div>
        </div>
      </div>
      <div id="va-toast"></div>`;
    document.body.appendChild(root);

    // fill selects
    const llmSel = $id("va-llm");
    LLMS.forEach((m, i) => llmSel.add(new Option(m.name, i)));
    llmSel.value = "1";
    const asrSel = $id("va-asr");
    ASRS.forEach((m, i) => asrSel.add(new Option(m.name, i)));

    // events
    const openSet = (on) => root.classList.toggle("set-open", on);
    $id("va-open-settings").onclick = () => { if (narrow()) setSide(false); openSet(true); };
    $id("va-close-settings").onclick = () => openSet(false);
    $id("va-scrim").onclick = () => openSet(false);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") openSet(false); });
    const narrow = () => window.innerWidth <= 900;
    const setSide = (show) => { root.classList.toggle("side-hidden", !show); if (!narrow()) localStorage.setItem("va-side", show ? "1" : "0"); };
    setSide(narrow() ? false : localStorage.getItem("va-side") !== "0");
    $id("va-collapse").onclick = () => setSide(false);
    $id("va-menu").onclick = () => setSide(true);

    $id("va-reload").onclick = () => { openSet(false); loadModels(); };
    $id("va-new").onclick = () => { if (narrow()) setSide(false); newChat(); };
    $id("va-send").onclick = () => sendText();
    $id("va-stop").onclick = () => { S.worker.postMessage({ type: "interrupt" }); speechSynthesis.cancel(); };
    $id("va-mic").onclick = toggleRecording;
    $id("va-clear").onclick = () => { if (confirm("Delete all saved chats?")) { S.chats = []; saveChats(); newChat(); } };
    $id("va-csv").onclick = copyCSV;
    $id("va-temp").oninput = (e) => ($id("va-tempv").textContent = e.target.value);
    $id("va-max").oninput = (e) => ($id("va-maxv").textContent = e.target.value);
    $id("va-persona").onchange = () => toast("Persona applies from your next message.");
    $id("va-conv").onchange = (e) => {
      S.conversation = e.target.checked;
      if (S.conversation) { $id("va-speak").checked = true; $id("va-auto").checked = true; toast("Hands-free: press the mic once. It listens again after each reply."); }
    };
    $id("va-speak").onchange = (e) => { if (!e.target.checked) speechSynthesis.cancel(); };
    $id("va-file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) { openSet(false); transcribeBlob(f); } };

    const ta = $id("va-text");
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
    });
    ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 140) + "px"; });

    // hold Space (when not typing) = push-to-talk
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && document.activeElement !== ta && !root.classList.contains("set-open") && !e.repeat && !$id("va-mic").disabled && !S.rec) { e.preventDefault(); startRecording(true); }
    });
    document.addEventListener("keyup", (e) => {
      if (e.code === "Space" && S.rec && S.rec.pushToTalk) { e.preventDefault(); stopRecording(); }
    });

    // going offline is the proof that it runs locally, so say so when it happens
    window.addEventListener("offline", () => toast("You are offline. The assistant keeps working: the models run on this device."));

    // voices
    const fillVoices = () => {
      const sel = $id("va-voice"); const keep = sel.value; sel.innerHTML = "";
      // English voices only, minus the macOS novelty voices
      const silly = /^(Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Fred|Good News|Jester|Junior|Organ|Ralph|Superstar|Trinoids|Whisper|Wobble|Zarvox)\b/;
      const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang) && !silly.test(v.name));
      voices.forEach((v) => sel.add(new Option(v.name + (v.localService ? "" : " (online voice)"), v.name)));
      if (keep) sel.value = keep;
      if (!keep) { const local = voices.find((v) => v.localService); if (local) sel.value = local.name; }
    };
    if ("speechSynthesis" in window) { fillVoices(); speechSynthesis.onvoiceschanged = fillVoices; }

    $id("va-chat").addEventListener("click", (e) => {
      const chip = e.target.closest(".va-chip");
      if (chip) sendText(chip.dataset.q);
    });

    S.chats = loadChats();
    renderHist();
    renderWelcome();
    startOrb();
  }

  // ---------------------------------------------------------------------------------
  // Chat history - saved in the browser's localStorage (stays on this device)
  // ---------------------------------------------------------------------------------

  const STORE = "va-chats-v1";
  function loadChats() { try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch (e) { return []; } }
  function saveChats() { try { localStorage.setItem(STORE, JSON.stringify(S.chats.slice(0, 50))); } catch (e) { console.warn("Could not save chats:", e); } renderHist(); }

  function renderHist() {
    const box = $id("va-hist");
    if (!S.chats.length) { box.innerHTML = '<div class="va-empty">No chats yet.</div>'; return; }
    box.innerHTML = "";
    S.chats.forEach((c) => {
      const row = document.createElement("div");
      row.className = "va-h" + (c.id === S.chatId ? " on" : "");
      row.innerHTML = `<span></span><button class="va-link" title="Delete chat">×</button>`;
      row.firstChild.textContent = c.title;
      row.onclick = () => openChat(c.id);
      row.lastChild.onclick = (e) => { e.stopPropagation(); deleteChat(c.id); };
      box.appendChild(row);
    });
  }

  function newChat() {
    if (S.busy) return;
    S.chatId = null; S.history = [];
    renderWelcome(); renderHist();
  }

  function openChat(id) {
    if (S.busy || S.rec) return;
    const c = S.chats.find((x) => x.id === id); if (!c) return;
    S.chatId = id; S.history = c.messages.slice();
    $id("va-chat").innerHTML = "";
    c.messages.forEach((m) => addMsg(m.role, m.role === "user" ? esc(m.content) : mini_md(m.content)));
    renderHist();
    if (window.innerWidth <= 900) $id("va-root").classList.add("side-hidden");
  }

  function deleteChat(id) {
    S.chats = S.chats.filter((x) => x.id !== id);
    saveChats();
    if (S.chatId === id) newChat();
  }

  // store the current conversation (creates the chat on its first message)
  function storeCurrent() {
    let c = S.chats.find((x) => x.id === S.chatId);
    if (!c) {
      const first = (S.history.find((m) => m.role === "user") || { content: "New chat" }).content;
      c = { id: "c" + Date.now(), title: first.length > 42 ? first.slice(0, 40) + "…" : first, messages: [] };
      S.chatId = c.id; S.chats.unshift(c);
    }
    c.messages = S.history.slice();
    c.ts = Date.now();
    // most recently used first
    S.chats = [c].concat(S.chats.filter((x) => x !== c));
    saveChats();
  }

  let toastTimer = 0;
  function toast(text) {
    const t = $id("va-toast"); t.textContent = text; t.classList.add("on");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("on"), 3200);
  }

  // ---------------------------------------------------------------------------------
  // Chat display helpers
  // ---------------------------------------------------------------------------------

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function mini_md(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  }
  function addMsg(role, html) {
    const w = $id("va-welcome"); if (w) w.remove();
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
      b.textContent = "· WebGPU found";
    } else {
      b.textContent = "· no WebGPU, CPU only";
      devSel.options[0].disabled = true;
    }
    devSel.value = S.device;
    devSel.onchange = () => { S.device = devSel.value; applyDevice(); toast("Press Load models to reload on the " + (S.device === "webgpu" ? "GPU." : "CPU.")); };
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
    S.worker.onerror = (e) => { console.error("Worker error:", e.message || e); sysMsg("Worker error: " + esc(String(e.message || e))); };
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
    $id("va-reload").disabled = true;
    loadPending = 2;
    S.loading = true; S.progressText = "";
    $id("va-status").textContent = "loading"; $id("va-status").className = "";
    if (!S.history.length) renderWelcome();
    // load one after the other (less memory pressure / bandwidth contention)
    S.worker.postMessage({ type: "loadLLM", tjs: TJS_URL, id: llm.id, device: S.device, dtype: llmDtype(llm) });
  }

  function updateProgress(which, msg) {
    const key = which + ":" + msg.file;
    if (msg.status === "progress" || msg.status === "done") {
      if (msg.total) S.files[key] = { loaded: msg.status === "done" ? msg.total : msg.loaded, total: msg.total, which };
    }
    let loaded = 0, total = 0;
    for (const k in S.files) { loaded += S.files[k].loaded; total += S.files[k].total; }
    if (total > 0) S.progressText = (loaded / 1e6).toFixed(0) + " / " + (total / 1e6).toFixed(0) + " MB";
    if (msg.status === "ready") S.progressText = "preparing " + (which === "llm" ? "language model" : "Whisper") + "…";
    const bar = $id("va-wbar");
    if (bar && total > 0) bar.style.width = (100 * loaded / total).toFixed(1) + "%";
    
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
          $id("st-llmload").textContent = fmtMs(m.ms);
          // now Whisper
          const asr = ASRS[$id("va-asr").value];
          S.worker.postMessage({ type: "loadASR", tjs: TJS_URL, id: asr.id, device: S.device, dtype: asrDtype() });
        } else {
          S.asrReady = true; S.asrLoadMs = m.ms;
          $id("st-asrload").textContent = fmtMs(m.ms);
        }
        if (--loadPending === 0) {
          S.loading = false;
          $id("va-reload").disabled = false; $id("va-reload").textContent = "Reload models";
          setInputs(true);
          $id("va-text").placeholder = "Message, or hold Space to talk";
          $id("va-status").textContent = "ready"; $id("va-status").className = "ok";
          if (!S.history.length) renderWelcome();
          $id("va-text").focus();
        }
        break;

      case "transcript":
        $id("st-asr").textContent = fmtMs(m.ms) + " for " + m.seconds.toFixed(1) + " s";
        $id("st-rtf").textContent = (m.ms / 1000 / m.seconds).toFixed(2) + "×";
        S.lastAsr = { ms: m.ms, seconds: m.seconds };
        S.transcribing = false;
        if (!m.text || /^\[.*\]$|^\(.*\)$/.test(m.text)) {   // Whisper outputs e.g. "[BLANK_AUDIO]" for silence
          toast("Didn't catch that. Try again.");
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
        sysMsg("Error during <b>" + esc(m.where) + "</b>: " + esc(m.message));
        S.loading = false; S.transcribing = false;
        S.busy = false;
        $id("va-reload").disabled = false;
        if (!S.llmReady) { $id("va-status").textContent = "not loaded"; $id("va-status").className = ""; if (!S.history.length) renderWelcome(); }
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

    addMsg("user", esc(text));
    S.history.push({ role: "user", content: text });
    storeCurrent();

    // keep only the last few turns: small models have small context windows and slow down with long prompts
    const recent = S.history.slice(-8);
    const messages = [{ role: "system", content: PERSONAS[$id("va-persona").value] }].concat(recent);

    const el = addMsg("assistant", '<span class="va-cursor"></span>');
    const body = document.createElement("span"); el.innerHTML = ""; el.appendChild(body);
    body.innerHTML = '<span class="va-dots"><span></span><span></span><span></span></span>';
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
      S.history.push({ role: "assistant", content: text });
      storeCurrent();
      current = null;
    }
    flushSpeech();

    $id("st-ttft").textContent = fmtMs(m.ttft);
    $id("st-tps").textContent = m.tps.toFixed(1) + " tok/s";
    $id("st-tok").textContent = m.tokens;

    S.tpsHist.push(m.tps);
    drawSpark();
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
    if (!S.results.length) { toast("No results yet. Ask a few questions first."); return; }
    const cols = Object.keys(S.results[0]);
    const csv = [cols.join(",")].concat(S.results.map((r) => cols.map((c) => r[c]).join(","))).join("\n");
    navigator.clipboard.writeText(csv).then(
      () => toast(`Copied ${S.results.length} rows of results to the clipboard.`),
      () => { console.log(csv); toast("Clipboard blocked. Results printed to the console."); }
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
      toast("Microphone not available (" + (err.name || String(err)) + "). Check the browser's mic permission.");
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
      S.level = rms;
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
    $id("va-mic").classList.add("rec");
    $id("va-text").disabled = true; $id("va-send").disabled = true;
    tick();
  }

  function stopRecording(discard) {
    const rec = S.rec;
    if (!rec) return;
    S.rec = null;
    rec.discard = !!discard;
    if (discard) toast("No speech heard. Stopped listening.");
    $id("va-mic").classList.remove("rec");
    $id("va-mic").disabled = true;
    if (rec.recorder.state !== "inactive") rec.recorder.stop();
  }

  // Recording or uploaded file -> Whisper
  async function transcribeBlob(blob) {
    if (S.busy || !S.asrReady) return;
    setInputs(false);
    try {
      let audio = await blobTo16k(blob);
      if (audio.length < 16000 * 0.3) { toast("Recording too short."); setInputs(true); return; }
      if (audio.length > 16000 * 30) { audio = audio.slice(0, 16000 * 30); toast("Audio longer than 30 s. Using the first 30 s."); }
      S.transcribing = true;
      const asr = ASRS[$id("va-asr").value];
      S.worker.postMessage({ type: "transcribe", audio, multilingual: asr.multilingual }, [audio.buffer]);
    } catch (err) {
      console.error("Audio decode error:", err);
      sysMsg("Could not decode the audio: " + esc(String(err.message || err)));
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
  // Background: a WebGL shader showing one floating object that reacts to what the AI is doing.
  // Like the "canvas webgl" example World on Ancient Brain, the JS builds a GLSL program, compiles it,
  // and the GPU runs it for every pixel.
  // ---------------------------------------------------------------------------------

  const STATES = {
    off:          { label: "Ready to load", sub: "Pick your models on the left and press Load models.", stage: null },
    loading:      { label: "Downloading", sub: "Fetching model weights into your browser. Cached for next time.", stage: null },
    idle:         { label: "Ready", sub: "Type below, press the mic, or hold Space and talk.", stage: null },
    listening:    { label: "Listening", sub: "Speak now — recording stops when you pause.", stage: "mic" },
    transcribing: { label: "Transcribing", sub: "Whisper is turning your speech into text, on this device.", stage: "asr" },
    thinking:     { label: "Thinking", sub: "The language model is reading the conversation.", stage: "llm" },
    writing:      { label: "Writing", sub: "Generating the reply token by token on your GPU.", stage: "llm" },
    speaking:     { label: "Speaking", sub: "Reading the reply aloud with an on-device voice.", stage: "tts" },
  };

  function currentState() {
    if (S.rec) return "listening";
    if (S.transcribing) return "transcribing";
    if (S.busy) return current && current.text ? "writing" : "thinking";
    if ("speechSynthesis" in window && speechSynthesis.speaking) return "speaking";
    if (S.loading) return "loading";
    return S.llmReady ? "idle" : "off";
  }

  // The object: an obsidian liquid sphere, drawn by a raymarching WebGL fragment shader
  // (the GPU traces a ray for every pixel against a signed distance function). The surface is
  // near-black glass that only shows itself through blue / violet / amber reflections at the rim.
  // It ripples harder while the AI listens, thinks, writes or speaks; a white ring shows download
  // progress; it steps back while you chat.
  // Technique: raymarching signed distance fields, Inigo Quilez https://iquilezles.org/articles/distfunctions/
  const FRAG = `
    precision highp float;
    uniform vec2 r; uniform vec2 c; uniform float t; uniform float e; uniform float p; uniform float d;
    mat2 rot(float a){ float s = sin(a), co = cos(a); return mat2(co, -s, s, co); }
    float map(vec3 q){
      q.xz *= rot(t*0.18); q.xy *= rot(t*0.11);
      float amp = 0.05 + 0.13*e;
      float wob = sin(q.x*3.1 + t*1.3) * sin(q.y*2.7 - t*1.1) * sin(q.z*3.4 + t*0.9)
                + 0.5 * sin(q.x*6.3 - t*1.7) * sin(q.y*5.9 + t*1.4) * sin(q.z*6.7 - t*1.2);
      return length(q) - 1.0 + wob * amp;
    }
    vec3 nrm(vec3 q){ vec2 h = vec2(0.002, 0.0);
      return normalize(vec3(map(q+h.xyy)-map(q-h.xyy), map(q+h.yxy)-map(q-h.yxy), map(q+h.yyx)-map(q-h.yyx))); }
    vec3 env(vec3 v){                                       // fake studio: cool key left, amber kicker right, violet floor
      vec3 blue   = vec3(0.20, 0.38, 1.00) * smoothstep(0.15, 0.9, -v.x) * smoothstep(-0.4, 0.6, v.y);
      vec3 amber  = vec3(1.00, 0.58, 0.18) * smoothstep(0.30, 0.95,  v.x) * smoothstep(-0.1, 0.8, v.y);
      vec3 violet = vec3(0.52, 0.28, 0.95) * smoothstep(0.20, 0.9, -v.y);
      return blue + amber + violet * 0.7;
    }
    void main(){
      vec2 uv = (gl_FragCoord.xy - c) / r.y;
      uv *= mix(2.3, 4.6, d);                                // smaller (further away) while chatting
      vec3 col = vec3(0.0);
      vec3 ro = vec3(0.0, 0.0, 3.4), rd = normalize(vec3(uv, -1.9));
      float tt = 0.0, hit = 0.0;
      for (int i = 0; i < 72; i++) {
        float dist = map(ro + rd*tt);
        if (dist < 0.0015) { hit = 1.0; break; }
        tt += dist * 0.8; if (tt > 7.0) break;
      }
      if (hit > 0.5) {
        vec3 q = ro + rd*tt, n = nrm(q), rf = reflect(rd, n);
        float fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        vec3 m = vec3(0.008) + env(rf) * (0.07 + 1.15*fres) * (0.8 + 0.6*e);
        m += vec3(1.0) * pow(max(dot(rf, normalize(vec3(-0.4, 0.7, 0.6))), 0.0), 80.0) * 0.9;   // glossy highlight
        col = m * (1.0 - 0.7*d);
      }
      if (p > 0.0) {                                         // download progress ring
        float rr = length(uv), ang = atan(uv.x, uv.y) / 6.2832 + 0.5;
        float ring = smoothstep(0.016, 0.0, abs(rr - 0.86));
        col = mix(col, vec3(1.0), ring * (ang < p ? 0.95 : 0.15));
      }
      gl_FragColor = vec4(col, 1.0);
    }`;

  function startOrb() {
    const cv = $id("va-liquid");
    let gl = null;
    try { gl = cv.getContext("webgl", { antialias: false, premultipliedAlpha: false }); } catch (e) {}
    const U = {};
    if (gl) {
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      try {
        const prog = gl.createProgram();
        gl.attachShader(prog, sh(gl.VERTEX_SHADER, "attribute vec2 v; void main(){ gl_Position = vec4(v, 0.0, 1.0); }"));
        gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
        gl.linkProgram(prog); gl.useProgram(prog);
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, "v"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        ["r", "c", "t", "e", "p", "d"].forEach((k) => (U[k] = gl.getUniformLocation(prog, k)));
      } catch (err) { console.warn("WebGL object disabled:", err); gl = null; }
    }

    let shown = null, energy = 0, T = 0, last = performance.now(), dim = 0, cx = -1, cy = -1;

    function frame(now) {
      if (!document.body.contains(cv)) return;           // UI was rebuilt
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      const st = currentState(), def = STATES[st];

      if (st !== shown) { shown = st; $id("va-state").innerHTML = def.label + "<small></small>"; }
      if (st === "loading") { const sm = $id("va-state").lastChild; if (sm.textContent !== S.progressText) sm.textContent = S.progressText; }

      let target = 0.0;
      if (st === "listening") target = 0.25 + Math.min(1, S.level * 14);
      else if (st === "thinking" || st === "transcribing") target = 0.55;
      else if (st === "writing") target = 0.7;
      else if (st === "speaking") target = 0.4 + 0.35 * Math.abs(Math.sin(now / 170));
      else if (st === "loading") target = 0.2;
      energy += (target - energy) * Math.min(1, dt * 2.5);
      T += dt * (0.6 + energy * 1.6);
      dim += ((S.history.length ? 1 : 0) - dim) * Math.min(1, dt * 1.5);

      if (gl) {
        const scale = Math.min(window.devicePixelRatio || 1, 2) * 0.5;   // half resolution keeps the GPU free for the models
        const w = Math.round(cv.clientWidth * scale), h = Math.round(cv.clientHeight * scale);
        if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; gl.viewport(0, 0, w, h); }
        // place the object in the main column: right of the welcome text, drifting to the far right while chatting
        const main = $id("va-main").getBoundingClientRect(), rootR = cv.getBoundingClientRect();
        const tx = (main.left - rootR.left + main.width * (0.66 + 0.27 * dim)) * scale;
        const ty = (rootR.height - (main.top - rootR.top + main.height * (0.5 - 0.14 * dim))) * scale;
        if (cx < 0) { cx = tx; cy = ty; }
        cx += (tx - cx) * Math.min(1, dt * 3); cy += (ty - cy) * Math.min(1, dt * 3);
        gl.uniform2f(U.r, w, h); gl.uniform2f(U.c, cx, cy); gl.uniform1f(U.t, T); gl.uniform1f(U.e, energy); gl.uniform1f(U.d, dim);
        gl.uniform1f(U.p, st === "loading" ? Math.max(0.001, loadFraction()) : 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function loadFraction() {
    let loaded = 0, total = 0;
    for (const k in S.files) { loaded += S.files[k].loaded; total += S.files[k].total; }
    return total ? loaded / total : 0;
  }

  // small bar chart of tokens/second for each reply (good for the report's results section)
  function drawSpark() {
    const cv = $id("va-spark"), g = cv.getContext("2d"), W = cv.width, H = cv.height;
    const data = S.tpsHist.slice(-24);
    g.clearRect(0, 0, W, H);
    if (!data.length) return;
    const max = Math.max(...data) * 1.15, bw = W / 24;
    data.forEach((v, i) => {
      const h = Math.max(3, (v / max) * (H - 6)), x = i * bw + 2, y = H - h;
      const grad = g.createLinearGradient(0, y, 0, H);
      grad.addColorStop(0, "#fff"); grad.addColorStop(1, "#fff");
      g.fillStyle = grad;
      g.fillRect(x, y, bw - 4, h);
    });
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
