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
//  - Visual design: monochrome editorial style based on the "monopo saigon" style reference on Refero
//      https://styles.refero.design/style/3e52dd36-6ab1-48c6-bc40-47ef6d33abc2
//    Font: Inter by Rasmus Andersson, via Google Fonts https://fonts.google.com/specimen/Inter
//    Liquid hero shader: fbm + domain warping technique by Inigo Quilez https://iquilezles.org/articles/warp/
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
    loading: false, transcribing: false, level: 0, tpsHist: [],
  };

  const $id = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------------

  // Design: monochrome editorial system (black / white / greys, 0px corners, 75px pills, hairline borders,
  // no shadows) with ONE chromatic element - the iridescent liquid hero (green -> amber -> oxblood).
  // Style reference: https://styles.refero.design/style/3e52dd36-6ab1-48c6-bc40-47ef6d33abc2
  // Font: Inter (substitute for Roobert) from Google Fonts https://fonts.google.com/specimen/Inter
  const CSS = `
  #va-root{--ink:#000;--paper:#fff;--inkstone:#181818;--felt:#6d6d6d;--slate:#636363;--ash:#9a9a9a;--pewter:#808080;--hair:#e6e6e6;
    --ease:cubic-bezier(0.19,1,0.22,1);
    position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;overflow:hidden;background:var(--paper);color:var(--ink);
    font-family:"Inter","Roobert",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:16px;line-height:1.21;
    -webkit-font-smoothing:antialiased}
  #va-root *{box-sizing:border-box}
  #va-root ::selection{background:#000;color:#fff}
  #va-root ::-webkit-scrollbar{width:6px}#va-root ::-webkit-scrollbar-thumb{background:#d0d0d0}

  /* top bar: transparent, 66px, wordmark left, status centre, labels right */
  #va-head{height:66px;flex:none;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:0 28px;border-bottom:1px solid var(--ink)}
  #va-head .mark{font-size:16px;font-weight:400;letter-spacing:0;display:flex;align-items:center;gap:14px}
  #va-head .mark i{font-style:normal;color:var(--felt)}
  #va-head .mid{display:flex;gap:28px}
  #va-head .right{justify-self:end;display:flex;gap:28px;font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  .va-badge{font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:.6px;color:var(--ink);white-space:nowrap;display:inline-flex;align-items:center;gap:8px}
  .va-badge::before{content:"";width:6px;height:6px;border-radius:75px;background:var(--ash);transition:background .8s var(--ease)}
  .va-badge.ok::before{background:var(--ink)}
  .va-badge.warn{color:var(--felt)}.va-badge.warn::before{background:transparent;border:1px solid var(--ink)}
  .va-link{background:none;border:0;padding:0;font:inherit;color:inherit;text-transform:inherit;letter-spacing:inherit;cursor:pointer;transition:opacity .4s ease}
  .va-link:hover{opacity:.5}
  #va-toggle{display:none}

  /* body */
  #va-body{flex:1;display:flex;min-height:0}
  #va-side{width:320px;flex:none;overflow-y:auto;border-right:1px solid var(--ink)}
  .va-card{padding:28px 28px 34px;border-bottom:1px solid var(--hair)}
  .va-card h2{display:flex;align-items:baseline;gap:14px;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:.6px;margin:0 0 28px;color:var(--ink)}
  .va-card h2 b{font-weight:400;color:var(--felt);font-variant-numeric:tabular-nums}
  #va-side label{display:block;font-size:12px;line-height:1.19;color:var(--felt);margin:20px 0 8px}
  #va-side h2 + label{margin-top:0}
  #va-side select{width:100%;appearance:none;-webkit-appearance:none;border:0;border-bottom:1px solid var(--ink);border-radius:0;padding:8px 22px 9px 0;
    background:transparent url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23000' fill='none'/%3E%3C/svg%3E") no-repeat right 2px center;
    font:inherit;font-size:14px;color:var(--ink);cursor:pointer;transition:border-color .4s ease;text-overflow:ellipsis}
  #va-side select:focus{outline:none;border-bottom-width:2px;padding-bottom:8px}
  #va-side input[type=range]{width:100%;-webkit-appearance:none;appearance:none;height:1px;background:var(--ink);outline:none;margin:12px 0 4px}
  #va-side input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:75px;background:var(--ink);border:0;cursor:pointer;transition:transform .8s var(--ease)}
  #va-side input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.4)}
  #va-side .va-lbl{display:flex;justify-content:space-between}#va-side .va-lbl b{font-weight:400;color:var(--ink);font-variant-numeric:tabular-nums}

  #va-side .va-switch{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:0;padding:12px 0;border-bottom:1px solid var(--hair);font-size:16px;color:var(--ink);cursor:pointer}
  #va-side .va-switch input{display:none}
  #va-side .va-switch i{flex:none;position:relative;width:34px;height:18px;border-radius:75px;border:1px solid var(--ink);transition:background .8s var(--ease)}
  #va-side .va-switch i::after{content:"";position:absolute;left:3px;top:3px;width:10px;height:10px;border-radius:75px;background:var(--ink);transition:transform .8s var(--ease),background .8s var(--ease)}
  #va-side .va-switch input:checked + i{background:var(--ink)}
  #va-side .va-switch input:checked + i::after{transform:translateX(16px);background:var(--paper)}

  /* pills: the only rounded shape (75px) */
  .va-btn{border:1px solid var(--ink);border-radius:75px;padding:11px 33px;background:transparent;color:var(--ink);font:inherit;font-size:16px;font-weight:400;cursor:pointer;
    transition:letter-spacing .8s var(--ease),background .8s var(--ease),color .8s var(--ease),opacity .4s ease;white-space:nowrap}
  .va-btn:hover{letter-spacing:.6px}
  .va-btn:disabled{opacity:.3;cursor:not-allowed;letter-spacing:0}
  .va-btn.solid{background:var(--ink);color:var(--paper)}
  .va-btn.sec{padding:9px 20px;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
  .va-btn.sec:hover{letter-spacing:1px}
  .va-btn.red{background:rgba(55,55,55,.78);border-color:var(--paper);color:var(--paper)}
  #va-load{width:100%;margin-top:28px}

  .va-prog-row{margin-top:20px}
  .va-prog-row .top{display:flex;justify-content:space-between;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--felt)}
  .va-prog-row .top span:last-child{color:var(--ink);font-variant-numeric:tabular-nums}
  .va-bar{height:1px;background:var(--hair);margin-top:8px;overflow:hidden}
  .va-bar>div{height:100%;width:0;background:var(--ink);transition:width 1.25s var(--ease)}
  .va-small{font-size:12px;color:var(--felt);line-height:1.5}
  #va-file{font-size:12px;color:var(--felt);width:100%}
  #va-file::file-selector-button{font:inherit;font-size:12px;color:var(--ink);background:transparent;border:1px solid var(--ink);border-radius:75px;padding:6px 16px;margin-right:12px;cursor:pointer}

  .va-stats{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--ink);border-left:1px solid var(--hair)}
  .va-stat{padding:12px 12px 14px;border-right:1px solid var(--hair);border-bottom:1px solid var(--hair)}
  .va-stat .k{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--felt)}
  .va-stat .v{font-size:29px;font-weight:300;line-height:1;margin-top:10px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.5px}
  #va-spark{width:100%;height:52px;display:block;margin-top:20px}
  .va-cap{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--felt);margin-top:8px}

  /* main column */
  #va-main{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--paper)}

  /* THE hero: the only chromatic surface. Liquid iridescent WebGL canvas behind white type. */
  #va-hero{position:relative;flex:none;height:300px;background:#000;color:#fff;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end;padding:28px 40px 28px}
  #va-liquid{position:absolute;inset:0;width:100%;height:100%;display:block}
  #va-hero > *:not(#va-liquid):not(#va-badge){position:relative}
  #va-state{font-size:78px;font-weight:300;line-height:1.1;letter-spacing:-2px;transition:opacity .8s var(--ease)}
  #va-state small{display:block;font-size:16px;font-weight:400;letter-spacing:0;line-height:1.21;margin-top:12px;color:rgba(255,255,255,.85);max-width:520px}
  #va-pipe{display:flex;align-items:center;gap:8px;margin-top:28px;flex-wrap:wrap}
  .pp{display:flex;align-items:baseline;gap:10px;padding:8px 20px;border-radius:75px;border:1px solid rgba(255,255,255,.3);font-size:12px;transition:background .8s var(--ease),color .8s var(--ease),border-color .8s var(--ease)}
  .pp .ic{display:none}
  .pp .t{display:flex;gap:10px;align-items:baseline}
  .pp .t span{font-size:11px;color:rgba(255,255,255,.65);font-variant-numeric:tabular-nums}
  .pp.on{background:#fff;color:#000;border-color:#fff}
  .pp.on .t span{color:var(--felt)}
  .pp-arrow{width:20px;height:1px;background:rgba(255,255,255,.3);font-size:0;transition:background .8s var(--ease)}
  .pp-arrow.on{background:#fff}

  /* rotating circular text badge */
  #va-badge{position:absolute;right:40px;top:28px;width:108px;height:108px;animation:varot 18s linear infinite}
  #va-badge text{font-size:9.4px;letter-spacing:2.4px;text-transform:uppercase;fill:#fff}
  @keyframes varot{to{transform:rotate(360deg)}}

  #va-chat{flex:1;overflow-y:auto;padding:46px 40px 20px;display:flex;flex-direction:column;gap:28px}
  .va-msg{max-width:720px;line-height:1.58;white-space:pre-wrap;word-wrap:break-word;font-size:18px;animation:vain 1.25s var(--ease)}
  @keyframes vain{from{opacity:0;transform:translateY(14px)}}
  .va-msg.user{align-self:flex-end;background:#000;color:#fff;padding:14px 20px;font-size:16px}
  .va-msg.assistant{align-self:flex-start;padding-top:14px;border-top:1px solid #000;width:min(720px,100%)}
  .va-msg.assistant::before{content:"Assistant";display:block;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--felt);margin-bottom:12px;line-height:1}
  .va-msg.system{align-self:flex-start;font-size:12px;line-height:1.5;color:var(--felt);max-width:640px}
  .va-msg.system b{font-weight:400;color:var(--ink)}
  .va-msg .meta{display:block;font-size:11px;color:var(--felt);margin-top:12px;font-variant-numeric:tabular-nums;text-transform:uppercase;letter-spacing:.4px}
  .va-msg.user .meta{color:var(--ash)}
  .va-msg code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;background:#f2f2f2;padding:1px 5px}
  .va-msg b{font-weight:600}

  #va-welcome{max-width:760px;padding:0 0 20px;animation:vain 1.25s var(--ease)}
  #va-welcome h3{font-size:54px;font-weight:300;line-height:1.1;letter-spacing:-1.4px;margin:0 0 28px}
  #va-welcome p{font-size:18px;line-height:1.58;color:var(--inkstone);margin:0 0 28px;max-width:600px}
  #va-welcome p b{font-weight:400;border-bottom:1px solid #000}
  .va-chips{display:flex;flex-wrap:wrap;gap:8px}
  .va-chip{font:inherit;font-size:16px;color:#000;padding:11px 26px;border-radius:75px;background:transparent;border:1px solid #000;cursor:pointer;transition:letter-spacing .8s var(--ease),background .8s var(--ease),color .8s var(--ease)}
  .va-chip:hover{background:#000;color:#fff}
  .va-feats{display:flex;gap:28px;margin-top:40px;flex-wrap:wrap}
  .va-feats div{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--felt)}

  #va-input{display:flex;gap:14px;padding:14px 40px 12px;align-items:center;border-top:1px solid #000}
  #va-inbox{flex:1;display:flex;align-items:center;gap:14px}
  #va-text{flex:1;resize:none;min-height:46px;max-height:140px;background:transparent;color:#000;border:0;border-radius:0;outline:none;padding:12px 0;font:inherit;font-size:18px;line-height:1.3}
  #va-text::placeholder{color:var(--ash)}
  #va-mic{position:relative;width:56px;height:56px;padding:0;border-radius:75px;font-size:0;flex:none}
  #va-mic::before{content:"";position:absolute;left:50%;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:75px;background:#000;transition:transform .8s var(--ease),border-radius .8s var(--ease),background .8s var(--ease)}
  #va-mic:hover{letter-spacing:0}
  #va-mic:hover::before{transform:scale(1.3)}
  #va-mic.rec{background:#000}
  #va-mic.rec::before{background:#fff;border-radius:0;transform:scale(.9)}
  #va-mic.rec::after{content:"";position:absolute;inset:-1px;border-radius:75px;border:1px solid #000;animation:varing 1.6s var(--ease) infinite}
  @keyframes varing{from{transform:scale(1);opacity:1}to{transform:scale(1.7);opacity:0}}
  #va-level{display:none}
  .va-kbd{display:inline-block;padding:0 8px;border-radius:75px;border:1px solid var(--ash);font-size:10px;color:#000}
  #va-hint{font-size:11px;color:var(--felt);padding:0 40px 14px;text-transform:uppercase;letter-spacing:.4px}
  .va-cursor{display:inline-block;width:9px;height:18px;background:#000;vertical-align:-3px;margin-left:3px;animation:vablink 1s steps(2) infinite}
  @keyframes vablink{50%{opacity:0}}
  .va-dots span{display:inline-block;width:6px;height:6px;margin-right:6px;border-radius:75px;background:#000;animation:vadot 1.25s var(--ease) infinite}
  .va-dots span:nth-child(2){animation-delay:.15s}.va-dots span:nth-child(3){animation-delay:.3s}
  @keyframes vadot{0%,80%,100%{opacity:.2}40%{opacity:1}}

  @media (max-width:1100px){ #va-state{font-size:54px;letter-spacing:-1.2px} #va-head .mid{display:none} #va-head{grid-template-columns:1fr auto} }
  @media (max-width:980px){
    #va-toggle{display:inline}
    #va-side{position:absolute;z-index:5;left:0;top:66px;bottom:0;background:#fff;transform:translateX(-101%);transition:transform 1.25s var(--ease)}
    #va-root.side-open #va-side{transform:none}
    #va-badge{display:none}
  }
  @media (max-width:640px){ #va-hero{height:240px;padding:20px} #va-state{font-size:39px} #va-pipe{display:none} #va-chat,#va-input{padding-left:20px;padding-right:20px} }
  `;

  const WELCOME = `
    <div id="va-welcome">
      <h3>Talk to an AI that lives in your browser.</h3>
      <p>Two neural networks are downloaded to this page and run on your own GPU. <b>Whisper</b> turns speech into text;
         a <b>small language model</b> writes the reply. No AI server. No API calls. It even works offline.</p>
      <div class="va-chips">
        <button class="va-chip" data-q="Explain what a neural network is, simply.">What is a neural network?</button>
        <button class="va-chip" data-q="Why is ReLU better than sigmoid for deep networks?">ReLU vs sigmoid</button>
        <button class="va-chip" data-q="Give me three tips for revising for an exam.">Revision tips</button>
        <button class="va-chip" data-q="Tell me a short joke about computers.">Tell me a joke</button>
      </div>
      <div class="va-feats"><div>Private — audio never leaves your device</div><div>WebGPU accelerated</div><div>Works offline</div></div>
    </div>`;

  function buildUI() {
    if ($id("va-root")) $id("va-root").remove();

    if (!document.getElementById("va-font")) {   // Inter, as a substitute for Roobert (falls back to system font offline)
      const lk = document.createElement("link"); lk.id = "va-font"; lk.rel = "stylesheet";
      lk.href = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap";
      document.head.appendChild(lk);
    }
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.id = "va-root";
    root.innerHTML = `
      <div id="va-head">
        <div class="mark"><button id="va-toggle" class="va-link" title="Settings">Menu</button>offline voice assistant</div>
        <div class="mid">
          <span id="va-backend" class="va-badge">checking GPU…</span>
          <span id="va-online" class="va-badge">online</span>
        </div>
        <div class="right"><span>Whisper</span><span>+</span><span>Small language model</span></div>
      </div>
      <div id="va-body">
        <div id="va-side">
          <div class="va-card">
            <h2><b>01</b> Models</h2>
            <label for="va-llm">Language model</label>
            <select id="va-llm"></select>
            <label for="va-asr">Speech-to-text</label>
            <select id="va-asr"></select>
            <label for="va-dev">Run on</label>
            <select id="va-dev">
              <option value="webgpu">GPU — WebGPU</option>
              <option value="wasm">CPU — WebAssembly</option>
            </select>
            <button id="va-load" class="va-btn solid">Load models</button>
            <div id="va-prog"></div>
          </div>

          <div class="va-card">
            <h2><b>02</b> Personality</h2>
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
            <label for="va-file">Or ask with an audio file</label>
            <input id="va-file" type="file" accept="audio/*" disabled>
          </div>

          <div class="va-card">
            <h2><b>04</b> Live performance</h2>
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
            <canvas id="va-spark" width="528" height="104" title="Tokens per second for each reply"></canvas>
            <div class="va-cap">Tokens per second, per reply</div>
            <div style="display:flex;gap:8px;margin-top:28px">
              <button id="va-csv" class="va-btn sec" style="flex:1">Copy CSV</button>
              <button id="va-clear" class="va-btn sec" style="flex:1">Clear chat</button>
            </div>
          </div>
          <div class="va-card" style="border:0">
            <p class="va-small" style="margin:0">Once loaded, switch off Wi-Fi. It keeps working — the neural networks run on your machine.</p>
          </div>
        </div>

        <div id="va-main">
          <div id="va-hero">
            <canvas id="va-liquid"></canvas>
            <svg id="va-badge" viewBox="0 0 100 100" aria-hidden="true">
              <defs><path id="va-circ" d="M50,50 m-40,0 a40,40 0 1,1 80,0 a40,40 0 1,1 -80,0"/></defs>
              <text><textPath href="#va-circ">on-device ai · runs offline · on-device ai · runs offline ·</textPath></text>
            </svg>
            <div id="va-state">Ready to load<small>Pick your models on the left and press Load models.</small></div>
            <div id="va-pipe">
              <div class="pp" id="pp-mic"><div class="t">Microphone<span id="pv-mic">16 kHz</span></div></div>
              <div class="pp-arrow" id="pa-1"></div>
              <div class="pp" id="pp-asr"><div class="t">Whisper<span id="pv-asr">speech → text</span></div></div>
              <div class="pp-arrow" id="pa-2"></div>
              <div class="pp" id="pp-llm"><div class="t">Language model<span id="pv-llm">text → reply</span></div></div>
              <div class="pp-arrow" id="pa-3"></div>
              <div class="pp" id="pp-tts"><div class="t">Voice<span id="pv-tts">reply → speech</span></div></div>
            </div>
          </div>
          <div id="va-chat"></div>
          <div id="va-input">
            <div id="va-inbox">
              <textarea id="va-text" rows="1" placeholder="Load the models, then type a message or press the mic…" disabled></textarea>
              <button id="va-send" class="va-btn" disabled>Send</button>
              <button id="va-stop" class="va-btn red" style="display:none">Stop</button>
            </div>
            <button id="va-mic" class="va-btn" title="Click to talk (or hold Space)" disabled>Mic</button>
            <div id="va-level"><div></div></div>
          </div>
          <div id="va-hint">Hold <span class="va-kbd">Space</span> to talk &nbsp;·&nbsp; <span class="va-kbd">Enter</span> to send &nbsp;·&nbsp; <span class="va-kbd">Shift + Enter</span> new line</div>
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
    $id("va-clear").onclick = () => { S.history = []; $id("va-chat").innerHTML = WELCOME; };
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

    $id("va-chat").innerHTML = WELCOME;
    $id("va-chat").addEventListener("click", (e) => {
      const chip = e.target.closest(".va-chip");
      if (!chip) return;
      if (!S.llmReady) { sysMsg("Load the models first (left panel) - then try that again."); return; }
      sendText(chip.dataset.q);
    });
    $id("va-toggle").onclick = () => root.classList.toggle("side-open");

    startOrb();
  }

  // ---------------------------------------------------------------------------------
  // Chat display helpers
  // ---------------------------------------------------------------------------------

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function mini_md(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  }
  function addMsg(role, html) {
    const w = $id("va-welcome"); if (w && role !== "system") w.remove();
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
    $id("va-load").disabled = true;
    $id("va-load").textContent = "Loading…";
    $id("va-prog").innerHTML =
      `<div class="va-prog-row"><div class="top"><span>Language model</span><span id="pl-llm">starting…</span></div><div class="va-bar"><div id="pb-llm"></div></div></div>
       <div class="va-prog-row"><div class="top"><span>Whisper</span><span id="pl-asr">waiting…</span></div><div class="va-bar"><div id="pb-asr"></div></div></div>`;
    sysMsg(`Loading <b>${esc(llm.name)}</b> and <b>${esc(asr.name)}</b> on <b>${S.device}</b>…`);
    loadPending = 2;
    S.loading = true;
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
          S.loading = false;
          $id("va-load").disabled = false; $id("va-load").textContent = "Reload models";
          setInputs(true);
          $id("va-text").placeholder = "Ask me anything…  (or press the mic and talk)";
          sysMsg(`Both models loaded (LLM ${fmtMs(S.llmLoadMs)}, Whisper ${fmtMs(S.asrLoadMs)}). Type, press the mic, or hold <b>Space</b> to talk.`);
          $id("va-text").focus();
        }
        break;

      case "transcript":
        $id("st-asr").textContent = fmtMs(m.ms) + " for " + m.seconds.toFixed(1) + " s";
        $id("st-rtf").textContent = (m.ms / 1000 / m.seconds).toFixed(2) + "×";
        S.lastAsr = { ms: m.ms, seconds: m.seconds };
        S.transcribing = false;
        $id("pv-asr").textContent = fmtMs(m.ms);
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
        sysMsg("Error during <b>" + esc(m.where) + "</b>: " + esc(m.message));
        S.loading = false; S.transcribing = false;
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

    const meta = fromVoice && S.lastAsr ? `<span class="meta">Heard by Whisper in ${fmtMs(S.lastAsr.ms)}</span>` : "";
    addMsg("user", esc(text) + meta);
    S.history.push({ role: "user", content: text });

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

    S.tpsHist.push(m.tps);
    $id("pv-llm").textContent = m.tps.toFixed(0) + " tok/s";
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
      sysMsg("Microphone not available (" + esc(err.name || String(err)) + "). Check the browser's mic permission.");
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
    if (discard) sysMsg("No speech heard - stopped listening.");
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
      if (audio.length < 16000 * 0.3) { sysMsg("Recording too short."); setInputs(true); return; }
      if (audio.length > 16000 * 30) { audio = audio.slice(0, 16000 * 30); sysMsg("Audio longer than 30 s - using the first 30 s."); }
      S.transcribing = true;
      S.pendingVoice = sysMsg("Transcribing with Whisper…");
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
  // The hero: a liquid, iridescent WebGL shader (sage green -> amber -> oxblood) that
  // reacts to what the AI is doing. Like the "canvas webgl" example World on Ancient Brain, the JS builds
  // a GLSL program, compiles it, and the GPU runs it for every pixel.
  // Technique: fractal Brownian motion + domain warping, after Inigo Quilez
  //   https://iquilezles.org/articles/warp/   https://iquilezles.org/articles/fbm/
  // ---------------------------------------------------------------------------------

  const STATES = {
    off:          { label: "Ready to load", sub: "Pick your models on the left and press Load models.", stage: null },
    loading:      { label: "Downloading", sub: "Fetching model weights into your browser. Cached for next time.", stage: null },
    idle:         { label: "Listening for you", sub: "Type below, press the mic, or hold Space and talk.", stage: null },
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

  const FRAG = `
    precision highp float;
    uniform vec2 r; uniform float t; uniform float e; uniform float p;
    float h(vec2 q){ return fract(sin(dot(q, vec2(127.1, 311.7))) * 43758.5453); }
    float n(vec2 q){ vec2 i = floor(q), f = fract(q); f = f*f*(3.0-2.0*f);
      return mix(mix(h(i), h(i+vec2(1,0)), f.x), mix(h(i+vec2(0,1)), h(i+vec2(1,1)), f.x), f.y); }
    float fbm(vec2 q){ float v = 0.0, a = 0.5;
      for (int i = 0; i < 4; i++) { v += a * n(q); q = q * 1.9 + vec2(1.7, 9.2); a *= 0.45; } return v; }
    void main(){
      vec2 uv = gl_FragCoord.xy / r;
      vec2 q0 = uv * vec2(r.x / r.y, 1.0) * 0.75;
      float T = t;
      vec2 a = vec2(fbm(q0 + vec2(0.0, T*0.6)), fbm(q0 + vec2(5.2, 1.3) - T*0.5));
      vec2 b = vec2(fbm(q0 + 3.0*a + vec2(1.7, 9.2) + T*0.8), fbm(q0 + 3.0*a + vec2(8.3, 2.8) - T*0.7));
      float f = smoothstep(0.1, 0.9, fbm(q0 + (2.2 + 2.0*e) * b));
      vec3 sage  = vec3(160.0, 224.0, 171.0) / 255.0;
      vec3 amber = vec3(255.0, 172.0,  46.0) / 255.0;
      vec3 ox    = vec3(165.0,  45.0,  37.0) / 255.0;
      vec3 c = mix(sage, amber, smoothstep(0.25, 0.55, f));
      c = mix(c, ox, smoothstep(0.50, 0.80, f + 0.15*b.x));
      c = mix(c, vec3(0.02, 0.01, 0.01), smoothstep(0.70, 1.05, f + 0.2*a.y));   // deep pools of black
      float sheen = pow(1.0 - abs(fract(f*2.0 + T*0.15) - 0.5) * 2.0, 14.0);       // thin oily highlights
      c += sheen * (0.10 + 0.35*e) * vec3(1.0, 0.95, 0.9);
      c *= mix(0.30, 1.0, smoothstep(0.0, 1.0, uv.y*0.9 + uv.x*0.35));            // darker lower-left: white text reads
      if (p > 0.0) { c *= mix(0.45, 1.0, step(uv.x, p)); }                          // loading: light fills left to right
      gl_FragColor = vec4(c, 1.0);
    }`;

  function startOrb() {
    const cv = $id("va-liquid");
    let gl = null;
    try { gl = cv.getContext("webgl", { antialias: false, premultipliedAlpha: false }); } catch (e) {}
    let prog = null, U = {};
    if (gl) {
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      try {
        prog = gl.createProgram();
        gl.attachShader(prog, sh(gl.VERTEX_SHADER, "attribute vec2 v; void main(){ gl_Position = vec4(v, 0.0, 1.0); }"));
        gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
        gl.linkProgram(prog); gl.useProgram(prog);
        const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, "v"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        ["r", "t", "e", "p"].forEach((k) => (U[k] = gl.getUniformLocation(prog, k)));
      } catch (err) { console.warn("WebGL hero disabled:", err); gl = null; }
    }
    if (!gl) $id("va-hero").style.background = "linear-gradient(90deg, rgb(160,224,171), rgb(255,172,46) 50%, rgb(165,45,37))";

    let shown = null, energy = 0, T = Math.random() * 100, last = performance.now();
    const stages = ["mic", "asr", "llm", "tts"];

    function frame(now) {
      if (!document.body.contains(cv)) return;           // UI was rebuilt
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      const st = currentState(), def = STATES[st];

      if (st !== shown) {
        shown = st;
        $id("va-state").innerHTML = def.label + "<small>" + def.sub + "</small>";
        stages.forEach((k, i) => {
          $id("pp-" + k).classList.toggle("on", def.stage === k);
          if (i) $id("pa-" + i).classList.toggle("on", stages.indexOf(def.stage) >= i);
        });
      }

      // how "excited" the liquid is in each state
      let target = 0.05;
      if (st === "listening") target = 0.2 + Math.min(1, S.level * 14);
      else if (st === "thinking" || st === "transcribing") target = 0.5;
      else if (st === "writing") target = 0.65;
      else if (st === "speaking") target = 0.45 + 0.3 * Math.abs(Math.sin(now / 160));
      else if (st === "loading") target = 0.3;
      energy += (target - energy) * Math.min(1, dt * 3);    // patient easing
      T += dt * (0.05 + energy * 0.35);                      // flow speed follows energy

      if (gl) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2) * 0.5;   // half resolution: cheap, and looks softer
        const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
        if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; gl.viewport(0, 0, w, h); }
        gl.uniform2f(U.r, w, h); gl.uniform1f(U.t, T); gl.uniform1f(U.e, energy);
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
      grad.addColorStop(0, "#000"); grad.addColorStop(1, "#000");
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
