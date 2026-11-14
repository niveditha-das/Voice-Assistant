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

  const CSS = `
  #va-root{--bg:#07080f;--panel:rgba(20,22,40,.55);--line:rgba(255,255,255,.08);--text:#eef0ff;--muted:#9096bf;--dim:#62678f;
    --a1:#7c5cff;--a2:#22d3ee;--a3:#f472b6;--ok:#34d399;--warn:#fbbf24;--bad:#fb7185;
    position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;overflow:hidden;
    font-family:"Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);font-size:14.5px}
  #va-root *{box-sizing:border-box}
  #va-root ::-webkit-scrollbar{width:8px}#va-root ::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:8px}

  /* animated aurora background */
  #va-aurora{position:absolute;inset:-20%;z-index:0;pointer-events:none;filter:blur(70px);opacity:.55}
  #va-aurora i{position:absolute;border-radius:50%;mix-blend-mode:screen}
  #va-aurora i:nth-child(1){width:48vw;height:48vw;left:5%;top:0;background:radial-gradient(circle,#5b3bff 0,transparent 65%);animation:vaf1 22s ease-in-out infinite alternate}
  #va-aurora i:nth-child(2){width:42vw;height:42vw;right:0;top:20%;background:radial-gradient(circle,#0ea5e9 0,transparent 65%);animation:vaf2 26s ease-in-out infinite alternate}
  #va-aurora i:nth-child(3){width:38vw;height:38vw;left:35%;bottom:-5%;background:radial-gradient(circle,#db2777 0,transparent 65%);animation:vaf3 30s ease-in-out infinite alternate;opacity:.6}
  @keyframes vaf1{to{transform:translate(18vw,12vh) scale(1.15)}}
  @keyframes vaf2{to{transform:translate(-20vw,-8vh) scale(.9)}}
  @keyframes vaf3{to{transform:translate(-12vw,-14vh) scale(1.2)}}
  #va-grain{position:absolute;inset:0;z-index:0;pointer-events:none;opacity:.5;
    background-image:radial-gradient(rgba(255,255,255,.035) 1px,transparent 1px);background-size:3px 3px}

  .va-glass{background:var(--panel);border:1px solid var(--line);backdrop-filter:blur(18px) saturate(140%);-webkit-backdrop-filter:blur(18px) saturate(140%)}

  /* header */
  #va-head{position:relative;z-index:2;display:flex;align-items:center;gap:14px;padding:12px 20px;border-width:0 0 1px 0}
  #va-logo{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;font-size:17px;
    background:conic-gradient(from 200deg,var(--a1),var(--a2),var(--a3),var(--a1));box-shadow:0 0 22px rgba(124,92,255,.55)}
  #va-head h1{font-size:16.5px;margin:0;font-weight:700;letter-spacing:-.2px}
  #va-head h1 span{background:linear-gradient(90deg,#c4b5fd,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
  #va-head .sub{font-size:12px;color:var(--muted)}
  .va-badge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:600;padding:5px 10px;border-radius:999px;
    background:rgba(255,255,255,.06);border:1px solid var(--line);color:#c8cbe8;white-space:nowrap}
  .va-badge::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--dim)}
  .va-badge.ok{color:#a7f3d0;border-color:rgba(52,211,153,.3)}.va-badge.ok::before{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  .va-badge.warn{color:#fde68a;border-color:rgba(251,191,36,.3)}.va-badge.warn::before{background:var(--warn)}
  .va-badge.bad{color:#fecdd3}.va-badge.bad::before{background:var(--bad)}
  #va-toggle{display:none}

  /* layout */
  #va-body{position:relative;z-index:1;flex:1;display:flex;min-height:0;gap:14px;padding:14px}
  #va-side{width:310px;flex:none;overflow-y:auto;display:flex;flex-direction:column;gap:12px;padding-right:2px}
  .va-card{border-radius:18px;padding:14px 15px}
  .va-card h2{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted);margin:0 0 10px}
  .va-card h2 b{display:grid;place-items:center;width:20px;height:20px;border-radius:7px;font-size:11px;color:#fff;
    background:linear-gradient(135deg,var(--a1),var(--a2))}
  #va-side label{display:block;font-size:12.5px;color:#b9bde0;margin:10px 0 5px}
  #va-side label:first-of-type{margin-top:0}
  #va-side select{width:100%;appearance:none;-webkit-appearance:none;background:rgba(0,0,0,.28) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%239096bf' stroke-width='1.6' fill='none'/%3E%3C/svg%3E") no-repeat right 11px center;
    color:var(--text);border:1px solid var(--line);border-radius:11px;padding:9px 30px 9px 11px;font:inherit;font-size:13px;cursor:pointer}
  #va-side select:focus{outline:none;border-color:rgba(124,92,255,.6);box-shadow:0 0 0 3px rgba(124,92,255,.18)}
  #va-side input[type=range]{width:100%;-webkit-appearance:none;appearance:none;height:5px;border-radius:5px;background:rgba(255,255,255,.1);outline:none}
  #va-side input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#fff;box-shadow:0 0 0 4px rgba(124,92,255,.35);cursor:pointer}
  #va-side .va-lbl{display:flex;justify-content:space-between}.va-lbl b{color:#fff;font-variant-numeric:tabular-nums}

  #va-side .va-switch{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:9px 0;font-size:13px;color:#d4d7f0;cursor:pointer}
  #va-side .va-switch input{display:none}
  #va-side .va-switch i{flex:none;position:relative;width:36px;height:20px;border-radius:20px;background:rgba(255,255,255,.12);transition:.25s}
  #va-side .va-switch i::after{content:"";position:absolute;left:3px;top:3px;width:14px;height:14px;border-radius:50%;background:#fff;transition:.25s}
  #va-side .va-switch input:checked + i{background:linear-gradient(90deg,var(--a1),var(--a2))}
  #va-side .va-switch input:checked + i::after{left:19px}

  .va-btn{border:0;border-radius:12px;padding:10px 15px;font:inherit;font-weight:650;font-size:13.5px;cursor:pointer;color:#fff;
    background:linear-gradient(135deg,var(--a1),#5b8cff);box-shadow:0 6px 20px -6px rgba(124,92,255,.7);transition:transform .15s,filter .15s,box-shadow .15s}
  .va-btn:hover{filter:brightness(1.12);transform:translateY(-1px)}
  .va-btn:active{transform:translateY(0)}
  .va-btn:disabled{opacity:.4;cursor:not-allowed;filter:none;transform:none;box-shadow:none}
  .va-btn.sec{background:rgba(255,255,255,.07);border:1px solid var(--line);box-shadow:none;font-weight:600}
  .va-btn.red{background:linear-gradient(135deg,#f43f5e,#fb7185);box-shadow:0 6px 20px -6px rgba(244,63,94,.7)}
  #va-load{width:100%;margin-top:14px;padding:12px}

  .va-prog-row{margin-top:12px}
  .va-prog-row .top{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
  .va-prog-row .top span:last-child{color:#dfe2ff;font-variant-numeric:tabular-nums}
  .va-bar{height:6px;background:rgba(255,255,255,.08);border-radius:6px;overflow:hidden;margin-top:6px}
  .va-bar>div{height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,var(--a1),var(--a2));box-shadow:0 0 12px var(--a2);transition:width .25s}
  .va-small{font-size:12px;color:var(--muted);line-height:1.5}
  #va-file{font-size:12px;color:var(--muted);width:100%}
  #va-file::file-selector-button{font:inherit;font-size:12px;color:#fff;background:rgba(255,255,255,.08);border:1px solid var(--line);border-radius:9px;padding:6px 10px;margin-right:8px;cursor:pointer}

  .va-stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .va-stat{background:rgba(0,0,0,.22);border:1px solid var(--line);border-radius:12px;padding:9px 10px}
  .va-stat .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim)}
  .va-stat .v{font-size:15px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #va-spark{width:100%;height:46px;display:block;margin-top:10px}

  /* main column */
  #va-main{flex:1;display:flex;flex-direction:column;min-width:0;border-radius:22px;overflow:hidden}
  #va-hero{display:flex;align-items:center;gap:22px;padding:16px 24px 14px;border-bottom:1px solid var(--line)}
  #va-orbwrap{position:relative;width:132px;height:132px;flex:none}
  #va-orb{width:132px;height:132px;display:block}
  #va-herotext{flex:1;min-width:0}
  #va-state{font-size:22px;font-weight:750;letter-spacing:-.4px}
  #va-state small{display:block;font-size:13px;font-weight:500;color:var(--muted);letter-spacing:0;margin-top:3px}

  #va-pipe{display:flex;align-items:center;gap:6px;margin-top:14px;flex-wrap:wrap}
  .pp{display:flex;align-items:center;gap:8px;padding:7px 11px 7px 8px;border-radius:12px;background:rgba(0,0,0,.25);border:1px solid var(--line);transition:.3s}
  .pp .ic{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:14px;background:rgba(255,255,255,.06)}
  .pp .t{font-size:12px;font-weight:650;line-height:1.15}
  .pp .t span{display:block;font-size:10.5px;font-weight:500;color:var(--dim);font-variant-numeric:tabular-nums}
  .pp.on{border-color:rgba(124,92,255,.7);background:rgba(124,92,255,.16);box-shadow:0 0 22px -4px rgba(124,92,255,.8)}
  .pp.on .ic{background:linear-gradient(135deg,var(--a1),var(--a2));animation:vapp 1s ease-in-out infinite}
  @keyframes vapp{50%{transform:scale(1.12)}}
  .pp-arrow{color:var(--dim);font-size:12px}
  .pp-arrow.on{color:var(--a2);text-shadow:0 0 8px var(--a2)}

  #va-chat{flex:1;overflow-y:auto;padding:22px 26px 10px;display:flex;flex-direction:column;gap:14px;scroll-behavior:smooth}
  .va-msg{position:relative;max-width:min(720px,86%);padding:12px 16px;border-radius:18px;line-height:1.55;white-space:pre-wrap;word-wrap:break-word;animation:vain .35s cubic-bezier(.2,.8,.2,1)}
  @keyframes vain{from{opacity:0;transform:translateY(8px) scale(.98)}}
  .va-msg.user{align-self:flex-end;background:linear-gradient(135deg,#6d4dff,#3b82f6);color:#fff;border-bottom-right-radius:6px;box-shadow:0 10px 30px -12px rgba(109,77,255,.8)}
  .va-msg.assistant{align-self:flex-start;margin-left:40px;background:rgba(255,255,255,.055);border:1px solid var(--line);border-bottom-left-radius:6px}
  .va-msg.assistant::before{content:"";position:absolute;left:-40px;bottom:0;width:30px;height:30px;border-radius:50%;
    background:radial-gradient(circle at 35% 30%,#fff 0,#a78bfa 18%,#7c3aed 45%,#0e7490 100%);box-shadow:0 0 16px rgba(124,92,255,.7)}
  .va-msg.system{align-self:center;background:transparent;color:var(--muted);font-size:12.5px;text-align:center;max-width:600px;padding:4px 10px}
  .va-msg .meta{display:block;font-size:11px;color:var(--dim);margin-top:7px;font-variant-numeric:tabular-nums}
  .va-msg.user .meta{color:rgba(255,255,255,.7)}
  .va-msg code{background:rgba(0,0,0,.35);padding:1px 6px;border-radius:5px;font-size:.92em}

  #va-welcome{margin:auto;text-align:center;max-width:560px;padding:10px 0 20px;animation:vain .5s}
  #va-welcome h3{font-size:26px;font-weight:800;letter-spacing:-.6px;margin:0 0 8px;
    background:linear-gradient(90deg,#fff,#c4b5fd 40%,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
  #va-welcome p{color:var(--muted);line-height:1.6;margin:0 0 16px}
  .va-chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
  .va-chip{font:inherit;font-size:13px;color:#dfe2ff;padding:8px 13px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--line);cursor:pointer;transition:.2s}
  .va-chip:hover{background:rgba(124,92,255,.2);border-color:rgba(124,92,255,.6);transform:translateY(-1px)}
  .va-feats{display:flex;gap:10px;justify-content:center;margin-top:18px;flex-wrap:wrap}
  .va-feats div{font-size:11.5px;color:var(--muted);padding:6px 10px;border-radius:10px;background:rgba(0,0,0,.2);border:1px solid var(--line)}

  #va-input{display:flex;gap:10px;padding:12px 16px 16px;align-items:flex-end}
  #va-inbox{flex:1;display:flex;align-items:flex-end;gap:8px;padding:6px 6px 6px 14px;border-radius:18px;background:rgba(0,0,0,.32);border:1px solid var(--line);transition:.2s}
  #va-inbox:focus-within{border-color:rgba(124,92,255,.6);box-shadow:0 0 0 4px rgba(124,92,255,.15)}
  #va-text{flex:1;resize:none;min-height:40px;max-height:140px;background:transparent;color:var(--text);border:0;outline:none;padding:10px 0;font:inherit;line-height:1.4}
  #va-text::placeholder{color:var(--dim)}
  #va-mic{width:52px;height:52px;border-radius:50%;font-size:20px;padding:0;flex:none;background:linear-gradient(135deg,var(--a1),var(--a2))}
  #va-mic.rec{background:linear-gradient(135deg,#f43f5e,#f472b6);animation:vapulse 1.2s infinite}
  @keyframes vapulse{0%{box-shadow:0 0 0 0 rgba(244,63,94,.6)}70%{box-shadow:0 0 0 16px rgba(244,63,94,0)}100%{box-shadow:0 0 0 0 rgba(244,63,94,0)}}
  #va-send,#va-stop{height:40px;border-radius:13px}
  #va-level{display:none}
  .va-kbd{display:inline-block;padding:1px 6px;border-radius:5px;border:1px solid var(--line);background:rgba(255,255,255,.06);font-size:11px;color:#dfe2ff}
  #va-hint{font-size:11px;color:var(--dim);text-align:center;padding:0 0 10px;margin-top:-8px}
  .va-cursor{display:inline-block;width:8px;height:16px;border-radius:2px;background:linear-gradient(var(--a1),var(--a2));vertical-align:-3px;margin-left:2px;animation:vablink 1s steps(2) infinite}
  @keyframes vablink{50%{opacity:0}}
  .va-dots span{display:inline-block;width:7px;height:7px;margin:0 2px;border-radius:50%;background:#a78bfa;animation:vadot 1.2s infinite}
  .va-dots span:nth-child(2){animation-delay:.15s}.va-dots span:nth-child(3){animation-delay:.3s}
  @keyframes vadot{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}

  @media (max-width:980px){
    #va-toggle{display:inline-flex}
    #va-side{position:absolute;z-index:5;left:14px;top:14px;bottom:14px;transform:translateX(-120%);transition:transform .3s}
    #va-root.side-open #va-side{transform:none}
    #va-head .sub{display:none}
  }
  @media (max-width:640px){ #va-orbwrap,#va-orb{width:92px;height:92px} #va-pipe{display:none} #va-state{font-size:18px} }
  `;

  const WELCOME = `
    <div id="va-welcome">
      <h3>Talk to an AI that lives in your browser</h3>
      <p>Two neural networks are downloaded to this page and run on your own GPU: <b>Whisper</b> (OpenAI) turns speech into text,
         and a <b>small language model</b> writes the reply. No AI server, no API calls - it even works offline.</p>
      <div class="va-chips">
        <button class="va-chip" data-q="Explain what a neural network is, simply.">🧠 What is a neural network?</button>
        <button class="va-chip" data-q="Why is ReLU better than sigmoid for deep networks?">📈 ReLU vs sigmoid</button>
        <button class="va-chip" data-q="Give me three tips for revising for an exam.">📚 Revision tips</button>
        <button class="va-chip" data-q="Tell me a short joke about computers.">😄 Tell me a joke</button>
      </div>
      <div class="va-feats"><div>🔒 Private: audio never leaves your device</div><div>⚡ WebGPU accelerated</div><div>✈️ Works offline</div></div>
    </div>`;

  function buildUI() {
    if ($id("va-root")) $id("va-root").remove();

    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.id = "va-root";
    root.innerHTML = `
      <div id="va-aurora"><i></i><i></i><i></i></div><div id="va-grain"></div>
      <div id="va-head" class="va-glass">
        <button id="va-toggle" class="va-btn sec" title="Settings">☰</button>
        <div id="va-logo">✦</div>
        <div>
          <h1>Offline <span>Voice Assistant</span></h1>
          <div class="sub">Whisper speech recognition + a small language model, running entirely in your browser</div>
        </div>
        <span style="flex:1"></span>
        <span id="va-backend" class="va-badge">checking GPU…</span>
        <span id="va-online" class="va-badge">online</span>
      </div>
      <div id="va-body">
        <div id="va-side">
          <div class="va-card va-glass">
            <h2><b>1</b> Models</h2>
            <label for="va-llm">Language model (the "brain")</label>
            <select id="va-llm"></select>
            <label for="va-asr">Speech-to-text (the "ears")</label>
            <select id="va-asr"></select>
            <label for="va-dev">Run on</label>
            <select id="va-dev">
              <option value="webgpu">⚡ GPU (WebGPU)</option>
              <option value="wasm">🐢 CPU (WebAssembly)</option>
            </select>
            <button id="va-load" class="va-btn">Load models</button>
            <div id="va-prog"></div>
          </div>

          <div class="va-card va-glass">
            <h2><b>2</b> Personality</h2>
            <label for="va-persona">Persona</label>
            <select id="va-persona">
              <option value="assistant">🤖 Helpful assistant</option>
              <option value="tutor">🎓 AI module revision tutor</option>
              <option value="pirate">🏴‍☠️ Pirate captain</option>
            </select>
            <label for="va-temp" class="va-lbl"><span>Creativity (temperature)</span><b id="va-tempv">0.3</b></label>
            <input id="va-temp" type="range" min="0" max="1.2" step="0.1" value="0.3">
            <label for="va-max" class="va-lbl"><span>Max reply length (tokens)</span><b id="va-maxv">200</b></label>
            <input id="va-max" type="range" min="32" max="512" step="16" value="200">
          </div>

          <div class="va-card va-glass">
            <h2><b>3</b> Voice</h2>
            <label class="va-switch"><span>Read replies aloud</span><input type="checkbox" id="va-speak" checked><i></i></label>
            <label class="va-switch"><span>Auto-stop when I stop talking</span><input type="checkbox" id="va-auto" checked><i></i></label>
            <label class="va-switch"><span>Hands-free conversation</span><input type="checkbox" id="va-conv"><i></i></label>
            <label for="va-voice">Speaking voice</label>
            <select id="va-voice"></select>
            <label for="va-file">Or ask with an audio file</label>
            <input id="va-file" type="file" accept="audio/*" disabled>
          </div>

          <div class="va-card va-glass">
            <h2><b>4</b> Live performance</h2>
            <div class="va-stats">
              <div class="va-stat"><div class="k">Backend</div><div class="v" id="st-dev">–</div></div>
              <div class="va-stat"><div class="k">Speed</div><div class="v" id="st-tps">–</div></div>
              <div class="va-stat"><div class="k">First token</div><div class="v" id="st-ttft">–</div></div>
              <div class="va-stat"><div class="k">Reply tokens</div><div class="v" id="st-tok">–</div></div>
              <div class="va-stat"><div class="k">LLM load</div><div class="v" id="st-llmload">–</div></div>
              <div class="va-stat"><div class="k">Whisper load</div><div class="v" id="st-asrload">–</div></div>
              <div class="va-stat"><div class="k">Transcription</div><div class="v" id="st-asr">–</div></div>
              <div class="va-stat"><div class="k">Real-time factor</div><div class="v" id="st-rtf">–</div></div>
            </div>
            <canvas id="va-spark" width="560" height="92" title="Tokens per second for each reply"></canvas>
            <div class="va-small" style="text-align:center">tokens / second per reply</div>
            <div style="display:flex;gap:8px;margin-top:12px">
              <button id="va-csv" class="va-btn sec" style="flex:1">⬇ Copy CSV</button>
              <button id="va-clear" class="va-btn sec" style="flex:1">✕ Clear chat</button>
            </div>
          </div>
          <p class="va-small" style="padding:0 6px">💡 Once loaded, switch off Wi-Fi: it keeps working, because the neural networks run on <b>your</b> machine.</p>
        </div>

        <div id="va-main" class="va-glass">
          <div id="va-hero">
            <div id="va-orbwrap"><canvas id="va-orb" width="264" height="264"></canvas></div>
            <div id="va-herotext">
              <div id="va-state">Ready to load<small>Pick your models on the left and press Load models.</small></div>
              <div id="va-pipe">
                <div class="pp" id="pp-mic"><div class="ic">🎤</div><div class="t">Microphone<span id="pv-mic">16 kHz audio</span></div></div>
                <div class="pp-arrow" id="pa-1">➜</div>
                <div class="pp" id="pp-asr"><div class="ic">👂</div><div class="t">Whisper<span id="pv-asr">speech → text</span></div></div>
                <div class="pp-arrow" id="pa-2">➜</div>
                <div class="pp" id="pp-llm"><div class="ic">🧠</div><div class="t">Language model<span id="pv-llm">text → reply</span></div></div>
                <div class="pp-arrow" id="pa-3">➜</div>
                <div class="pp" id="pp-tts"><div class="ic">🔊</div><div class="t">Voice<span id="pv-tts">reply → speech</span></div></div>
              </div>
            </div>
          </div>
          <div id="va-chat"></div>
          <div id="va-input">
            <div id="va-inbox">
              <textarea id="va-text" rows="1" placeholder="Load the models, then type a message or press the mic…" disabled></textarea>
              <button id="va-send" class="va-btn" disabled>Send ➤</button>
              <button id="va-stop" class="va-btn red" style="display:none">■ Stop</button>
            </div>
            <button id="va-mic" class="va-btn" title="Click to talk (or hold Space)" disabled>🎤</button>
            <div id="va-level"><div></div></div>
          </div>
          <div id="va-hint">Hold <span class="va-kbd">Space</span> to talk · <span class="va-kbd">Enter</span> to send · <span class="va-kbd">Shift+Enter</span> new line</div>
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
      `<div class="va-prog-row"><div class="top"><span>🧠 Language model</span><span id="pl-llm">starting…</span></div><div class="va-bar"><div id="pb-llm"></div></div></div>
       <div class="va-prog-row"><div class="top"><span>👂 Whisper</span><span id="pl-asr">waiting…</span></div><div class="va-bar"><div id="pb-asr"></div></div></div>`;
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
          $id("va-text").placeholder = "Ask me anything…  (or press 🎤 and talk)";
          sysMsg(`✅ Both models loaded (LLM ${fmtMs(S.llmLoadMs)}, Whisper ${fmtMs(S.asrLoadMs)}). Type, click 🎤, or hold <b>Space</b> to talk.`);
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
        sysMsg("⚠️ Error during <b>" + esc(m.where) + "</b>: " + esc(m.message));
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

    const meta = fromVoice && S.lastAsr ? `<span class="meta">🎤 heard by Whisper in ${fmtMs(S.lastAsr.ms)}</span>` : "";
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
      S.transcribing = true;
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
  // The animated "orb" + pipeline highlight. One animation loop reads the app state
  // and shows what the AI is doing right now (idle / listening / thinking / ...).
  // ---------------------------------------------------------------------------------

  const STATES = {
    off:          { label: "Ready to load", sub: "Pick your models on the left and press Load models.", hue: 250, stage: null },
    loading:      { label: "Downloading the brain…", sub: "Fetching model weights to your browser (cached for next time).", hue: 200, stage: null },
    idle:         { label: "Ready", sub: "Type below, press 🎤, or hold Space and talk.", hue: 255, stage: null },
    listening:    { label: "Listening…", sub: "Speak now - I'll stop when you pause.", hue: 330, stage: "mic" },
    transcribing: { label: "Transcribing…", sub: "Whisper is turning your speech into text, locally.", hue: 190, stage: "asr" },
    thinking:     { label: "Thinking…", sub: "The language model is reading the conversation.", hue: 270, stage: "llm" },
    writing:      { label: "Writing…", sub: "Generating the reply token by token on your GPU.", hue: 280, stage: "llm" },
    speaking:     { label: "Speaking…", sub: "Reading the reply aloud with an on-device voice.", hue: 160, stage: "tts" },
  };

  function currentState() {
    if (S.rec) return "listening";
    if (S.transcribing) return "transcribing";
    if (S.busy) return current && current.text ? "writing" : "thinking";
    if ("speechSynthesis" in window && speechSynthesis.speaking) return "speaking";
    if (S.loading) return "loading";
    return S.llmReady ? "idle" : "off";
  }

  function startOrb() {
    const cv = $id("va-orb"), g = cv.getContext("2d");
    const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2, R = W * 0.3;
    let shown = null, hue = 250, energy = 0, t = 0;
    const stages = ["mic", "asr", "llm", "tts"];

    function frame() {
      if (!document.body.contains(cv)) return;           // UI was rebuilt
      t += 1 / 60;
      const st = currentState(), def = STATES[st];

      // header text + pipeline only change when the state changes
      if (st !== shown) {
        shown = st;
        $id("va-state").innerHTML = def.label + "<small>" + def.sub + "</small>";
        stages.forEach((k, i) => {
          $id("pp-" + k).classList.toggle("on", def.stage === k);
          if (i) $id("pa-" + i).classList.toggle("on", stages.indexOf(def.stage) === i);
        });
      }

      // target "energy" of the orb for each state
      let target = 0.15;
      if (st === "listening") target = 0.25 + Math.min(1, S.level * 14);
      else if (st === "thinking" || st === "transcribing") target = 0.45;
      else if (st === "writing") target = 0.6 + 0.2 * Math.sin(t * 9);
      else if (st === "speaking") target = 0.5 + 0.35 * Math.abs(Math.sin(t * 6.5));
      else if (st === "loading") target = 0.35;
      energy += (target - energy) * 0.12;
      hue += (((def.hue - hue + 540) % 360) - 180) * 0.05;

      g.clearRect(0, 0, W, H);

      // outer glow
      const glow = g.createRadialGradient(cx, cy, R * 0.3, cx, cy, R * 1.65);
      glow.addColorStop(0, `hsla(${hue},95%,65%,${0.35 + energy * 0.3})`);
      glow.addColorStop(1, "hsla(0,0%,0%,0)");
      g.fillStyle = glow; g.fillRect(0, 0, W, H);

      // three wobbly blob layers, blended additively
      g.globalCompositeOperation = "lighter";
      for (let L = 0; L < 3; L++) {
        const hL = hue + (L - 1) * 38, rot = t * (0.4 + L * 0.25) * (L % 2 ? -1 : 1);
        g.beginPath();
        for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI / 48) {
          const wob = Math.sin(a * 3 + rot * 2) * 0.06 + Math.sin(a * 5 - rot * 3 + L) * 0.04 + Math.sin(a * 2 + t * 1.3 + L * 2) * 0.05;
          const r = R * (0.82 + L * 0.06) * (1 + wob * (0.6 + energy * 2.2) + energy * 0.12);
          const x = cx + Math.cos(a + rot) * r, y = cy + Math.sin(a + rot) * r;
          a === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
        }
        const fill = g.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.1, cx, cy, R * 1.2);
        fill.addColorStop(0, `hsla(${hL},100%,80%,.75)`);
        fill.addColorStop(0.55, `hsla(${hL},90%,55%,.45)`);
        fill.addColorStop(1, `hsla(${hL + 40},90%,40%,.05)`);
        g.fillStyle = fill; g.fill();
      }
      g.globalCompositeOperation = "source-over";

      // loading: progress ring.  thinking: orbiting dots.
      if (st === "loading") {
        const p = loadFraction();
        g.lineWidth = 5; g.lineCap = "round";
        g.strokeStyle = "rgba(255,255,255,.12)"; g.beginPath(); g.arc(cx, cy, R * 1.32, 0, Math.PI * 2); g.stroke();
        g.strokeStyle = `hsl(${hue},95%,70%)`; g.beginPath(); g.arc(cx, cy, R * 1.32, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * p); g.stroke();
      }
      if (st === "thinking" || st === "transcribing") {
        for (let i = 0; i < 6; i++) {
          const a = t * 2.4 + i * Math.PI / 3, rr = R * 1.28;
          g.fillStyle = `hsla(${hue + i * 12},100%,80%,${0.35 + 0.65 * ((i + 1) / 6)})`;
          g.beginPath(); g.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, 3 + i * 0.6, 0, Math.PI * 2); g.fill();
        }
      }
      // white core highlight
      const core = g.createRadialGradient(cx - R * 0.25, cy - R * 0.3, 0, cx - R * 0.25, cy - R * 0.3, R * 0.55);
      core.addColorStop(0, "rgba(255,255,255,.55)"); core.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = core; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();

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
      grad.addColorStop(0, "#22d3ee"); grad.addColorStop(1, "#7c5cff");
      g.fillStyle = grad;
      g.beginPath(); g.roundRect ? g.roundRect(x, y, bw - 4, h, 4) : g.rect(x, y, bw - 4, h); g.fill();
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
