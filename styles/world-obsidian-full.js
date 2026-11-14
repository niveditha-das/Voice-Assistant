// =====================================================================================
//  Offline Voice Assistant - small language model + Whisper + embeddings, in the browser
//  Personal project - runs entirely in the browser, no server
//  Author: <YOUR FULL NAME>            (replace before submitting)
// =====================================================================================
//
//  WHAT IT DOES
//  You speak (in almost any language) or type. Whisper turns speech into English text, a small
//  language model (SLM) writes the reply, and the browser reads it aloud. A third model (a
//  sentence-embedding network) lets the "Tutor" and "Quiz" modes search built-in sample notes.
//  All three neural networks are DOWNLOADED to the browser and RUN LOCALLY - no AI API is called.
//  After the models have loaded you can switch off Wi-Fi and it still works.
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
//                                                    https://huggingface.co/onnx-community/whisper-base
//                                                    https://github.com/openai/whisper
//      all-MiniLM-L6-v2 sentence embeddings          https://huggingface.co/Xenova/all-MiniLM-L6-v2
//                                                    https://www.sbert.net
//  - Built-in sample notes (paraphrased) from Mark Humphrys' public notes on neural networks and AI:
//      https://humphryscomputing.com/Notes/Neural/deep.html   https://humphryscomputing.com/ca318/notes.html
//  - Retrieval-augmented generation (RAG) idea: Lewis et al. 2020 https://arxiv.org/abs/2005.11401
//  - Streaming / interrupt pattern adapted from the official Transformers.js WebGPU chat
//    examples: https://github.com/huggingface/transformers.js-examples
//  - Web APIs (MDN): MediaRecorder https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder
//                    Web Audio     https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
//                    speechSynthesis https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis
//                    WebGPU        https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
//                    Web Workers   https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
//                    Cache Storage https://developer.mozilla.org/en-US/docs/Web/API/CacheStorage
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
//  - Chats, settings and benchmark results are saved in localStorage, on this device only.
// =====================================================================================


(function () {
  "use strict";

  // ---------------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------------

  const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

  // Language models. dtype (quantisation) chosen at runtime from what the GPU supports.
  const LLMS = [
    { id: "HuggingFaceTB/SmolLM2-135M-Instruct", short: "SmolLM2 135M", name: "SmolLM2 135M (fastest, ~120 MB)", params: "135M" },
    { id: "HuggingFaceTB/SmolLM2-360M-Instruct", short: "SmolLM2 360M", name: "SmolLM2 360M (balanced, ~270 MB)", params: "360M" },
    { id: "onnx-community/Qwen2.5-0.5B-Instruct", short: "Qwen2.5 0.5B", name: "Qwen2.5 0.5B (smarter, ~480 MB)", params: "500M" },
    { id: "onnx-community/Llama-3.2-1B-Instruct-q4f16", short: "Llama 3.2 1B", name: "Llama 3.2 1B (best, ~1.2 GB, GPU only)", params: "1.2B", gpuOnly: true },
  ];

  // Speech-to-text models. The multilingual one TRANSLATES any language into English text.
  const ASRS = [
    { id: "onnx-community/whisper-tiny.en", short: "Whisper tiny.en", name: "English only — Whisper tiny (~45 MB)", params: "39M", multilingual: false },
    { id: "onnx-community/whisper-base", short: "Whisper base", name: "Any language → English — Whisper base (~80 MB)", params: "74M", multilingual: true },
  ];

  const EMB = { id: "Xenova/all-MiniLM-L6-v2", short: "all-MiniLM-L6-v2", params: "23M" };

  // Personas = system prompts. Gives the assistant a "character".
  const PERSONAS = {
    assistant: "You are a friendly, helpful voice assistant. Keep answers short (2-4 sentences) because they will be read aloud.",
    tutor: "You are a patient tutor for AI topics (search, neural networks, deep learning, LLMs). " +
           "Explain clearly and simply, give a small example, and finish by asking the learner one short question to check understanding. Keep it short.",
    pirate: "You are a cheerful pirate captain. Answer every question correctly but in pirate speak. Keep answers short.",
  };

  // Modes change how a message is handled
  const MODES = {
    chat:      { label: "Chat",      title: "Ask me anything.",             hint: "Type below, or press the mic and speak in any language." },
    tutor:     { label: "Tutor",     title: "Ask about the sample notes.", hint: "Answers come from the built-in notes, and show which section they used." },
    quiz:      { label: "Quiz",      title: "Test yourself.",               hint: "The assistant asks a question from the notes, then marks your answer." },
    summarise: { label: "Summarise", title: "Make it short.",               hint: "Paste a long text below, or press the mic and read it out." },
  };
  const CHIPS = {
    chat: [["What is a neural network?", "Explain what a neural network is, simply."],
           ["ReLU vs sigmoid", "Why is ReLU better than sigmoid for deep networks?"],
           ["Tell me a joke", "Tell me a short joke about computers."]],
    tutor: [["What is ReLU?", "What is the ReLU function and why does it help deep networks?"],
            ["Why GPUs?", "Why are GPUs used for neural networks?"],
            ["Benchmark contamination", "What is benchmark contamination?"]],
    quiz: [["Start the quiz", "Start the quiz."]],
    summarise: [],
  };

  // ---------------------------------------------------------------------------------
  // Built-in sample notes for the Tutor / Quiz modes (paraphrased summaries; source pages credited).
  // Each section is turned into a 384-number "meaning vector" by the embedding model.
  // ---------------------------------------------------------------------------------

  const NOTES_SRC = "https://humphryscomputing.com/Notes/Neural/deep.html";
  const NOTES = [
    ["Deep learning", "Neural networks grew massively after 2005 and especially after 2017. The defining feature of deep learning is a way to scale neural networks to a huge size, with many layers. The original back-propagation algorithm does not scale well to many layers or to long time series, so new architectures were needed."],
    ["Vanishing gradient problem", "When back-propagation is used on networks with many layers, the error signal (gradient) gets smaller and smaller as it is passed back through the layers, so the early layers barely learn. This vanishing gradient problem stopped the original back-prop from scaling to deep networks."],
    ["LSTM", "Long short-term memory (LSTM) is a modification of back-propagation for recurrent networks. It lets recurrent networks learn from long sequences, which the original method could not do."],
    ["Deep belief nets (Hinton 2006)", "A breakthrough paper for deep neural networks was 'A fast learning algorithm for deep belief nets' by Hinton, Osindero and Teh (2006). The key idea was that weights are initialised by training in a certain way rather than randomly."],
    ["ReLU (rectifier)", "The rectifier function, or ReLU (rectified linear unit), is f(x) = max(0, x). Its slope is 0 below zero and constant above zero. Compared with the sigmoid, it makes many more nodes output exactly zero, giving sparse representations, so those nodes can be ignored by the next layer. It is also much faster to calculate than the sigmoid, which is very important for scaling up."],
    ["Neural networks on GPUs", "A neural network is mostly simple calculations that can be done in parallel, so running it on serial hardware is wasteful. GPUs are massively parallel processors for simple calculations, so implementing neural networks on GPUs became an important part of deep learning."],
    ["NPUs and TPUs", "Neural processing units (NPUs), also called AI accelerators, are becoming part of consumer PCs and phones. The tensor processing unit (TPU) was developed by Google for neural networks and is used on its AI servers."],
    ["Neural networks in JavaScript on the GPU", "WebGL lets JavaScript in the browser use the GPU. Its calculations do not have to be graphics: libraries such as TensorFlow.js use it for neural networks, which can be around 100 times faster than the plain CPU backend. Raw WebGL coding means the JavaScript builds a second program, a shader written in GLSL, compiles it and sends it to the GPU."],
    ["Image recognition with MobileNet", "The 'Recognise any image' World on Ancient Brain uses the ml5.js library's image classifier with a pre-trained model called MobileNet. MobileNet was trained on a sample of a database of around 15 million images, is a bit over 10 MB, and is downloaded to the browser when the page runs."],
    ["Applications of neural networks", "Neural networks are used for image recognition, machine translation (for example Google Neural Machine Translation), face recognition (Facebook DeepFace), text understanding, image search (Pinterest), optical character recognition, handwriting recognition, recommendation systems (Amazon DSSTNE) and algorithmic trading."],
    ["Nobel Prizes for AI", "In 2024 John Hopfield and Geoffrey Hinton won the Nobel Prize in Physics for work behind neural networks and deep learning. The inventors of AlphaFold, an AI system that predicts protein structure, won the Nobel Prize in Chemistry in 2024."],
    ["The brain", "The brain has around 86 to 100 billion neurons, each with up to thousands of connections, all working in parallel. It may be the most complex object known. Estimates from 1999 and 2008 compared all the world's computers to one brain, but the brain's dense connectivity still puts it ahead."],
    ["LLM size", "GPT-3 from OpenAI has 96 layers of multilayer perceptrons plus 96 attention layers. Very large AI models are thought to have between 1 trillion and 10 trillion weights."],
    ["Pre-trained and small models", "We are in a golden age of using other people's pre-trained models, after vast amounts of training. They can be used by local install, by remote API call, or through a remote website. Small language models (SLMs) are small enough to download and run locally, even in a web browser."],
    ["Benchmark contamination", "If a set of question-and-answer examples becomes a popular benchmark, it is published online, and large language models are trained on online text, including the benchmark itself. A model may then answer by associating the question text with the answer, without understanding. So with other people's models it is hard to guarantee that the test set is new and unseen, which breaks the rule that test data must not overlap training data."],
  ];

  // Quiz question bank: [note title, question, model answer]. Marking compares the meaning of the
  // learner's answer with the model answer using sentence embeddings (cosine similarity).
  const QUIZ = [
    ["ReLU (rectifier)", "What is the formula for the ReLU (rectifier) activation function?", "ReLU is f(x) = max(0, x): it outputs zero for negative inputs and x for positive inputs."],
    ["ReLU (rectifier)", "Why does ReLU help deep networks compared with the sigmoid?", "ReLU makes many nodes output zero, giving sparse representations, and it is much faster to calculate than the sigmoid."],
    ["Vanishing gradient problem", "What is the vanishing gradient problem?", "In deep networks the error gradient gets smaller as it is passed back through the layers, so the early layers barely learn."],
    ["LSTM", "What problem does LSTM (long short-term memory) solve?", "LSTM lets recurrent networks learn from long sequences, which the original back-propagation could not do."],
    ["Deep belief nets (Hinton 2006)", "What was the key idea of Hinton's 2006 deep belief nets paper?", "The weights are initialised by training in a certain way instead of randomly."],
    ["Neural networks on GPUs", "Why are GPUs good for running neural networks?", "Neural networks are many simple calculations that can be done in parallel, and GPUs are massively parallel processors."],
    ["NPUs and TPUs", "What is a TPU?", "A tensor processing unit is a chip developed by Google for neural networks, used on its AI servers."],
    ["Neural networks in JavaScript on the GPU", "How can JavaScript in a web browser use the GPU for neural networks?", "Through WebGL, which libraries like TensorFlow.js use to run neural network calculations on the GPU."],
    ["Image recognition with MobileNet", "Which pre-trained model does the 'Recognise any image' World use, and how big is it?", "It uses MobileNet through ml5.js, a model of a bit over 10 MB downloaded to the browser."],
    ["Nobel Prizes for AI", "Who won the 2024 Nobel Prize in Physics for work on neural networks?", "John Hopfield and Geoffrey Hinton."],
    ["LLM size", "How many layers does GPT-3 have?", "GPT-3 has 96 layers of multilayer perceptrons plus 96 attention layers."],
    ["Pre-trained and small models", "What are three ways to use someone else's pre-trained model?", "Local install, a remote API call, or a remote website."],
    ["Benchmark contamination", "What is benchmark contamination?", "A popular benchmark is published online, models train on it, so they may answer from memorised text instead of understanding, and the test set is not really unseen."],
    ["Deep learning", "What is the defining feature of deep learning?", "A way to scale neural networks to a huge size with many layers."],
  ];

  // ---------------------------------------------------------------------------------
  // The Web Worker. All the AI runs here, off the main thread.
  // This function is turned into a string and run as a worker from a Blob URL,
  // so the whole World stays in ONE file on Ancient Brain.
  // ---------------------------------------------------------------------------------

  function workerMain() {
    let T = null;          // the Transformers.js library
    let llm = null;        // text-generation pipeline
    let asr = null;        // speech-recognition pipeline
    let emb = null;        // feature-extraction (sentence embeddings) pipeline
    let stopper = null;    // lets the user interrupt generation
    let asrQueue = Promise.resolve();   // Whisper calls run one at a time (live captions + final)

    const post = (msg) => self.postMessage(msg);
    const progress = (which) => (p) => post({ type: "progress", which, status: p.status, file: p.file || "", loaded: p.loaded || 0, total: p.total || 0 });

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
          await llm([{ role: "user", content: "Hi" }], { max_new_tokens: 1 });   // warm-up: compiles GPU shaders now
          post({ type: "loaded", which: "llm", ms: performance.now() - t0 });
        }

        else if (m.type === "loadASR") {
          if (asr) { try { await asr.dispose(); } catch (err) {} asr = null; }
          const t0 = performance.now();
          asr = await T.pipeline("automatic-speech-recognition", m.id, { device: m.device, dtype: m.dtype, progress_callback: progress("asr") });
          await asr(new Float32Array(16000));      // warm-up with 1 second of silence
          post({ type: "loaded", which: "asr", ms: performance.now() - t0 });
        }

        else if (m.type === "loadEMB") {
          const t0 = performance.now();
          // small model: runs on the CPU (WASM) so it never competes with the language model for the GPU
          emb = await T.pipeline("feature-extraction", m.id, { device: "wasm", dtype: "q8", progress_callback: progress("emb") });
          post({ type: "loaded", which: "emb", ms: performance.now() - t0 });
        }

        else if (m.type === "embed") {
          const out = await emb(m.texts, { pooling: "mean", normalize: true });
          post({ type: "embedded", rid: m.rid, vectors: out.tolist() });
        }

        else if (m.type === "transcribe") {
          asrQueue = asrQueue.then(async () => {
            const t0 = performance.now();
            // multilingual Whisper: no language given -> it detects the language; task "translate" -> English text
            const opts = m.multilingual ? { task: "translate" } : {};
            const out = await asr(m.audio, opts);
            post({ type: "transcript", partial: !!m.partial, text: (out.text || "").trim(), ms: performance.now() - t0, seconds: m.audio.length / 16000 });
          }).catch((err) => post({ type: "error", where: "transcribe", message: (err && err.message) || String(err) }));
        }

        else if (m.type === "generate") {
          stopper = new T.InterruptableStoppingCriteria();
          let nTokens = 0, tFirst = null;
          const t0 = performance.now();
          const streamer = new T.TextStreamer(llm.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text) => post({ type: "token", rid: m.rid, text }),
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
          post({ type: "done", rid: m.rid, text, tokens: nTokens, ttft: tFirst ? tFirst - t0 : 0,
                 tps: genSecs > 0 ? (nTokens - 1) / genSecs : 0, total: t1 - t0 });
        }

        else if (m.type === "interrupt") {
          if (stopper) stopper.interrupt();
        }
      } catch (err) {
        post({ type: "error", where: m.type, rid: m.rid, message: (err && err.message) ? err.message : String(err) });
      }
    };
  }

  // ---------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------

  const S = {
    worker: null,
    device: "wasm", hasF16: false, hasGPU: false,
    llmReady: false, asrReady: false, embReady: false, embLoading: false,
    busy: false,            // generating
    history: [],            // [{role, content, sources?}]
    results: [],            // experiment log rows (CSV)
    llmLoadMs: 0, asrLoadMs: 0,
    rec: null, speakBuffer: "", files: {}, sizes: {},
    conversation: false, loading: false, transcribing: false, level: 0, tpsHist: [], progressText: "",
    chats: [], chatId: null, mode: "chat",
    notes: [],               // [{title, text, src, vec}]
    quiz: { score: 0, total: 0, item: null },
    bench: [], benchText: "",
    gen: null,               // the generation in progress {rid, onToken, resolve, reject}
    embReq: {},              // pending embedding requests
    barge: null,             // interrupt-by-speaking monitor
  };

  const $id = (id) => document.getElementById(id);
  const LS = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { console.warn("Could not save " + k, e); } },
  };

  // ---------------------------------------------------------------------------------
  // Styles
  // Design: "obsidian monument with crimson signal" - pure black canvas, ash (#ccc) body, frost (#fff) display,
  // graphite (#4c4c4c) meta, ONE crimson pill for the primary action, transparent components, hairlines, no shadows.
  // A dark obsidian liquid sphere is the only decoration. Light theme = the same system inverted.
  // Style reference: https://styles.refero.design/style/82d52a5f-b1bb-4a69-91a3-15a7eb8bbe99
  // Font: Space Grotesk (substitute for Sui) from Google Fonts https://fonts.google.com/specimen/Space+Grotesk
  // ---------------------------------------------------------------------------------

  const CSS = `
  #va-root{--bg:#000;--text:#ccc;--strong:#fff;--meta:#5a5a5a;--hair:rgba(255,255,255,.14);--panel:rgba(0,0,0,.6);--crimson:#fc1c46;
    --ease:cubic-bezier(0.19,1,0.22,1);--sideW:288px;
    position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;overflow:hidden;background:var(--bg);color:var(--text);
    font-family:"Space Grotesk","Sui",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;font-weight:400;line-height:1.15;
    -webkit-font-smoothing:antialiased;transition:background .6s ease,color .6s ease}
  #va-root.light{--bg:#fff;--text:#333;--strong:#000;--meta:#8c8c8c;--hair:rgba(0,0,0,.13);--panel:rgba(255,255,255,.7)}
  #va-root *{box-sizing:border-box}
  #va-root ::selection{background:var(--crimson);color:#fff}
  #va-root ::-webkit-scrollbar{width:6px}#va-root ::-webkit-scrollbar-thumb{background:var(--meta);border-radius:9999px}
  #va-root input,#va-root textarea{font:inherit}

  #va-liquid{position:absolute;inset:0;width:100%;height:100%;display:block;z-index:0}
  #va-body{position:relative;z-index:1;flex:1;display:flex;min-height:0}

  /* pills only. Ghost by default; ONE crimson fill for the primary action in view */
  .va-btn{border:1px solid var(--hair);border-radius:9999px;padding:0 22px;height:40px;background:transparent;color:var(--strong);font:inherit;font-size:15px;
    text-transform:uppercase;letter-spacing:-.2px;cursor:pointer;white-space:nowrap;transition:border-color .6s var(--ease),opacity .4s ease,background .6s var(--ease)}
  .va-btn:hover{border-color:var(--strong)}
  .va-btn:disabled{opacity:.25;cursor:not-allowed;border-color:var(--hair)}
  .va-btn.solid,#va-send:not(:disabled){background:var(--crimson);border-color:var(--crimson);color:#fff;padding:0 30px}
  .va-btn.solid:hover,#va-send:not(:disabled):hover{border-color:var(--crimson);opacity:.88}
  .va-btn.sec{font-size:12px;height:34px;padding:0 16px}
  .va-btn.red{border-color:var(--strong)}
  .va-link{background:none;border:0;padding:0;font:inherit;font-size:14px;color:var(--strong);cursor:pointer;text-align:left;
    text-decoration:underline;text-decoration-color:transparent;text-underline-offset:4px;transition:text-decoration-color .4s ease,color .4s ease}
  .va-link:hover{text-decoration-color:currentColor}
  .va-link.dim{color:var(--meta)}.va-link.dim:hover{color:var(--strong)}

  /* ---- collapsible sidebar ---- */
  #va-side{width:var(--sideW);flex:none;display:flex;flex-direction:column;border-right:1px solid var(--hair);background:var(--panel);transition:margin-left 1s var(--ease)}
  #va-root.side-hidden #va-side{margin-left:calc(-1 * var(--sideW) - 1px)}
  #va-brand{height:72px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-size:15px;color:var(--strong);font-weight:500}
  #va-brand span{display:flex;align-items:center;gap:9px}
  #va-brand span::before{content:"";width:10px;height:10px;border-radius:9999px;background:var(--crimson)}
  #va-new{margin:9px 22px 22px;width:calc(100% - 44px)}
  #va-search{margin:0 22px 22px;width:calc(100% - 44px);background:transparent;border:0;border-bottom:1px solid var(--hair);border-radius:0;color:var(--strong);
    padding:7px 2px;font-size:14px;outline:none;transition:border-color .4s ease}
  #va-search:focus{border-bottom-color:var(--strong)}
  #va-search::placeholder{color:var(--meta)}
  .va-label{font-size:10px;text-transform:uppercase;color:var(--meta);padding:13px 22px 7px;letter-spacing:.3px}
  #va-hist{flex:1;overflow-y:auto;padding:0 0 12px}
  .va-h{display:flex;align-items:center;gap:7px;padding:8px 22px;cursor:pointer;color:var(--text);transition:color .4s ease}
  .va-h:hover,.va-h.on{color:var(--strong)}
  .va-h.on .ttl::before{content:"— "}
  .va-h .ttl{flex:1;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .va-h input{flex:1;min-width:0;background:transparent;border:0;border-bottom:1px solid var(--strong);color:var(--strong);font-size:14px;outline:none;padding:1px 0}
  .va-h .ha{display:none;gap:9px}
  .va-h:hover .ha{display:flex}
  .va-h .ha button{font-size:10px;text-transform:uppercase;line-height:1;color:var(--meta);text-decoration:none}
  .va-h .ha button:hover{color:var(--strong)}
  .va-empty{font-size:14px;color:var(--meta);padding:0 22px}
  #va-side-foot{border-top:1px solid var(--hair);padding:18px 22px 9px;display:flex;justify-content:space-between;align-items:center;gap:13px}
  #va-status{font-size:10px;text-transform:uppercase;color:var(--meta);display:flex;align-items:center;gap:7px;letter-spacing:.3px;margin-left:auto}
  #va-status::before{content:"";width:6px;height:6px;border-radius:9999px;border:1px solid var(--meta)}
  #va-status.ok{color:var(--text)}#va-status.ok::before{background:var(--strong);border-color:var(--strong)}
  #va-credit{padding:0 22px 18px;font-size:10px;color:var(--meta);line-height:1.25}

  /* ---- main ---- */
  #va-main{position:relative;flex:1;display:flex;flex-direction:column;min-width:0}
  #va-hero{flex:none;height:72px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:22px;padding:0 72px}
  #va-hero .l,#va-hero .r{display:flex;align-items:center;gap:22px;min-width:0}
  #va-hero .r{justify-self:end}
  #va-menu{display:none}
  #va-root.side-hidden #va-menu{display:inline-block}
  #va-state{font-size:13px;text-transform:uppercase;color:var(--text);display:flex;align-items:baseline;gap:9px;white-space:nowrap}
  #va-state small{font-size:10px;color:var(--meta);font-variant-numeric:tabular-nums}
  #va-tabs{display:flex;gap:6px}
  .va-tab{background:none;border:1px solid transparent;border-radius:9999px;height:32px;padding:0 15px;font:inherit;font-size:13px;text-transform:uppercase;color:var(--meta);cursor:pointer;
    transition:color .4s ease,border-color .4s ease}
  .va-tab:hover{color:var(--strong)}
  .va-tab.on{color:var(--strong);border-color:var(--hair)}
  .va-pill{font-size:10px;text-transform:uppercase;letter-spacing:.3px;color:var(--strong);border:1px solid var(--hair);border-radius:9999px;padding:5px 11px;white-space:nowrap}
  #va-offline{display:none}
  #va-root.is-offline #va-offline{display:inline-block}
  #va-score{display:none}
  #va-root.mode-quiz #va-score{display:inline-block}
  #va-export-wrap{position:relative;display:none}
  #va-root.has-msgs #va-export-wrap{display:block}
  #va-export-menu{position:absolute;right:0;top:30px;z-index:6;display:none;flex-direction:column;gap:9px;padding:13px 18px;background:var(--bg);border:1px solid var(--hair);min-width:140px}
  #va-export-menu.on{display:flex}

  #va-chat{flex:1;overflow-y:auto;padding:22px 72px 22px;display:flex;flex-direction:column;gap:36px}
  .va-msg{position:relative;max-width:680px;line-height:1.25;word-wrap:break-word;font-size:18px;color:var(--text);animation:vain 1s var(--ease)}
  @keyframes vain{from{opacity:0;transform:translateY(12px)}}
  .va-msg.user{align-self:flex-end;color:var(--strong);font-size:27px;line-height:1.2;max-width:620px;white-space:pre-wrap}
  .va-msg.user.caption{color:var(--meta);font-style:italic}
  .va-msg.user.long{font-size:16px;line-height:1.4;max-width:680px}
  .va-msg.user::before,.va-msg.assistant::before{display:block;font-size:10px;font-style:normal;text-transform:uppercase;color:var(--meta);margin-bottom:9px;line-height:1;letter-spacing:.3px}
  .va-msg.user::before{content:"You"}
  .va-msg.user.caption::before{content:"Listening — live captions"}
  .va-msg.assistant{align-self:flex-start;width:min(680px,100%)}
  .va-msg.assistant::before{content:"Assistant"}
  .va-msg.system{align-self:flex-start;font-size:14px;color:var(--meta)}
  .va-msg .body p{margin:0 0 .7em}.va-msg .body p:last-child{margin-bottom:0}
  .va-msg .body ul,.va-msg .body ol{margin:0 0 .7em;padding-left:1.3em}.va-msg .body li{margin:.25em 0}
  .va-msg .body h4{font-size:18px;font-weight:700;color:var(--strong);margin:.6em 0 .4em;text-transform:uppercase}
  .va-msg .body pre{margin:.4em 0 .8em;padding:13px 0;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);overflow-x:auto;
    font-family:ui-monospace,Menlo,monospace;font-size:14px;line-height:1.45;color:var(--strong);white-space:pre}
  .va-msg code{font-family:ui-monospace,Menlo,monospace;font-size:.86em;color:var(--strong)}
  .va-msg b{font-weight:700;color:var(--strong)}
  .va-acts{display:flex;gap:18px;margin-top:13px;opacity:0;transition:opacity .4s ease}
  .va-msg:hover .va-acts,.va-msg.assistant.last .va-acts{opacity:1}
  .va-acts button{font-size:11px;text-transform:uppercase;letter-spacing:.3px;color:var(--meta);text-decoration:none}
  .va-acts button:hover{color:var(--strong)}
  .va-msg.user .va-acts{justify-content:flex-end;font-style:normal}
  .va-msg:not(.last) .act-last{display:none}
  .va-src{display:flex;flex-wrap:wrap;gap:7px;margin-top:13px;align-items:center}
  .va-src span{font-size:10px;text-transform:uppercase;color:var(--meta);letter-spacing:.3px;margin-right:2px}
  .va-src a{font-size:11px;color:var(--strong);border:1px solid var(--hair);border-radius:9999px;padding:4px 10px;text-decoration:none;transition:border-color .4s ease}
  .va-src a:hover{border-color:var(--strong)}

  #va-welcome{max-width:900px;margin:auto 0;padding-bottom:65px;animation:vain 1s var(--ease)}
  #va-welcome h3{font-size:91px;font-weight:700;line-height:.92;letter-spacing:-1.82px;text-transform:uppercase;color:var(--strong);margin:0 0 36px}
  #va-welcome p{font-size:18px;line-height:1.1;margin:0 0 36px;max-width:440px}
  .va-chips{display:flex;flex-wrap:wrap;gap:9px}
  .va-chip{font:inherit;font-size:15px;text-transform:uppercase;color:var(--strong);height:40px;padding:0 22px;border-radius:9999px;background:transparent;border:1px solid var(--hair);cursor:pointer;
    transition:border-color .6s var(--ease)}
  .va-chip:hover{border-color:var(--strong)}
  .va-small{font-size:10px;text-transform:uppercase;color:var(--meta)}
  .va-bar{height:1px;background:var(--hair);overflow:hidden;max-width:440px;margin-top:9px}
  .va-bar>div{height:100%;width:0;background:var(--strong);transition:width 1s var(--ease)}

  /* input: hairline only */
  #va-input{display:flex;gap:9px;padding:13px 72px 36px;align-items:flex-end}
  #va-text{flex:1;resize:none;min-height:44px;max-height:180px;background:transparent;color:var(--strong);border:0;border-bottom:1px solid var(--strong);border-radius:0;outline:none;
    padding:9px 2px;font-size:18px;line-height:1.25;margin-right:13px}
  #va-text::placeholder{color:var(--meta)}
  #va-text:disabled{border-bottom-color:var(--meta)}
  #va-mic{position:relative;width:40px;height:40px;padding:0;font-size:0;flex:none}
  #va-mic::before{content:"";position:absolute;left:50%;top:50%;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;border-radius:9999px;background:var(--strong);transition:transform .6s var(--ease)}
  #va-mic:hover::before{transform:scale(1.3)}
  #va-mic.rec{border-color:var(--strong)}
  #va-mic.rec::before{background:var(--crimson);transform:scale(1.2)}
  #va-mic.rec::after{content:"";position:absolute;inset:-1px;border-radius:9999px;border:1px solid var(--strong);animation:varing 1.6s var(--ease) infinite}
  @keyframes varing{from{transform:scale(1);opacity:1}to{transform:scale(1.8);opacity:0}}
  .va-cursor{display:inline-block;width:8px;height:17px;background:var(--strong);vertical-align:-3px;margin-left:3px;animation:vablink 1s steps(2) infinite}
  @keyframes vablink{50%{opacity:0}}
  .va-dots span{display:inline-block;width:6px;height:6px;margin-right:6px;border-radius:9999px;background:var(--strong);animation:vadot 1.2s var(--ease) infinite}
  .va-dots span:nth-child(2){animation-delay:.15s}.va-dots span:nth-child(3){animation-delay:.3s}
  @keyframes vadot{0%,80%,100%{opacity:.2}40%{opacity:1}}

  /* ---- settings: overlay panel ---- */
  #va-scrim{position:absolute;inset:0;z-index:20;background:rgba(0,0,0,.55);opacity:0;pointer-events:none;transition:opacity .8s var(--ease)}
  #va-settings{position:absolute;z-index:21;top:0;right:0;bottom:0;width:440px;max-width:100%;background:var(--bg);border-left:1px solid var(--hair);overflow-y:auto;
    transform:translateX(101%);transition:transform 1s var(--ease)}
  #va-root.set-open #va-scrim{opacity:1;pointer-events:auto}
  #va-root.set-open #va-settings{transform:none}
  #va-set-head{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 36px;position:sticky;top:0;background:var(--bg);z-index:1;
    font-size:27px;font-weight:700;letter-spacing:-.5px;text-transform:uppercase;color:var(--strong)}
  .va-card{padding:22px 36px 36px;border-top:1px solid var(--hair)}
  .va-card h2{display:flex;gap:9px;font-size:15px;font-weight:400;text-transform:uppercase;color:var(--text);margin:0 0 22px}
  .va-card h2 b{font-weight:400;color:var(--meta)}
  #va-settings label{display:block;font-size:10px;text-transform:uppercase;color:var(--meta);margin:22px 0 7px;letter-spacing:.3px}
  #va-settings h2 + label{margin-top:0}
  #va-settings select{width:100%;appearance:none;-webkit-appearance:none;border:0;border-bottom:1px solid var(--strong);border-radius:0;padding:7px 22px 9px 2px;
    background:transparent url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23888' fill='none'/%3E%3C/svg%3E") no-repeat right 2px center;
    font-size:15px;color:var(--strong);cursor:pointer;text-overflow:ellipsis}
  #va-settings select:focus{outline:none}
  #va-settings select option{background:var(--bg);color:var(--strong)}
  #va-settings textarea{width:100%;min-height:84px;resize:vertical;background:transparent;border:0;border-bottom:1px solid var(--strong);color:var(--strong);font-size:14px;line-height:1.3;outline:none;padding:7px 2px}
  #va-settings input[type=range]{width:100%;-webkit-appearance:none;appearance:none;height:1px;background:var(--strong);outline:none;margin:12px 0 4px}
  #va-settings input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:9999px;background:var(--strong);cursor:pointer}
  #va-settings .va-lbl{display:flex;justify-content:space-between}#va-settings .va-lbl b{font-weight:400;color:var(--strong)}
  #va-settings .va-switch{display:flex;align-items:center;justify-content:space-between;gap:9px;margin:0;padding:13px 0;border-bottom:1px solid var(--hair);
    font-size:15px;color:var(--text);cursor:pointer}
  #va-settings .va-switch input{display:none}
  #va-settings .va-switch i{flex:none;position:relative;width:34px;height:18px;border-radius:9999px;border:1px solid var(--strong);transition:background .6s var(--ease)}
  #va-settings .va-switch i::after{content:"";position:absolute;left:3px;top:3px;width:10px;height:10px;border-radius:9999px;background:var(--strong);transition:transform .6s var(--ease),background .6s var(--ease)}
  #va-settings .va-switch input:checked + i{background:var(--strong)}
  #va-settings .va-switch input:checked + i::after{transform:translateX(16px);background:var(--bg)}
  .va-row{display:flex;gap:9px;align-items:flex-end}
  .va-row select{flex:1}
  #va-reload{width:100%;margin-top:22px}
  #va-file{font-size:12px;color:var(--meta);width:100%}
  #va-file::file-selector-button{font:inherit;font-size:12px;text-transform:uppercase;color:var(--strong);background:transparent;border:1px solid var(--hair);border-radius:9999px;padding:7px 16px;margin-right:12px;cursor:pointer}
  .va-info{display:grid;grid-template-columns:auto 1fr;gap:7px 18px;font-size:13px;margin-bottom:9px}
  .va-info dt{font-size:10px;text-transform:uppercase;color:var(--meta);letter-spacing:.3px;padding-top:2px}
  .va-info dd{margin:0;color:var(--strong)}
  .va-info dd small{display:block;color:var(--meta);font-size:11px;margin-top:2px}
  .va-stats{display:grid;grid-template-columns:1fr 1fr;column-gap:22px}
  .va-stat{padding:12px 0;border-bottom:1px solid var(--hair)}
  .va-stat .k{font-size:10px;text-transform:uppercase;color:var(--meta);letter-spacing:.3px}
  .va-stat .v{font-size:27px;font-weight:700;letter-spacing:-.5px;color:var(--strong);margin-top:7px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #va-spark{width:100%;height:52px;display:block;margin-top:22px}
  #va-bench-chart{width:100%;display:block;margin-top:18px}
  .va-keys{display:grid;grid-template-columns:auto 1fr;gap:9px 18px;font-size:13px}
  .va-keys kbd{font-family:inherit;font-size:11px;text-transform:uppercase;color:var(--strong);border:1px solid var(--hair);border-radius:9999px;padding:3px 9px;justify-self:start}
  .va-btns{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap}
  .va-btns .va-btn{flex:1}
  .va-note{font-size:12px;color:var(--meta);line-height:1.3;margin-top:9px}

  #va-toast{position:absolute;z-index:30;left:50%;bottom:110px;transform:translate(-50%,16px);opacity:0;pointer-events:none;background:var(--bg);color:var(--strong);border:1px solid var(--hair);
    font-size:14px;padding:11px 22px;border-radius:9999px;transition:opacity .8s var(--ease),transform .8s var(--ease);max-width:80%;text-align:center}
  #va-toast.on{opacity:1;transform:translate(-50%,0)}

  @media (max-width:1200px){ #va-welcome h3{font-size:72px;letter-spacing:-1.4px} #va-hero{padding:0 36px} #va-chat,#va-input{padding-left:36px;padding-right:36px} }
  @media (max-width:900px){
    #va-side{position:absolute;z-index:15;left:0;top:0;bottom:0;background:var(--bg)}
    #va-menu{display:inline-block}
    #va-hero{grid-template-columns:auto 1fr;padding:0 18px} #va-hero .r{display:none} #va-state{display:none}
    #va-welcome h3{font-size:54px;letter-spacing:-1px}
    #va-chat,#va-input{padding-left:18px;padding-right:18px}
  }
  `;

  // ---------------------------------------------------------------------------------
  // Page structure
  // ---------------------------------------------------------------------------------

  const HTML = `
      <canvas id="va-liquid"></canvas>
      <div id="va-body">
        <div id="va-side">
          <div id="va-brand"><span>Offline voice assistant</span><button id="va-collapse" class="va-link dim" title="Hide sidebar">Hide</button></div>
          <button id="va-new" class="va-btn" title="New chat (N)">New chat</button>
          <input id="va-search" type="search" placeholder="Search chats" title="Search chats (Ctrl/Cmd + K)" autocomplete="off">
          <div id="va-hist"></div>
          <div id="va-side-foot">
            <button id="va-open-settings" class="va-link">Settings</button>
            <button id="va-theme" class="va-link dim" title="Light / dark">Light</button>
            <span id="va-status">not loaded</span>
          </div>
          <div id="va-credit">Built with Transformers.js, Whisper, SmolLM2 and MiniLM. Runs on this device.</div>
        </div>

        <div id="va-main">
          <div id="va-hero">
            <div class="l"><button id="va-menu" class="va-link" title="Show sidebar">Chats</button><div id="va-state">Ready to load</div></div>
            <div id="va-tabs"></div>
            <div class="r">
              <span id="va-offline" class="va-pill">Offline · running locally</span>
              <span id="va-score" class="va-pill">Score 0 / 0</span>
              <div id="va-export-wrap">
                <button id="va-export" class="va-link dim">Export</button>
                <div id="va-export-menu">
                  <button id="va-export-txt" class="va-link">Text file (.txt)</button>
                  <button id="va-export-pdf" class="va-link">PDF</button>
                </div>
              </div>
            </div>
          </div>
          <div id="va-chat"></div>
          <div id="va-input">
            <textarea id="va-text" rows="1" placeholder="Load the models to start…" disabled></textarea>
            <button id="va-send" class="va-btn" disabled>Send</button>
            <button id="va-stop" class="va-btn red" style="display:none" title="Stop (Esc)">Stop</button>
            <button id="va-mic" class="va-btn" title="Click to talk (or hold Space)" disabled>Mic</button>
          </div>
        </div>
      </div>

      <div id="va-scrim"></div>
      <div id="va-settings">
        <div id="va-set-head"><span>Settings</span><button id="va-close-settings" class="va-link">Close</button></div>

        <div class="va-card">
          <h2><b>01</b> Models</h2>
          <dl class="va-info" id="va-info"></dl>
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
          <div class="va-btns"><button id="va-clear-cache" class="va-btn sec">Clear downloaded models</button></div>
          <div class="va-note" id="va-storage"></div>
        </div>

        <div class="va-card">
          <h2><b>02</b> Assistant</h2>
          <label for="va-persona">Persona (Chat mode)</label>
          <select id="va-persona">
            <option value="assistant">Helpful assistant</option>
            <option value="tutor">AI module revision tutor</option>
            <option value="pirate">Pirate captain</option>
            <option value="custom">Custom…</option>
          </select>
          <div id="va-custom-wrap" style="display:none">
            <label for="va-custom">Your persona</label>
            <textarea id="va-custom" placeholder="e.g. You are a calm football coach who explains things with sports examples. Keep answers short."></textarea>
          </div>
          <label for="va-temp" class="va-lbl"><span>Creativity (temperature)</span><b id="va-tempv">0.3</b></label>
          <input id="va-temp" type="range" min="0" max="1.2" step="0.1" value="0.3">
          <label for="va-max" class="va-lbl"><span>Max reply length (tokens)</span><b id="va-maxv">200</b></label>
          <input id="va-max" type="range" min="32" max="512" step="16" value="200">
        </div>

        <div class="va-card">
          <h2><b>03</b> Voice</h2>
          <label class="va-switch"><span>Read replies aloud</span><input type="checkbox" id="va-speak" checked><i></i></label>
          <label class="va-switch"><span>Live captions while I speak</span><input type="checkbox" id="va-captions" checked><i></i></label>
          <label class="va-switch"><span>Interrupt by speaking</span><input type="checkbox" id="va-barge" checked><i></i></label>
          <label class="va-switch"><span>Auto-stop when I pause</span><input type="checkbox" id="va-auto" checked><i></i></label>
          <label class="va-switch"><span>Hands-free conversation</span><input type="checkbox" id="va-conv"><i></i></label>
          <label for="va-voice">Speaking voice</label>
          <div class="va-row"><select id="va-voice"></select><button id="va-preview" class="va-btn sec">Preview</button></div>
          <label for="va-rate" class="va-lbl"><span>Speaking speed</span><b id="va-ratev">1.0×</b></label>
          <input id="va-rate" type="range" min="0.6" max="1.6" step="0.1" value="1.0">
          <label for="va-file">Ask with an audio file</label>
          <input id="va-file" type="file" accept="audio/*" disabled>
          <div class="va-note">Pick "Any language → English" above to speak in French, Hindi, Spanish and many more.</div>
        </div>

        <div class="va-card">
          <h2><b>04</b> Performance</h2>
          <div class="va-stats">
            <div class="va-stat"><div class="k">Tokens / s</div><div class="v" id="st-tps">–</div></div>
            <div class="va-stat"><div class="k">First token</div><div class="v" id="st-ttft">–</div></div>
            <div class="va-stat"><div class="k">LLM load</div><div class="v" id="st-llmload">–</div></div>
            <div class="va-stat"><div class="k">Whisper load</div><div class="v" id="st-asrload">–</div></div>
            <div class="va-stat"><div class="k">Transcription</div><div class="v" id="st-asr">–</div></div>
            <div class="va-stat"><div class="k">Real-time factor</div><div class="v" id="st-rtf">–</div></div>
          </div>
          <canvas id="va-spark" width="736" height="104" title="Tokens per second for each reply"></canvas>
          <label>Benchmark</label>
          <div class="va-note">Runs the same 3 questions on the loaded model and records speed. Change model or GPU/CPU, reload, and run again to compare.</div>
          <canvas id="va-bench-chart" width="736" height="40"></canvas>
          <div class="va-btns">
            <button id="va-bench" class="va-btn sec">Run benchmark</button>
            <button id="va-bench-clear" class="va-btn sec">Clear results</button>
            <button id="va-csv" class="va-btn sec">Copy CSV</button>
          </div>
        </div>

        <div class="va-card">
          <h2><b>05</b> Keyboard</h2>
          <div class="va-keys">
            <kbd>Space</kbd><span>Hold to talk</span>
            <kbd>Enter</kbd><span>Send · Shift + Enter for a new line</span>
            <kbd>N</kbd><span>New chat</span>
            <kbd>/</kbd><span>Type a message</span>
            <kbd>Ctrl/⌘ K</kbd><span>Search chats</span>
            <kbd>Esc</kbd><span>Stop speaking or writing · close panels</span>
          </div>
        </div>

        <div class="va-card">
          <h2><b>06</b> Data</h2>
          <div class="va-btns"><button id="va-clear" class="va-btn sec">Delete all chats</button></div>
          <div class="va-note">Chats are stored only in this browser.</div>
        </div>
      </div>
      <div id="va-toast"></div>`;

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
    root.innerHTML = HTML;
    document.body.appendChild(root);

    LLMS.forEach((m, i) => $id("va-llm").add(new Option(m.name, i)));
    ASRS.forEach((m, i) => $id("va-asr").add(new Option(m.name, i)));

    // restore preferences
    const P = LS.get("va-prefs", {});
    if (P.llm !== undefined) $id("va-llm").value = P.llm; else $id("va-llm").value = "1";
    if (P.asr !== undefined) $id("va-asr").value = P.asr;
    if (P.persona) $id("va-persona").value = P.persona;
    $id("va-custom").value = P.custom || "";
    $id("va-custom-wrap").style.display = $id("va-persona").value === "custom" ? "" : "none";
    ["speak", "captions", "barge", "auto"].forEach((k) => { if (P[k] !== undefined) $id("va-" + k).checked = P[k]; });
    if (P.temp) { $id("va-temp").value = P.temp; $id("va-tempv").textContent = P.temp; }
    if (P.max) { $id("va-max").value = P.max; $id("va-maxv").textContent = P.max; }
    if (P.rate) { $id("va-rate").value = P.rate; $id("va-ratev").textContent = (+P.rate).toFixed(1) + "×"; }
    const savePrefs = () => LS.set("va-prefs", {
      llm: $id("va-llm").value, asr: $id("va-asr").value, persona: $id("va-persona").value, custom: $id("va-custom").value,
      speak: $id("va-speak").checked, captions: $id("va-captions").checked, barge: $id("va-barge").checked, auto: $id("va-auto").checked,
      temp: $id("va-temp").value, max: $id("va-max").value, rate: $id("va-rate").value, voice: $id("va-voice").value,
    });
    $id("va-settings").addEventListener("change", savePrefs);
    $id("va-settings").addEventListener("input", (e) => { if (e.target.type === "range" || e.target.id === "va-custom") savePrefs(); });

    // theme
    const setTheme = (light) => { root.classList.toggle("light", light); $id("va-theme").textContent = light ? "Dark" : "Light"; LS.set("va-theme", light ? "light" : "dark"); };
    setTheme(LS.get("va-theme", "dark") === "light");
    $id("va-theme").onclick = () => setTheme(!root.classList.contains("light"));

    // sidebar slide in / out (remembered)
    const narrow = () => window.innerWidth <= 900;
    const setSide = (show) => { root.classList.toggle("side-hidden", !show); if (!narrow()) LS.set("va-side", show ? 1 : 0); };
    setSide(narrow() ? false : LS.get("va-side", 1) !== 0);
    S.setSide = setSide; S.narrow = narrow;
    $id("va-collapse").onclick = () => setSide(false);
    $id("va-menu").onclick = () => setSide(true);

    // settings panel
    const openSet = (on) => { root.classList.toggle("set-open", on); if (on) { renderInfo(); updateStorage(); drawBench(); } };
    S.openSet = openSet;
    $id("va-open-settings").onclick = () => { if (narrow()) setSide(false); openSet(true); };
    $id("va-close-settings").onclick = () => openSet(false);
    $id("va-scrim").onclick = () => openSet(false);

    // mode tabs
    const tabs = $id("va-tabs");
    Object.keys(MODES).forEach((k) => {
      const b = document.createElement("button");
      b.className = "va-tab"; b.dataset.mode = k; b.textContent = MODES[k].label;
      b.onclick = () => setMode(k);
      tabs.appendChild(b);
    });

    // main controls
    $id("va-reload").onclick = () => { openSet(false); loadModels(); };
    $id("va-new").onclick = () => { if (narrow()) setSide(false); newChat(); };
    $id("va-send").onclick = () => sendText();
    $id("va-stop").onclick = stopAll;
    $id("va-mic").onclick = toggleRecording;
    $id("va-clear").onclick = () => { if (confirm("Delete all saved chats?")) { S.chats = []; saveChats(); newChat(); } };
    $id("va-csv").onclick = copyCSV;
    $id("va-bench").onclick = runBenchmark;
    $id("va-bench-clear").onclick = () => { S.bench = []; LS.set("va-bench", []); drawBench(); };
    $id("va-clear-cache").onclick = clearModelCache;
    $id("va-temp").oninput = (e) => ($id("va-tempv").textContent = e.target.value);
    $id("va-max").oninput = (e) => ($id("va-maxv").textContent = e.target.value);
    $id("va-rate").oninput = (e) => ($id("va-ratev").textContent = (+e.target.value).toFixed(1) + "×");
    $id("va-persona").onchange = () => {
      $id("va-custom-wrap").style.display = $id("va-persona").value === "custom" ? "" : "none";
      toast("Persona applies from your next message.");
    };
    $id("va-asr").onchange = () => toast("Press Load models to switch the speech model.");
    $id("va-llm").onchange = () => toast("Press Load models to switch the language model.");
    $id("va-conv").onchange = (e) => {
      S.conversation = e.target.checked;
      if (S.conversation) { $id("va-speak").checked = true; $id("va-auto").checked = true; toast("Hands-free: press the mic once. It listens again after each reply."); }
    };
    $id("va-speak").onchange = (e) => { if (!e.target.checked) speechSynthesis.cancel(); };
    $id("va-file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) { openSet(false); transcribeBlob(f, false); } };
    $id("va-preview").onclick = previewVoice;
    $id("va-voice").addEventListener("change", previewVoice);

    // export
    $id("va-export").onclick = (e) => { e.stopPropagation(); $id("va-export-menu").classList.toggle("on"); };
    document.addEventListener("click", () => $id("va-export-menu").classList.remove("on"));
    $id("va-export-txt").onclick = exportText;
    $id("va-export-pdf").onclick = exportPDF;

    // history search
    $id("va-search").addEventListener("input", renderHist);

    // text box
    const ta = $id("va-text");
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); } });
    ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 180) + "px"; });

    // keyboard shortcuts
    const typing = () => { const a = document.activeElement; return a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT" || a.isContentEditable); };
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setSide(true); $id("va-search").focus(); return; }
      if (e.key === "Escape") {
        if (root.classList.contains("set-open")) openSet(false);
        else if (S.busy || S.rec || speechSynthesis.speaking) stopAll();
        else if (narrow()) setSide(false);
        if (typing()) document.activeElement.blur();
        return;
      }
      if (typing() || e.metaKey || e.ctrlKey || e.altKey || root.classList.contains("set-open")) return;
      if (e.code === "Space" && !e.repeat && !$id("va-mic").disabled && !S.rec) { e.preventDefault(); startRecording(true); }
      else if (e.key === "n" || e.key === "N") { e.preventDefault(); newChat(); }
      else if (e.key === "/") { e.preventDefault(); if (!ta.disabled) ta.focus(); }
    });
    document.addEventListener("keyup", (e) => { if (e.code === "Space" && S.rec && S.rec.pushToTalk) { e.preventDefault(); stopRecording(); } });

    // offline badge: proof that it runs locally
    const net = () => root.classList.toggle("is-offline", !navigator.onLine);
    window.addEventListener("online", net); window.addEventListener("offline", () => { net(); toast("Offline. The assistant keeps working: the models run on this device."); }); net();

    // voices
    const fillVoices = () => {
      const sel = $id("va-voice"); const keep = sel.value || LS.get("va-prefs", {}).voice; sel.innerHTML = "";
      const silly = /^(Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Fred|Good News|Jester|Junior|Organ|Ralph|Superstar|Trinoids|Whisper|Wobble|Zarvox)\b/;
      const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang) && !silly.test(v.name));
      voices.forEach((v) => sel.add(new Option(v.name + " · " + v.lang + (v.localService ? "" : " (online)"), v.name)));
      if (keep && voices.some((v) => v.name === keep)) sel.value = keep;
      else { const local = voices.find((v) => v.localService); if (local) sel.value = local.name; }
    };
    if ("speechSynthesis" in window) { fillVoices(); speechSynthesis.onvoiceschanged = fillVoices; }

    // clicks inside the chat: suggestion chips, message actions, source links
    $id("va-chat").addEventListener("click", (e) => {
      const chip = e.target.closest(".va-chip");
      if (chip) { sendText(chip.dataset.q); return; }
      const act = e.target.closest("[data-act]");
      if (act) msgAction(act.dataset.act, act.closest(".va-msg"));
    });

    S.chats = LS.get("va-chats-v1", []);
    S.bench = LS.get("va-bench", []);
    setMode(LS.get("va-mode", "chat"), true);
    renderHist();
    renderWelcome();
    startOrb();
  }

  // ---------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms"; }
  function scrollDown() { const c = $id("va-chat"); c.scrollTop = c.scrollHeight; }
  let toastTimer = 0;
  function toast(text) {
    const t = $id("va-toast"); t.textContent = text; t.classList.add("on");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("on"), 3400);
  }

  // Minimal, safe Markdown: paragraphs, **bold**, *italic*, `code`, ``` blocks, - lists, 1. lists, # headings.
  function renderMarkdown(src) {
    const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<i>$2</i>");
    const lines = String(src).replace(/\r/g, "").split("\n");
    let html = "", para = [], list = null, items = [], code = null;
    const flushPara = () => { if (para.length) { html += "<p>" + para.map(inline).join("<br>") + "</p>"; para = []; } };
    const flushList = () => { if (list) { html += "<" + list + ">" + items.map((i) => "<li>" + inline(i) + "</li>").join("") + "</" + list + ">"; list = null; items = []; } };
    for (const line of lines) {
      if (/^\s*```/.test(line)) {
        if (code) { html += "<pre>" + esc(code.join("\n")) + "</pre>"; code = null; }
        else { flushPara(); flushList(); code = []; }
        continue;
      }
      if (code) { code.push(line); continue; }
      let m;
      if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { flushPara(); if (list !== "ul") { flushList(); list = "ul"; } items.push(m[1]); }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flushPara(); if (list !== "ol") { flushList(); list = "ol"; } items.push(m[1]); }
      else if ((m = line.match(/^\s*#{1,4}\s+(.*)$/))) { flushPara(); flushList(); html += "<h4>" + inline(m[1]) + "</h4>"; }
      else if (!line.trim()) { flushPara(); flushList(); }
      else { flushList(); para.push(line); }
    }
    flushPara(); flushList();
    if (code) html += "<pre>" + esc(code.join("\n")) + "</pre>";   // still streaming
    return html;
  }
  const plain = (s) => String(s).replace(/```[\s\S]*?```/g, " ").replace(/[*_`#>]/g, "").replace(/^\s*[-•]\s+/gm, "").trim();

  // ---------------------------------------------------------------------------------
  // Messages on screen
  // ---------------------------------------------------------------------------------

  function addMsg(role, html) {
    const w = $id("va-welcome"); if (w) w.remove();
    const d = document.createElement("div");
    d.className = "va-msg " + role;
    d.innerHTML = html;
    $id("va-chat").appendChild(d);
    $id("va-root").classList.toggle("has-msgs", S.history.length > 0);
    scrollDown();
    return d;
  }
  function sysMsg(html) { return addMsg("system", html); }

  function userEl(text) {
    return addMsg(text.length > 160 ? "user long" : "user", `<div class="body">${esc(text)}</div><div class="va-acts"><button class="va-link act-last" data-act="edit">Edit</button></div>`);
  }
  function assistantEl() {
    const el = addMsg("assistant", `<div class="body"></div><div class="va-src" style="display:none"></div>
      <div class="va-acts">
        <button class="va-link" data-act="copy">Copy</button>
        <button class="va-link" data-act="speak">Read aloud</button>
        <button class="va-link" data-act="simpler">Explain like I'm five</button>
        <button class="va-link act-last" data-act="retry">Retry</button>
      </div>`);
    return el;
  }
  function setSources(el, sources) {
    const box = el.querySelector(".va-src");
    if (!sources || !sources.length) { box.style.display = "none"; return; }
    box.style.display = "";
    box.innerHTML = "<span>From the notes:</span>" + sources.map((s) =>
      `<a href="${esc(s.src || NOTES_SRC)}" target="_blank" rel="noopener" title="${esc(s.text)}">${esc(s.title)}</a>`).join("");
  }
  function markLast() {
    document.querySelectorAll("#va-chat .va-msg.last").forEach((e) => e.classList.remove("last"));
    const us = document.querySelectorAll("#va-chat .va-msg.user:not(.caption)"), as = document.querySelectorAll("#va-chat .va-msg.assistant");
    if (us.length) us[us.length - 1].classList.add("last");
    if (as.length) as[as.length - 1].classList.add("last");
  }
  function renderHistoryMessages() {
    $id("va-chat").innerHTML = "";
    S.history.forEach((m) => {
      if (m.role === "user") userEl(m.display || m.content);
      else { const el = assistantEl(); el.querySelector(".body").innerHTML = renderMarkdown(m.content); setSources(el, m.sources); }
    });
    markLast();
    $id("va-root").classList.toggle("has-msgs", S.history.length > 0);
  }

  function msgAction(act, el) {
    if (!el) return;
    const isLastA = el.classList.contains("assistant") && el.classList.contains("last");
    const all = [...document.querySelectorAll("#va-chat .va-msg.user:not(.caption), #va-chat .va-msg.assistant")];
    const idx = all.indexOf(el);                      // same order as S.history
    const msg = S.history[idx];
    if (act === "copy" && msg) {
      navigator.clipboard.writeText(msg.content).then(() => toast("Copied."), () => toast("Clipboard blocked by the browser."));
    } else if (act === "speak" && msg) {
      speechSynthesis.cancel(); say(plain(msg.content));
    } else if (act === "simpler") {
      if (S.busy) return;
      sendText("Explain your last answer again as if I were five years old, in two or three very short, simple sentences.", { display: "Explain that like I'm five." });
    } else if (act === "retry" && isLastA) {
      if (S.busy) return;
      S.history.pop(); el.remove(); storeCurrent(); markLast();
      respond();
    } else if (act === "edit") {
      if (S.busy || S.rec) return;
      const ta = $id("va-text");
      ta.value = msg ? (msg.display || msg.content) : "";
      S.history = S.history.slice(0, idx);
      all.slice(idx).forEach((e) => e.remove());
      storeCurrent(); markLast();
      if (!S.history.length) renderWelcome();
      ta.focus(); ta.dispatchEvent(new Event("input"));
    }
  }

  // ---------------------------------------------------------------------------------
  // Modes and the welcome screen
  // ---------------------------------------------------------------------------------

  const PLACEHOLDERS = {
    chat: "Message, or hold Space to talk", tutor: "Ask about the notes", quiz: "Type your answer, or hold Space to say it",
    summarise: "Paste a long text to summarise, or hold Space and read it out",
  };

  function setMode(k, silent) {
    if (!MODES[k] || S.busy) return;
    const changed = S.mode !== k;
    S.mode = k; LS.set("va-mode", k);
    document.querySelectorAll(".va-tab").forEach((b) => b.classList.toggle("on", b.dataset.mode === k));
    $id("va-root").classList.toggle("mode-quiz", k === "quiz");
    if (S.llmReady) $id("va-text").placeholder = PLACEHOLDERS[k];
    if (changed && S.history.length && !silent) newChat();
    else if (!S.history.length) renderWelcome();
    updateScore();
  }

  function renderWelcome() {
    const loaded = S.llmReady && !S.loading, M = MODES[S.mode];
    const body = loaded
      ? `<div class="va-chips">${(CHIPS[S.mode] || []).map(([l, q]) => `<button class="va-chip" data-q="${esc(q)}">${esc(l)}</button>`).join("")}</div>`
      : S.loading
        ? `<div class="va-small">Downloading the models…</div><div class="va-bar"><div id="va-wbar"></div></div>`
        : `<button id="va-load" class="va-btn solid">Load models</button>`;
    $id("va-chat").innerHTML = `
      <div id="va-welcome">
        <h3>${loaded ? esc(M.title) : "An AI that lives in your browser."}</h3>
        <p>${loaded ? esc(M.hint)
                    : "Whisper (speech-to-text) and a small language model are downloaded to this page and run on your own device. No AI server is used."}</p>
        ${body}
      </div>`;
    const lb = $id("va-load"); if (lb) lb.onclick = loadModels;
    $id("va-root").classList.remove("has-msgs");
  }

  function updateScore() {
    const q = S.quiz;
    $id("va-score").textContent = "Score " + (+q.score.toFixed(1)) + " / " + q.total;
  }

  // ---------------------------------------------------------------------------------
  // Chat history (localStorage): new, open, rename, pin, delete, search
  // ---------------------------------------------------------------------------------

  function saveChats() { LS.set("va-chats-v1", S.chats.slice(0, 80)); renderHist(); }

  function renderHist() {
    const box = $id("va-hist"), q = $id("va-search").value.trim().toLowerCase();
    const match = (c) => !q || c.title.toLowerCase().includes(q) || c.messages.some((m) => (m.content || "").toLowerCase().includes(q));
    const list = S.chats.filter(match);
    box.innerHTML = "";
    if (!list.length) { box.innerHTML = `<div class="va-empty">${q ? "No chats match." : "No chats yet."}</div>`; return; }
    const groups = [["Pinned", list.filter((c) => c.pinned)], ["Recent", list.filter((c) => !c.pinned)]];
    groups.forEach(([name, items]) => {
      if (!items.length) return;
      const lab = document.createElement("div"); lab.className = "va-label"; lab.textContent = name; box.appendChild(lab);
      items.forEach((c) => {
        const row = document.createElement("div");
        row.className = "va-h" + (c.id === S.chatId ? " on" : "");
        row.innerHTML = `<span class="ttl"></span><span class="ha">
            <button class="va-link" data-a="pin" title="${c.pinned ? "Unpin" : "Pin"}">${c.pinned ? "Unpin" : "Pin"}</button>
            <button class="va-link" data-a="ren" title="Rename">Rename</button>
            <button class="va-link" data-a="del" title="Delete">×</button></span>`;
        row.querySelector(".ttl").textContent = (c.mode && c.mode !== "chat" ? MODES[c.mode].label + " · " : "") + c.title;
        row.onclick = (e) => {
          const a = e.target.closest("[data-a]");
          if (!a) return openChat(c.id);
          e.stopPropagation();
          if (a.dataset.a === "pin") { c.pinned = !c.pinned; saveChats(); }
          else if (a.dataset.a === "del") deleteChat(c.id);
          else if (a.dataset.a === "ren") renameChat(c, row);
        };
        box.appendChild(row);
      });
    });
  }

  function renameChat(c, row) {
    const ttl = row.querySelector(".ttl");
    const inp = document.createElement("input"); inp.value = c.title;
    ttl.replaceWith(inp); row.querySelector(".ha").style.display = "none";
    inp.focus(); inp.select();
    let done = false;
    const finish = (save) => { if (done) return; done = true; if (save && inp.value.trim()) c.title = inp.value.trim().slice(0, 80); saveChats(); };
    inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
    inp.onblur = () => finish(true);
    inp.onclick = (e) => e.stopPropagation();
  }

  function newChat() {
    if (S.busy) return;
    S.chatId = null; S.history = [];
    S.quiz = { score: 0, total: 0, item: null }; updateScore();
    renderWelcome(); renderHist();
  }

  function openChat(id) {
    if (S.busy || S.rec) return;
    const c = S.chats.find((x) => x.id === id); if (!c) return;
    S.chatId = id; S.history = c.messages.slice();
    if (c.mode && c.mode !== S.mode) setMode(c.mode, true);
    S.quiz = c.quiz ? Object.assign({}, c.quiz) : { score: 0, total: 0, item: null }; updateScore();
    renderHistoryMessages(); renderHist();
    if (S.narrow()) S.setSide(false);
  }

  function deleteChat(id) {
    S.chats = S.chats.filter((x) => x.id !== id);
    saveChats();
    if (S.chatId === id) newChat();
  }

  function storeCurrent() {
    let c = S.chats.find((x) => x.id === S.chatId);
    if (!S.history.length && !c) return;
    if (!c) {
      const first = (S.history.find((m) => m.role === "user") || { content: "New chat" });
      const t = first.display || first.content;
      c = { id: "c" + Date.now(), title: t.length > 46 ? t.slice(0, 44) + "…" : t, messages: [], mode: S.mode, pinned: false };
      S.chatId = c.id; S.chats.unshift(c);
    }
    c.messages = S.history.slice(); c.ts = Date.now(); c.mode = S.mode; c.quiz = Object.assign({}, S.quiz);
    S.chats = [c].concat(S.chats.filter((x) => x !== c));
    saveChats();
  }

  // ---------------------------------------------------------------------------------
  // Notes: embed once, then find the closest sections to a question (RAG)
  // ---------------------------------------------------------------------------------

  let ridSeq = 0;
  function embed(texts) {
    return new Promise((resolve, reject) => {
      const rid = ++ridSeq; S.embReq[rid] = { resolve, reject };
      S.worker.postMessage({ type: "embed", tjs: TJS_URL, rid, texts });
    });
  }
  function loadEmb() {
    if (S.embReady) return Promise.resolve();
    if (!S.embLoadP) {
      toast("Loading the notes search model (about 23 MB)…");
      S.embLoading = true;
      S.embLoadP = new Promise((resolve, reject) => { S.embLoaded = resolve; S.embFailed = reject; });
      S.worker.postMessage({ type: "loadEMB", tjs: TJS_URL, id: EMB.id });
    }
    return S.embLoadP;
  }
  async function ensureNotes() {
    await loadEmb();
    if (S.notes.length) return;
    const vecs = await embed(NOTES.map(([t, x]) => t + ". " + x));
    S.notes = NOTES.map(([title, text], i) => ({ title, text, src: NOTES_SRC, vec: vecs[i] }));
  }
  async function retrieve(query, k) {
    await ensureNotes();
    const [q] = await embed([query]);
    const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };   // vectors are normalised: dot = cosine
    return S.notes.map((n) => ({ n, score: dot(q, n.vec) })).sort((a, b) => b.score - a.score).slice(0, k);
  }
  function randomNote(not) {
    const pool = S.notes.filter((n) => n !== not);
    return pool[Math.floor(Math.random() * pool.length)];
  }

  // ---------------------------------------------------------------------------------
  // Asking and answering
  // ---------------------------------------------------------------------------------

  let current = null;   // the assistant message being streamed

  function generate(messages, opts) {
    return new Promise((resolve, reject) => {
      const rid = ++ridSeq;
      S.gen = { rid, onToken: opts.onToken, resolve, reject };
      S.worker.postMessage({ type: "generate", rid, messages, maxTokens: opts.maxTokens, temperature: opts.temperature });
    });
  }

  function personaPrompt() {
    const p = $id("va-persona").value;
    if (p === "custom") return ($id("va-custom").value.trim() || PERSONAS.assistant) + " Keep answers short because they will be read aloud.";
    return PERSONAS[p];
  }

  // Build the prompt for the current mode. Returns {messages, sources, after(text)}
  async function buildPrompt() {
    const last = S.history[S.history.length - 1];
    const temperature = parseFloat($id("va-temp").value), maxTokens = parseInt($id("va-max").value, 10);
    const content = (m) => ({ role: m.role, content: m.content });

    if (S.mode === "tutor") {
      const hits = (await retrieve(last.content, 3)).filter((h, i) => i === 0 || h.score > 0.25);
      const notes = hits.map((h, i) => `[${i + 1}] ${h.n.title}: ${h.n.text}`).join("\n\n");
      const sys = "You are a study helper. Answer the user's question using ONLY the notes below. " +
        "Keep it short: two to four sentences. If the notes do not contain the answer, say \"I can't find that in the notes.\"\n\nNOTES:\n" + notes;
      return { messages: [{ role: "system", content: sys }].concat(S.history.slice(-4).map(content)), temperature: Math.min(temperature, 0.3), maxTokens,
               sources: hits.map((h) => ({ title: h.n.title, text: h.n.text, src: h.n.src })) };
    }

    if (S.mode === "quiz") {
      await ensureNotes();
      const q = S.quiz, noteOf = (it) => S.notes.find((n) => n.title === it[0]) || { title: it[0], text: it[2], src: NOTES_SRC };
      const pick = (not) => { const pool = QUIZ.filter((x) => x !== not); return pool[Math.floor(Math.random() * pool.length)]; };
      const current = q.item ? QUIZ.find((x) => x[1] === q.item) : null;
      const starting = !current || /^\s*(start|new quiz|begin|next|skip)\b/i.test(last.content);
      if (starting) {
        const it = pick(current);
        return { static: "**Question:** " + it[1], sources: [{ title: noteOf(it).title, text: noteOf(it).text, src: NOTES_SRC }],
                 after: () => { q.item = it[1]; } };
      }
      // mark by meaning: cosine similarity of the two sentence embeddings
      const [va, vb] = await embed([last.content, current[2]]);
      let sim = 0; for (let i = 0; i < va.length; i++) sim += va[i] * vb[i];
      const verdict = sim >= 0.55 ? "Correct!" : sim >= 0.38 ? "Partly correct." : "Not quite.";
      q.total += 1; q.score += sim >= 0.55 ? 1 : sim >= 0.38 ? 0.5 : 0;
      const next = pick(current);
      const sys = "You are a friendly quiz master. The user's answer has already been marked. In one or two short sentences, explain the correct answer. Do not ask a question.";
      const user = "Question: " + current[1] + "\nCorrect answer: " + current[2] + "\nUser's answer: " + last.content + "\nMark: " + verdict;
      return { messages: [{ role: "system", content: sys }, { role: "user", content: user }], temperature: 0.2, maxTokens: 90,
               prefix: "**" + verdict + "** ", suffix: "\n\n*Meaning match: " + Math.max(0, Math.round(sim * 100)) + "%*\n\n**Next question:** " + next[1],
               sources: [{ title: noteOf(current).title, text: noteOf(current).text, src: NOTES_SRC }],
               after: () => { q.item = next[1]; updateScore(); } };
    }

    if (S.mode === "summarise") {
      const sys = "Summarise the user's text. Reply with three short bullet points, each starting with \"- \", then one final line starting with \"In short:\". Do not add anything else.";
      return { messages: [{ role: "system", content: sys }, { role: "user", content: last.content }], temperature: 0.2, maxTokens: Math.max(maxTokens, 220) };
    }

    // chat: persona + recent turns (small models have small context windows)
    return { messages: [{ role: "system", content: personaPrompt() }].concat(S.history.slice(-8).map(content)), temperature, maxTokens };
  }

  function sendText(text, opts) {
    opts = opts || {};
    if (S.busy || !S.llmReady) return;
    const ta = $id("va-text");
    const fromBox = text === undefined;
    text = (fromBox ? ta.value : text).trim();
    if (!text) return;
    if (fromBox) { ta.value = ""; ta.style.height = "auto"; }
    const entry = { role: "user", content: text };
    if (opts.display) entry.display = opts.display;
    S.history.push(entry);
    userEl(opts.display || text);
    storeCurrent(); markLast();
    respond();
  }

  async function respond() {
    S.busy = true; S.speakBuffer = "";
    speechSynthesis.cancel();
    setInputs(false);
    $id("va-send").style.display = "none"; $id("va-stop").style.display = "";
    const el = assistantEl(), body = el.querySelector(".body");
    body.innerHTML = '<span class="va-dots"><span></span><span></span><span></span></span>';
    current = { el, body, text: "" };
    markLast();
    try {
      const P = await buildPrompt();
      setSources(el, P.sources);
      let m = null;
      if (P.static) { current.text = P.static; speakStream(P.static + " "); }
      else {
        if (P.prefix) { current.text = P.prefix; speakStream(P.prefix); }
        m = await generate(P.messages, {
          maxTokens: P.maxTokens, temperature: P.temperature,
          onToken: (t) => { current.text += t; body.innerHTML = renderMarkdown(current.text) + '<span class="va-cursor"></span>'; scrollDown(); speakStream(t); },
        });
        if (P.suffix) { current.text = current.text.trimEnd() + P.suffix; speakStream(" " + P.suffix.replace(/\*Meaning match[^*]*\*/, "")); }
      }
      const text = current.text.trim() || (m ? m.text.trim() : "");
      body.innerHTML = renderMarkdown(text);
      if (P.after) P.after(text);
      const entry = { role: "assistant", content: text };
      if (P.sources) entry.sources = P.sources;
      S.history.push(entry);
      if (m) recordResult(m);
    } catch (err) {
      console.error("Reply failed:", err);
      el.remove();
      sysMsg("Something went wrong: " + esc(err.message || String(err)));
    }
    current = null;
    storeCurrent(); markLast();
    S.busy = false;
    $id("va-stop").style.display = "none"; $id("va-send").style.display = "";
    setInputs(true);
    flushSpeech();
    if (S.bargeRecord) { S.bargeRecord = false; startRecording(false); }
    else if (!S.conversation) $id("va-text").focus();
  }

  function recordResult(m) {
    const llm = LLMS[S.loadedLLM || 0];
    $id("st-ttft").textContent = fmtMs(m.ttft);
    $id("st-tps").textContent = m.tps.toFixed(1);
    S.tpsHist.push(m.tps); drawSpark();
    S.results.push({
      time: new Date().toISOString(), mode: S.mode, model: llm.id, device: S.device,
      llm_load_ms: Math.round(S.llmLoadMs), ttft_ms: Math.round(m.ttft), tokens: m.tokens,
      tokens_per_s: m.tps.toFixed(2), total_ms: Math.round(m.total),
      asr_model: ASRS[S.loadedASR || 0].id,
      asr_ms: S.lastAsr ? Math.round(S.lastAsr.ms) : "", audio_s: S.lastAsr ? S.lastAsr.seconds.toFixed(2) : "",
    });
    S.lastAsr = null;
  }

  function stopAll() {
    if (S.busy && S.gen) S.worker.postMessage({ type: "interrupt" });
    speechSynthesis.cancel(); S.speakBuffer = "";
    stopBarge();
    if (S.rec) stopRecording(true, true);
  }

  // ---------------------------------------------------------------------------------
  // GPU detection, worker, loading
  // ---------------------------------------------------------------------------------

  async function detectBackend() {
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) { S.device = "webgpu"; S.hasF16 = adapter.features.has("shader-f16"); }
      }
    } catch (e) { console.warn("WebGPU check failed:", e); }
    S.hasGPU = S.device === "webgpu";
    const devSel = $id("va-dev");
    $id("va-backend").textContent = S.hasGPU ? "· WebGPU found" : "· no WebGPU, CPU only";
    if (!S.hasGPU) devSel.options[0].disabled = true;
    const pref = LS.get("va-prefs", {}).dev;
    if (pref && (pref === "wasm" || S.hasGPU)) S.device = pref;
    devSel.value = S.device;
    devSel.onchange = () => {
      S.device = devSel.value; applyDevice();
      const P = LS.get("va-prefs", {}); P.dev = S.device; LS.set("va-prefs", P);
      toast("Press Load models to reload on the " + (S.device === "webgpu" ? "GPU." : "CPU."));
    };
    applyDevice();
  }

  function applyDevice() {
    const ok1B = S.device === "webgpu" && S.hasF16;      // the 1B model only ships a GPU fp16 (q4f16) version
    [...$id("va-llm").options].forEach((o) => { if (LLMS[o.value].gpuOnly) o.disabled = !ok1B; });
    if (!ok1B && LLMS[$id("va-llm").value].gpuOnly) $id("va-llm").value = "1";
    if (!S.hasGPU && !LS.get("va-prefs", {}).llm && $id("va-llm").value === "1") $id("va-llm").value = "0";
    if (S.device === "wasm") $id("va-captions").checked = false;   // live captions are too slow on the CPU
  }

  function startWorker() {
    const src = "(" + workerMain.toString() + ")();";
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    S.worker = new Worker(url, { type: "module" });
    S.worker.onmessage = onWorkerMessage;
    S.worker.onerror = (e) => { console.error("Worker error:", e.message || e); sysMsg("Worker error: " + esc(String(e.message || e))); };
  }

  // quantisation ("dtype") for each model + device
  function llmDtype() { return S.device === "webgpu" ? (S.hasF16 ? "q4f16" : "q4") : "q8"; }
  function asrDtype() {
    // NB: the fp16 merged decoder fails in ONNX Runtime ("Subgraph output (logits) is an outer scope value"),
    // so on the GPU we use the 4-bit decoder, as in the official Whisper WebGPU demo.
    return S.device === "webgpu" ? { encoder_model: "fp32", decoder_model_merged: "q4" } : { encoder_model: "q8", decoder_model_merged: "q8" };
  }
  const dtypeName = (d) => typeof d === "string" ? d : Object.values(d).join(" / ");

  let loadPending = 0;
  function loadModels() {
    if (S.busy) return;
    const li = +$id("va-llm").value, ai = +$id("va-asr").value;
    S.llmReady = S.asrReady = false;
    S.files = {}; S.sizes = {};
    S.info = S.info || {};
    S.pendingLoad = { li, ai };
    setInputs(false);
    $id("va-reload").disabled = true;
    loadPending = 2; S.loading = true; S.progressText = "";
    $id("va-status").textContent = "loading"; $id("va-status").className = "";
    if (!S.history.length) renderWelcome();
    S.worker.postMessage({ type: "loadLLM", tjs: TJS_URL, id: LLMS[li].id, device: S.device, dtype: llmDtype() });
  }

  function updateProgress(which, msg) {
    const key = which + ":" + msg.file;
    if ((msg.status === "progress" || msg.status === "done") && msg.total)
      S.files[key] = { loaded: msg.status === "done" ? msg.total : msg.loaded, total: msg.total, which };
    let loaded = 0, total = 0; const per = {};
    for (const k in S.files) { const f = S.files[k]; per[f.which] = (per[f.which] || 0) + f.total; if (f.which !== "emb") { loaded += f.loaded; total += f.total; } }
    Object.assign(S.sizes, per);
    if (which === "emb") return;
    if (total > 0) S.progressText = (loaded / 1e6).toFixed(0) + " / " + (total / 1e6).toFixed(0) + " MB";
    if (msg.status === "ready") S.progressText = "preparing " + (which === "llm" ? "language model" : "Whisper") + "…";
    const bar = $id("va-wbar");
    if (bar && total > 0) bar.style.width = (100 * loaded / total).toFixed(1) + "%";
  }

  function onWorkerMessage(e) {
    const m = e.data;
    switch (m.type) {
      case "progress": updateProgress(m.which, m); break;

      case "loaded":
        if (m.which === "emb") {
          S.embReady = true; S.embLoading = false;
          S.info.emb = { name: EMB.short, params: EMB.params, dtype: "q8", device: "wasm", ms: m.ms };
          if (S.embLoaded) S.embLoaded();
          break;
        }
        if (m.which === "llm") {
          const L = LLMS[S.pendingLoad.li];
          S.llmReady = true; S.llmLoadMs = m.ms; S.loadedLLM = S.pendingLoad.li;
          S.info.llm = { name: L.short, params: L.params, dtype: llmDtype(), device: S.device, ms: m.ms };
          $id("st-llmload").textContent = fmtMs(m.ms);
          S.worker.postMessage({ type: "loadASR", tjs: TJS_URL, id: ASRS[S.pendingLoad.ai].id, device: S.device, dtype: asrDtype() });
        } else {
          const A = ASRS[S.pendingLoad.ai];
          S.asrReady = true; S.asrLoadMs = m.ms; S.loadedASR = S.pendingLoad.ai;
          S.info.asr = { name: A.short + (A.multilingual ? " (translates to English)" : ""), params: A.params, dtype: dtypeName(asrDtype()), device: S.device, ms: m.ms };
          $id("st-asrload").textContent = fmtMs(m.ms);
        }
        if (--loadPending === 0) {
          S.loading = false;
          $id("va-reload").disabled = false; $id("va-reload").textContent = "Reload models";
          setInputs(true);
          $id("va-text").placeholder = PLACEHOLDERS[S.mode];
          $id("va-status").textContent = "ready"; $id("va-status").className = "ok";
          if (!S.history.length) renderWelcome();
          renderInfo();
        }
        break;

      case "embedded": {
        const r = S.embReq[m.rid]; delete S.embReq[m.rid];
        if (r) r.resolve(m.vectors);
        break;
      }

      case "transcript":
        if (m.partial) { onCaption(m.text); break; }
        $id("st-asr").textContent = fmtMs(m.ms) + " / " + m.seconds.toFixed(1) + " s";
        $id("st-rtf").textContent = (m.ms / 1000 / m.seconds).toFixed(2) + "×";
        S.lastAsr = { ms: m.ms, seconds: m.seconds };
        S.transcribing = false;
        removeCaption();
        if (!m.text || /^\[.*\]$|^\(.*\)$/.test(m.text)) {   // Whisper outputs e.g. "[BLANK_AUDIO]" for silence
          toast("Didn't catch that. Try again.");
          setInputs(true);
          if (S.conversation) setTimeout(() => startRecording(false), 600);
          return;
        }
        setInputs(true);
        sendText(m.text);
        break;

      case "token":
        if (S.gen && m.rid === S.gen.rid && S.gen.onToken) S.gen.onToken(m.text);
        break;

      case "done":
        if (S.gen && m.rid === S.gen.rid) { const g = S.gen; S.gen = null; g.resolve(m); }
        break;

      case "error":
        console.error("Model error (" + m.where + "):", m.message);
        if (m.where === "generate" && S.gen) { const g = S.gen; S.gen = null; g.reject(new Error(m.message)); break; }
        if (m.where === "embed" && S.embReq[m.rid]) { S.embReq[m.rid].reject(new Error(m.message)); delete S.embReq[m.rid]; break; }
        if (m.where === "loadEMB") { S.embLoading = false; S.embLoadP = null; if (S.embFailed) S.embFailed(new Error(m.message)); break; }
        sysMsg("Error during <b>" + esc(m.where) + "</b>: " + esc(m.message));
        S.transcribing = false; removeCaption();
        if (m.where === "loadLLM" || m.where === "loadASR") {
          S.loading = false; loadPending = 0; $id("va-reload").disabled = false;
          // if only Whisper failed, typing still works: say so, and let the user retry from Settings
          $id("va-status").textContent = S.llmReady ? "ready · no voice" : "not loaded";
          $id("va-status").className = S.llmReady ? "ok" : "";
          if (S.llmReady) { $id("va-text").placeholder = PLACEHOLDERS[S.mode]; toast("Speech model failed to download. Typing still works; retry with Load models in Settings."); }
          if (!S.history.length) renderWelcome();
          renderInfo();
        }
        setInputs(S.llmReady);
        break;
    }
  }

  function setInputs(on) {
    $id("va-text").disabled = !on;
    $id("va-send").disabled = !on;
    $id("va-mic").disabled = !(on && S.asrReady);
    $id("va-file").disabled = !(on && S.asrReady);
    $id("va-bench").disabled = !on;
  }

  // ---------------------------------------------------------------------------------
  // Settings: model info, storage, benchmark, CSV
  // ---------------------------------------------------------------------------------

  function renderInfo() {
    const I = S.info || {}, row = (label, x, which) => x
      ? `<dt>${label}</dt><dd>${esc(x.name)}<small>${esc(x.params)} parameters · ${esc(x.dtype)} · ${x.device === "webgpu" ? "GPU" : "CPU"}${S.sizes[which] ? " · " + (S.sizes[which] / 1e6).toFixed(0) + " MB" : ""}</small></dd>`
      : `<dt>${label}</dt><dd><small>not loaded</small></dd>`;
    $id("va-info").innerHTML = row("Language", I.llm, "llm") + row("Speech", I.asr, "asr") + row("Notes search", I.emb, "emb");
  }

  async function updateStorage() {
    try {
      const est = await navigator.storage.estimate();
      $id("va-storage").textContent = "This site is using " + (est.usage / 1e6).toFixed(0) + " MB of browser storage (mostly cached model files).";
    } catch (e) { $id("va-storage").textContent = ""; }
  }

  async function clearModelCache() {
    if (!("caches" in window)) { toast("This browser has no cache storage."); return; }
    if (!confirm("Delete the downloaded model files? They will download again next time you load.")) return;
    const keys = await caches.keys();
    let n = 0;
    for (const k of keys) if (/transformers/i.test(k)) { await caches.delete(k); n++; }
    toast(n ? "Downloaded models cleared. Models already loaded keep working until you reload." : "No downloaded models found.");
    updateStorage();
  }

  const BENCH_PROMPTS = [
    "What is a neural network? Answer in two sentences.",
    "Give three uses of artificial intelligence.",
    "Explain what a GPU is to a beginner.",
  ];
  async function runBenchmark() {
    if (!S.llmReady || S.busy) { toast("Load the models first."); return; }
    S.busy = true; setInputs(false); $id("va-bench").disabled = true;
    let ttft = 0, tps = 0;
    try {
      for (let i = 0; i < BENCH_PROMPTS.length; i++) {
        S.benchText = (i + 1) + " / " + BENCH_PROMPTS.length;
        const m = await generate([{ role: "user", content: BENCH_PROMPTS[i] }], { maxTokens: 64, temperature: 0 });
        ttft += m.ttft; tps += m.tps;
      }
      const n = BENCH_PROMPTS.length, L = LLMS[S.loadedLLM];
      S.bench.push({ model: L.short, device: S.device === "webgpu" ? "GPU" : "CPU", tps: tps / n, ttft: ttft / n, at: new Date().toISOString() });
      S.bench = S.bench.slice(-8); LS.set("va-bench", S.bench);
      toast(L.short + ": " + (tps / n).toFixed(1) + " tokens per second.");
    } catch (err) { console.error("Benchmark failed:", err); toast("Benchmark failed: " + err.message); }
    S.benchText = ""; S.busy = false; setInputs(true); drawBench();
  }

  function cssVar(name) { return getComputedStyle($id("va-root")).getPropertyValue(name).trim() || "#fff"; }

  function drawBench() {
    const cv = $id("va-bench-chart"), rows = S.bench;
    const rowH = 46, W = cv.width, H = Math.max(40, rows.length * rowH + 8);
    cv.height = H; cv.style.height = H / 2 + "px";
    const g = cv.getContext("2d"); g.clearRect(0, 0, W, H);
    const strong = cssVar("--strong"), meta = cssVar("--meta");
    if (!rows.length) { g.fillStyle = meta; g.font = "22px 'Space Grotesk', sans-serif"; g.fillText("No benchmark runs yet.", 0, 28); return; }
    const max = Math.max(...rows.map((r) => r.tps)) * 1.1;
    rows.forEach((r, i) => {
      const y = i * rowH + 4;
      g.fillStyle = meta; g.font = "20px 'Space Grotesk', sans-serif";
      g.fillText((r.model + " · " + r.device).toUpperCase(), 0, y + 18);
      const w = Math.max(4, (r.tps / max) * (W - 150));
      g.fillStyle = strong; g.fillRect(0, y + 26, w, 10);
      g.font = "bold 22px 'Space Grotesk', sans-serif";
      g.fillText(r.tps.toFixed(1) + " tok/s", w + 12, y + 37);
    });
  }

  function drawSpark() {
    const cv = $id("va-spark"), g = cv.getContext("2d"), W = cv.width, H = cv.height;
    const data = S.tpsHist.slice(-24);
    g.clearRect(0, 0, W, H);
    if (!data.length) return;
    const max = Math.max(...data) * 1.15, bw = W / 24;
    g.fillStyle = cssVar("--strong");
    data.forEach((v, i) => { const h = Math.max(3, (v / max) * (H - 6)); g.fillRect(i * bw + 2, H - h, bw - 4, h); });
  }

  function copyCSV() {
    if (!S.results.length) { toast("No results yet. Ask a few questions first."); return; }
    const cols = Object.keys(S.results[0]);
    const csv = [cols.join(",")].concat(S.results.map((r) => cols.map((c) => r[c]).join(","))).join("\n");
    navigator.clipboard.writeText(csv).then(
      () => toast(`Copied ${S.results.length} rows of results to the clipboard.`),
      () => { console.log(csv); toast("Clipboard blocked. Results printed to the console."); });
  }

  // ---------------------------------------------------------------------------------
  // Export the current chat as text or PDF (PDF uses the browser's own "Save as PDF")
  // ---------------------------------------------------------------------------------

  function chatTitle() { const c = S.chats.find((x) => x.id === S.chatId); return c ? c.title : "Chat"; }
  function exportText() {
    if (!S.history.length) return;
    const lines = [chatTitle(), "Offline Voice Assistant · " + new Date().toLocaleString(), ""];
    S.history.forEach((m) => {
      lines.push((m.role === "user" ? "You: " + (m.display || m.content) : "Assistant: " + plain(m.content)));
      if (m.sources && m.sources.length) lines.push("  (from the notes: " + m.sources.map((s) => s.title).join(", ") + ")");
      lines.push("");
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    a.download = chatTitle().replace(/[^\w\- ]+/g, "").trim().slice(0, 40) + ".txt";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function exportPDF() {
    if (!S.history.length) return;
    const msgs = S.history.map((m) => `<div class="m ${m.role}"><div class="who">${m.role === "user" ? "You" : "Assistant"}</div>
      <div>${m.role === "user" ? esc(m.display || m.content) : renderMarkdown(m.content)}</div>
      ${m.sources && m.sources.length ? `<div class="src">From the notes: ${m.sources.map((s) => esc(s.title)).join(", ")}</div>` : ""}</div>`).join("");
    const doc = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${esc(chatTitle())}</title><style>
      body{font-family:"Space Grotesk",Helvetica,Arial,sans-serif;color:#111;margin:40px;line-height:1.35}
      h1{font-size:28px;text-transform:uppercase;letter-spacing:-.5px;margin:0 0 4px}.sub{color:#777;font-size:12px;margin-bottom:28px}
      .m{margin:0 0 18px;page-break-inside:avoid}.who{font-size:10px;text-transform:uppercase;color:#777;margin-bottom:4px}
      .user div:nth-child(2){font-size:18px;font-weight:600}.src{font-size:11px;color:#777;margin-top:4px}
      pre{background:#f3f3f3;padding:8px;white-space:pre-wrap}p{margin:0 0 .6em}</style></head>
      <body><h1>${esc(chatTitle())}</h1><div class="sub">Offline Voice Assistant · ${esc(new Date().toLocaleString())}</div>${msgs}</body></html>`;
    const f = document.createElement("iframe");
    f.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0";
    document.body.appendChild(f);
    f.onload = () => { f.contentWindow.focus(); f.contentWindow.print(); setTimeout(() => f.remove(), 60000); };
    f.srcdoc = doc;
    toast("Choose “Save as PDF” in the print window.");
  }

  // ---------------------------------------------------------------------------------
  // Text-to-speech: speak each sentence as soon as it is complete (lower latency)
  // ---------------------------------------------------------------------------------

  function speakStream(chunk) {
    if (!$id("va-speak").checked || !("speechSynthesis" in window)) return;
    S.speakBuffer += chunk;
    const match = S.speakBuffer.match(/^([\s\S]*?[.!?…:\n])(\s+)([\s\S]*)$/);
    if (match && match[1].trim().length > 2) { say(plain(match[1])); S.speakBuffer = match[3]; }
  }
  function flushSpeech() {
    if ($id("va-speak").checked && S.speakBuffer.trim()) say(plain(S.speakBuffer));
    S.speakBuffer = "";
    if (S.conversation) {
      const wait = () => {
        if (speechSynthesis.speaking || speechSynthesis.pending) return setTimeout(wait, 250);
        if (S.conversation && !S.rec && !S.busy) startRecording(false);
      };
      setTimeout(wait, 400);
    }
  }
  function say(text, opts) {
    const clean = String(text).trim();
    if (!clean || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(clean);
    const v = speechSynthesis.getVoices().find((x) => x.name === $id("va-voice").value);
    if (v) u.voice = v;
    u.rate = parseFloat($id("va-rate").value) || 1;
    if (!(opts && opts.noBarge)) u.onstart = startBarge;
    speechSynthesis.speak(u);
  }
  function previewVoice() {
    speechSynthesis.cancel();
    const v = $id("va-voice").value.split(" ")[0] || "your assistant";
    say("Hi, I'm " + v + ". This is how I sound when I read your answers.", { noBarge: true });
  }

  // Interrupt by speaking: while a reply is read aloud, listen for the user's voice.
  async function startBarge() {
    if (S.barge || !$id("va-barge").checked || !S.asrReady || S.rec) return;
    const b = S.barge = { on: true };
    try {
      b.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false } });
    } catch (e) { S.barge = null; return; }
    if (!b.on) { b.stream.getTracks().forEach((t) => t.stop()); return; }
    b.ctx = new AudioContext();
    const an = b.ctx.createAnalyser(); an.fftSize = 1024;
    b.ctx.createMediaStreamSource(b.stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    let loud = 0, last = performance.now(); const t0 = last;
    const tick = () => {
      if (!b.on) return;
      const now = performance.now(), dt = now - last; last = now;
      if (!speechSynthesis.speaking && !speechSynthesis.pending && !S.busy) return stopBarge();
      an.getFloatTimeDomainData(buf);
      let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
      const rms = Math.sqrt(s / buf.length);
      loud = rms > 0.08 ? loud + dt : Math.max(0, loud - dt * 2);
      if (now - t0 > 500 && loud > 280) {                // ~0.3 s of clear speech: the user is talking
        stopBarge();
        speechSynthesis.cancel(); S.speakBuffer = "";
        if (S.busy) { S.bargeRecord = true; S.worker.postMessage({ type: "interrupt" }); }
        else startRecording(false);
        return;
      }
      b.raf = requestAnimationFrame(tick);
    };
    tick();
  }
  function stopBarge() {
    const b = S.barge; if (!b) return;
    S.barge = null; b.on = false;
    cancelAnimationFrame(b.raf);
    if (b.stream) b.stream.getTracks().forEach((t) => t.stop());
    if (b.ctx) b.ctx.close();
  }

  // ---------------------------------------------------------------------------------
  // Recording from the microphone -> 16 kHz mono Float32Array for Whisper (+ live captions)
  // ---------------------------------------------------------------------------------

  function toggleRecording() { if (S.rec) stopRecording(); else startRecording(false); }

  async function startRecording(pushToTalk) {
    if (S.rec || S.busy || !S.asrReady) return;
    speechSynthesis.cancel(); stopBarge();
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

    const ctx = new AudioContext();
    const an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    const started = performance.now();
    let heardSpeech = false, lastLoud = performance.now();

    const rec = { recorder, stream, ctx, chunks, pushToTalk, raf: 0, capTimer: 0 };
    S.rec = rec; S.capBusy = false;

    // live captions: every 1.4 s, transcribe everything recorded so far
    if ($id("va-captions").checked) {
      rec.capTimer = setInterval(async () => {
        if (S.capBusy || !chunks.length || S.rec !== rec) return;
        S.capBusy = true;
        try {
          const audio = await blobTo16k(new Blob(chunks, { type: recorder.mimeType }));
          if (audio.length < 16000 * 0.6 || S.rec !== rec) { S.capBusy = false; return; }
          S.worker.postMessage({ type: "transcribe", partial: true, audio, multilingual: ASRS[S.loadedASR].multilingual }, [audio.buffer]);
        } catch (e) { S.capBusy = false; }   // a partial recording can fail to decode: just try again next time
      }, 1400);
    }

    const tick = () => {
      if (S.rec !== rec) return;
      an.getFloatTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      S.level = rms;
      const now = performance.now();
      if (rms > 0.02) { heardSpeech = true; lastLoud = now; }
      const auto = $id("va-auto").checked && !pushToTalk;
      if (auto && heardSpeech && now - lastLoud > 1300) return stopRecording();      // 1.3 s of silence after speech
      if (auto && !heardSpeech && now - started > 8000) return stopRecording(true);  // nobody spoke
      if (now - started > 30000) return stopRecording();                             // Whisper window = 30 s
      rec.raf = requestAnimationFrame(tick);
    };

    recorder.onstop = () => {
      cancelAnimationFrame(rec.raf); clearInterval(rec.capTimer);
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      if (rec.discard) { removeCaption(); setInputs(true); return; }
      transcribeBlob(new Blob(chunks, { type: recorder.mimeType }), true);
    };

    recorder.start(500);   // emit data every 0.5 s, so live captions have audio to work with
    $id("va-mic").classList.add("rec");
    $id("va-text").disabled = true; $id("va-send").disabled = true;
    tick();
  }

  function stopRecording(discard, quiet) {
    const rec = S.rec;
    if (!rec) return;
    S.rec = null; S.level = 0;
    rec.discard = !!discard;
    if (discard && !quiet) toast("No speech heard. Stopped listening.");
    $id("va-mic").classList.remove("rec");
    $id("va-mic").disabled = true;
    if (rec.recorder.state !== "inactive") rec.recorder.stop();
  }

  function onCaption(text) {
    S.capBusy = false;
    if (!S.rec && !S.transcribing) return;
    if (!text || /^\[.*\]$|^\(.*\)$/.test(text)) return;
    if (!S.caption || !S.caption.isConnected) S.caption = addMsg("user caption", `<div class="body"></div>`);
    S.caption.querySelector(".body").textContent = text + " …";
    scrollDown();
  }
  function removeCaption() { if (S.caption) { S.caption.remove(); S.caption = null; } }

  async function transcribeBlob(blob, fromMic) {
    if (S.busy || !S.asrReady) return;
    setInputs(false);
    try {
      let audio = await blobTo16k(blob);
      if (audio.length < 16000 * 0.3) { toast("Recording too short."); removeCaption(); setInputs(true); return; }
      if (audio.length > 16000 * 30) { audio = audio.slice(0, 16000 * 30); toast("Audio longer than 30 s. Using the first 30 s."); }
      S.transcribing = true;
      S.worker.postMessage({ type: "transcribe", audio, multilingual: ASRS[S.loadedASR].multilingual }, [audio.buffer]);
    } catch (err) {
      console.error("Audio decode error:", err);
      sysMsg("Could not decode the audio: " + esc(String(err.message || err)));
      removeCaption(); setInputs(true);
    }
  }

  // Decode compressed audio and resample to 16 kHz mono (what Whisper expects)
  async function blobTo16k(blob) {
    const arr = await blob.arrayBuffer();
    const ctx = new AudioContext({ sampleRate: 16000 });   // the browser resamples for us
    try {
      const decoded = await ctx.decodeAudioData(arr);
      if (decoded.numberOfChannels === 1) return new Float32Array(decoded.getChannelData(0));
      const a = decoded.getChannelData(0), b = decoded.getChannelData(1);
      const out = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
      return out;
    } finally { ctx.close(); }
  }

  // ---------------------------------------------------------------------------------
  // Background: a WebGL shader showing one floating object that reacts to what the AI is doing.
  // Like the "canvas webgl" example World on Ancient Brain, the JS builds a GLSL program, compiles it,
  // and the GPU runs it for every pixel.
  // ---------------------------------------------------------------------------------

  const STATES = {
    off:          { label: "Ready to load" },
    loading:      { label: "Downloading" },
    idle:         { label: "Ready" },
    listening:    { label: "Listening" },
    transcribing: { label: "Transcribing" },
    thinking:     { label: "Thinking" },
    writing:      { label: "Writing" },
    speaking:     { label: "Speaking" },
    bench:        { label: "Benchmarking" },
  };

  function currentState() {
    if (S.rec) return "listening";
    if (S.transcribing) return "transcribing";
    if (S.benchText) return "bench";
    if (S.busy) return current && current.text ? "writing" : "thinking";
    if ("speechSynthesis" in window && speechSynthesis.speaking) return "speaking";
    if (S.loading) return "loading";
    return S.llmReady ? "idle" : "off";
  }

  // An obsidian liquid sphere, drawn by a raymarching fragment shader (the GPU traces a ray
  // for every pixel against a signed distance function). Near-black glass that only shows itself
  // through blue / violet / amber reflections at the rim. Light theme: pearl-white glass.
  // Technique: raymarching signed distance fields, Inigo Quilez https://iquilezles.org/articles/distfunctions/
  const FRAG = `
    precision highp float;
    uniform vec2 r; uniform vec2 c; uniform float t; uniform float e; uniform float p; uniform float d; uniform float lt;
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
      vec3 bg = vec3(lt);
      vec3 col = bg;
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
        float hl = pow(max(dot(rf, normalize(vec3(-0.4, 0.7, 0.6))), 0.0), 80.0);
        vec3 dark  = vec3(0.008) + env(rf) * (0.07 + 1.15*fres) * (0.8 + 0.6*e) + vec3(hl) * 0.9;
        vec3 light = vec3(0.93) - (vec3(1.0) - env(rf)) * (0.10 + 0.45*fres) * (0.8 + 0.5*e) + vec3(hl) * 0.4;
        vec3 m = mix(dark, light, lt);
        col = mix(m, bg, (lt > 0.5 ? 0.55 : 0.7) * d);
      }
      if (p > 0.0) {                                         // download progress ring
        float rr = length(uv), ang = atan(uv.x, uv.y) / 6.2832 + 0.5;
        float ring = smoothstep(0.016, 0.0, abs(rr - 0.86));
        col = mix(col, vec3(1.0 - lt), ring * (ang < p ? 0.95 : 0.15));
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
        ["r", "c", "t", "e", "p", "d", "lt"].forEach((k) => (U[k] = gl.getUniformLocation(prog, k)));
      } catch (err) { console.warn("WebGL sphere disabled:", err); gl = null; }
    }

    let shown = null, energy = 0, T = 0, last = performance.now(), dim = 0, cx = -1, cy = -1, lt = 0;

    function frame(now) {
      if (!document.body.contains(cv)) return;           // UI was rebuilt
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      const st = currentState(), def = STATES[st];

      if (st !== shown) { shown = st; $id("va-state").innerHTML = def.label + "<small></small>"; }
      const sm = $id("va-state").lastChild;
      const extra = st === "loading" ? S.progressText : st === "bench" ? S.benchText : "";
      if (sm && sm.textContent !== extra) sm.textContent = extra;

      let target = 0.0;
      if (st === "listening") target = 0.25 + Math.min(1, S.level * 14);
      else if (st === "thinking" || st === "transcribing" || st === "bench") target = 0.55;
      else if (st === "writing") target = 0.7;
      else if (st === "speaking") target = 0.4 + 0.35 * Math.abs(Math.sin(now / 170));
      else if (st === "loading") target = 0.2;
      energy += (target - energy) * Math.min(1, dt * 2.5);
      T += dt * (0.6 + energy * 1.6);
      dim += ((S.history.length ? 1 : 0) - dim) * Math.min(1, dt * 1.5);
      lt += (($id("va-root").classList.contains("light") ? 1 : 0) - lt) * Math.min(1, dt * 4);

      if (gl) {
        const scale = Math.min(window.devicePixelRatio || 1, 2) * 0.5;   // half resolution keeps the GPU free for the models
        const w = Math.round(cv.clientWidth * scale), h = Math.round(cv.clientHeight * scale);
        if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; gl.viewport(0, 0, w, h); }
        // the sphere overlaps the headline; while chatting it drifts to the far right and dims
        const main = $id("va-main").getBoundingClientRect(), rootR = cv.getBoundingClientRect();
        const tx = (main.left - rootR.left + main.width * (0.66 + 0.27 * dim)) * scale;
        const ty = (rootR.height - (main.top - rootR.top + main.height * (0.5 - 0.14 * dim))) * scale;
        if (cx < 0) { cx = tx; cy = ty; }
        cx += (tx - cx) * Math.min(1, dt * 3); cy += (ty - cy) * Math.min(1, dt * 3);
        gl.uniform2f(U.r, w, h); gl.uniform2f(U.c, cx, cy); gl.uniform1f(U.t, T); gl.uniform1f(U.e, energy); gl.uniform1f(U.d, dim);
        gl.uniform1f(U.p, st === "loading" ? Math.max(0.001, loadFraction()) : 0); gl.uniform1f(U.lt, lt);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function loadFraction() {
    let loaded = 0, total = 0;
    for (const k in S.files) { if (S.files[k].which === "emb") continue; loaded += S.files[k].loaded; total += S.files[k].total; }
    return total ? loaded / total : 0;
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
    AB.drawRunControls = false;                 // Running on Ancient Brain
    AB.world.newRun = function () { start(); };
  } else {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();   // stand-alone (local testing)
  }
})();
