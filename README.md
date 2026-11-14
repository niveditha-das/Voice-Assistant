# Offline Voice Assistant

A voice assistant that runs **entirely in your browser**. Speech recognition, the language model and the
search over your notes all run on your own device. There is no AI server, no account and no API key, and once
the models have loaded you can switch off the Wi-Fi and it keeps working.

## What it does

- **Chat**: talk or type; it answers out loud. Speak in English, or switch to Whisper base to speak almost any language.
- **Learn**: upload your own PDFs or text notes and ask questions about them. Answers come only from your notes and show the page they used.
- **Quiz**: it writes questions from your notes and marks your answers by meaning, not exact words.
- **Focus mode** keeps the background almost still; light and dark themes.
- A particle animation reacts to what the assistant is doing and spreads out behind the conversation.

## How it works

| Part | Model / tool |
| --- | --- |
| Speech to text | Whisper tiny.en or base (OpenAI), in a Web Worker |
| Replies | SmolLM2 135M / 360M, Qwen2.5 0.5B or Llama 3.2 1B |
| Searching notes, marking answers | all-MiniLM-L6-v2 sentence embeddings (cosine similarity) |
| Running the models | Transformers.js on WebGPU, falling back to WASM |
| Reading PDFs | PDF.js |
| Speaking | the browser's own speechSynthesis voices |

Everything you type, upload or save stays in your browser (localStorage).

## Run it locally

```bash
python3 -m http.server 8765
```

Then open <http://localhost:8765> in Chrome, press **Load models** and wait for the first download
(about 150–500 MB depending on the model; it is cached for next time). Allow the microphone to talk to it.

- `world.js` is the whole app.
- `styles/` holds earlier design versions: open `http://localhost:8765/?style=obsidian`, for example, or see them all in `gallery.html`.
- `test/` has sample notes and audio clips for trying uploads and speech.

## Credits

All libraries, models, fonts and design references are credited with links at the top of [`world.js`](world.js).

Built by Niveditha Das, with help from Claude (Anthropic).
