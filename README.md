# parley-post-train

Tools for post-training small Qwen models on [Parley](https://github.com/Metta-AI/cogame-parley),
a five-player talk-and-shoot party game from Softmax's Coworld platform.

Step 1 (this commit): watch one game where all five seats are Qwen3.5-4B, with every model
input and output shown live.

## How it fits together

```
Parley game (Docker, Mac)  --Anthropic Messages request-->  recorder.py (Mac)
recorder.py  --OpenAI chat request-->  vLLM serving Qwen on a Modal GPU (modal_qwen.py)
recorder.py  --reply-->  game; and every request/reply --> live page (viewer.html)
```

- `modal_qwen.py` — serves the model with vLLM on one Modal GPU. Starts on the first request and
  shuts down after 10 idle minutes. The API key is read from a local `.env` (not in Git).
- `recorder.py` — sits between the game and the GPU. Forwards each decision, saves the exact
  request and reply to the run folder, and streams them to the live page at `http://<host>:8790/`.
- `viewer.html` — left: Parley's own public table view. Right: every model call, with the system
  message, user message (including that seat's secret cards), the exact chat-template text the
  model reads, and the JSON it wrote.
- `run_game.py` — wakes the GPU, starts the game container, connects five players that all use
  Parley's default operator prompt, and waits for the final scores.

## Settings used for game 1

Qwen3.5-4B, thinking off, JSON-only decoding (vLLM `response_format: json_object`), temperature
0.7, top_p 0.8, top_k 20, presence_penalty 1.5 (Qwen's recommended non-thinking settings), 300
output tokens max. Parley pinned at commit `b46e7a8`; ordinary seed-sampled table. Pacing changed
for a watched local game: 2.5 s turn delay, 90 s model deadline, 40 min episode budget.

## Run

```bash
git clone https://github.com/Metta-AI/cogame-parley upstream && (cd upstream && git checkout b46e7a8)
docker build --build-arg PARLEY_SOURCE_REVISION=$(git -C upstream rev-parse HEAD) \
  --build-arg PARLEY_GAME_VERSION=0.4.0+local -t parley-local:b46e7a8 upstream
python3 -c "import secrets; print('VLLM_API_KEY=' + secrets.token_urlsafe(32))" > .env
modal deploy modal_qwen.py
uv venv -p 3.12 .venv && uv pip install -p .venv aiohttp websockets python-dotenv
.venv/bin/python recorder.py --modal-url <modal url> --run-dir runs/game1 &
.venv/bin/python run_game.py --modal-url <modal url> --run-dir runs/game1
```

Run folders (`runs/`) hold private game data and are not committed.
