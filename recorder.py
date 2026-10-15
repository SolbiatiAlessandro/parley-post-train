"""Recorder between the Parley game and Qwen on Modal, plus the live page.

The game sends every decision as an Anthropic Messages request to
POST /v1/messages. This server forwards it to vLLM on Modal (OpenAI chat
format), returns the reply in the format the game expects, saves the exact
request and reply under the run directory, and streams both to the live
page at /.

Usage: .venv/bin/python recorder.py --modal-url URL --run-dir runs/<id>
"""

import argparse
import asyncio
import ipaddress
import json
import re
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import dotenv_values

HERE = Path(__file__).parent
SERVED_NAME = "qwen3.5-4b"
HF_ID = "Qwen/Qwen3.5-4B"
GPU = "L40S"
GPU_DOLLARS_PER_HOUR = 0.000542 * 3600  # modal.com/pricing, 2026-10-14
THINKING = False
# Qwen's recommended non-thinking settings from the model card.
# Temperature and max_tokens come from the game's own request.
SAMPLING = {"top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
# Only the game container (via Docker's host gateway) and this machine may
# spend GPU time through /v1/messages; the page itself is open on the LAN.
TRUSTED = [ipaddress.ip_network(n) for n in
           ("127.0.0.0/8", "::1/128", "192.168.65.0/24", "172.16.0.0/12")]


def trusted(request):
    try:
        address = ipaddress.ip_address(request.remote)
    except ValueError:
        return False
    return any(address in net for net in TRUSTED)


def describe(system, user):
    seat = re.match(r"You are (.+?), a cog", system)
    if "You are IT." in user:
        kind = "IT — must shoot or pass"
    elif "You are OUT this round" in user:
        kind = "out — one line of talk"
    else:
        kind = "reaction — one line of talk"
    cards = re.search(r"FRIEND = (\S+) .*?ENEMY = (\S+) ", user)
    return {
        "seat": seat.group(1) if seat else "?",
        "kind": kind,
        "friend": cards.group(1) if cards else None,
        "enemy": cards.group(2) if cards else None,
    }


class Recorder:
    def __init__(self, modal_url, key, run_dir, parley_port):
        self.base = modal_url.rstrip("/")
        self.auth = {"Authorization": f"Bearer {key}"}
        self.run_dir = Path(run_dir)
        self.calls = []
        self.queues = set()
        self.session = None
        self.meta = {
            "model": HF_ID,
            "gpu": GPU,
            "gpu_dollars_per_hour": GPU_DOLLARS_PER_HOUR,
            "thinking": THINKING,
            "sampling": SAMPLING,
            "decoding": "JSON-only (vLLM response_format json_object)",
            "parley_port": parley_port,
            "status": "recorder started",
            "gpu_started": None,
            "game": None,
        }

    def publish(self, kind, payload):
        data = json.dumps({"type": kind, kind: payload})
        for queue in list(self.queues):
            queue.put_nowait(data)

    def save(self, call):
        with (self.run_dir / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(call) + "\n")

    async def start(self, _app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))

    async def stop(self, _app):
        await self.session.close()

    async def messages(self, request):
        if not trusted(request):
            return web.json_response({"error": "forbidden"}, status=403)
        body = json.loads(await request.read())
        system = body.get("system", "")
        user = body["messages"][0]["content"]
        gpu_request = {
            "model": SERVED_NAME,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": body["max_tokens"],
            "temperature": body["temperature"],
            **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": THINKING},
            "response_format": {"type": "json_object"},
        }
        call = {
            "n": len(self.calls) + 1,
            "slot": int(request.headers.get("x-coworld-player-slot", "-1")),
            **describe(system, user),
            "status": "waiting for Qwen",
            "started": time.time(),
            "game_request": body,
            "gpu_request": gpu_request,
        }
        self.calls.append(call)
        self.publish("call", call)
        began = time.monotonic()
        try:
            async with self.session.post(self.base + "/v1/chat/completions",
                                         json=gpu_request, headers=self.auth) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"GPU server returned {response.status}: {text[:400]}")
                gpu_response = json.loads(text)
        except Exception as error:  # the game sees the failure; no retry, no substitute
            call.update(status="error", error=str(error),
                        latency_ms=round((time.monotonic() - began) * 1000))
            self.publish("call", call)
            self.save(call)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(error)}},
                status=502)
        choice = gpu_response["choices"][0]
        message = choice["message"]
        output = message.get("content") or ""
        usage = gpu_response["usage"]
        reply = {
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "role": "assistant",
            "model": HF_ID,
            "content": [{"type": "text", "text": output}],
            "stop_reason": "max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": usage["prompt_tokens"],
                      "output_tokens": usage["completion_tokens"]},
        }
        try:
            parsed = json.loads(output)
        except ValueError:
            parsed = None
        call.update(
            status="done",
            latency_ms=round((time.monotonic() - began) * 1000),
            output=output,
            parsed=parsed,
            reasoning=message.get("reasoning_content") or message.get("reasoning"),
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            gpu_response=gpu_response,
            reply_to_game=reply,
        )
        self.publish("call", call)
        asyncio.create_task(self.add_template(call))
        return web.json_response(reply)

    async def add_template(self, call):
        """Ask vLLM for the exact text the model read: the chat template applied."""
        request = call["gpu_request"]
        try:
            async with self.session.post(self.base + "/tokenize", headers=self.auth, json={
                    "model": SERVED_NAME, "messages": request["messages"],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": request["chat_template_kwargs"]}) as response:
                tokens = (await response.json())["tokens"]
            async with self.session.post(self.base + "/detokenize", headers=self.auth, json={
                    "model": SERVED_NAME, "tokens": tokens}) as response:
                call["template_text"] = (await response.json())["prompt"]
            call["template_tokens"] = len(tokens)
        except Exception as error:
            call["template_error"] = str(error)
        self.publish("call", call)
        self.save(call)

    async def status(self, request):
        if not trusted(request):
            return web.json_response({"error": "forbidden"}, status=403)
        self.meta.update(await request.json())
        self.publish("meta", self.meta)
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=1))
        return web.json_response({"ok": True})

    async def events(self, request):
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        queue = asyncio.Queue()
        self.queues.add(queue)
        try:
            await response.write(f"data: {json.dumps({'type': 'meta', 'meta': self.meta})}\n\n".encode())
            for call in list(self.calls):
                await response.write(f"data: {json.dumps({'type': 'call', 'call': call})}\n\n".encode())
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), 15)
                    await response.write(f"data: {data}\n\n".encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.queues.discard(queue)
        return response

    async def page(self, _request):
        return web.FileResponse(HERE / "viewer.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-url", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--parley-port", type=int, default=8890)
    args = parser.parse_args()
    key = dotenv_values(HERE / ".env")["VLLM_API_KEY"]
    Path(args.run_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    recorder = Recorder(args.modal_url, key, args.run_dir, args.parley_port)
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.on_startup.append(recorder.start)
    app.on_cleanup.append(recorder.stop)
    app.router.add_get("/", recorder.page)
    app.router.add_get("/events", recorder.events)
    app.router.add_post("/v1/messages", recorder.messages)
    app.router.add_post("/api/status", recorder.status)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
