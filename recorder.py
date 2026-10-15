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
# modal.com/pricing, 2026-10-14, per hour
GPU_DOLLARS_PER_HOUR = {"L40S": 0.000542 * 3600, "H100": 0.001097 * 3600}
# Qwen's recommended settings from the model card, per mode. The game sets
# the temperature (COWORLD_LLM_TEMPERATURE) and the answer's token cap.
SAMPLING = {
    False: {"top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    True: {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    # RL rollouts: sample the policy's own distribution (no top-p/top-k/penalty), so the
    # trainer scores the distribution the samples came from.
    "rl": {"top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0},
}
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


def action_schema(system, user):
    """JSON schema for this call, built from the game's own instruction lines,
    so the decoder can only name cogs the prompt offers. The model still picks
    the move; this only rules out typos and names that are not at the table."""
    me = re.match(r"You are (.+?), a cog", system)
    me = me.group(1) if me else None
    seats = re.findall(r"^- (\S+): ", user, flags=re.M)
    others = [n for n in seats if n != me] or seats
    # Length limits match the game's 160-character cap and rule out empty strings,
    # which the game rejects ("whisper needs text").
    props = {"say": {"type": "string", "minLength": 1, "maxLength": 160}}
    required = ["say"]
    shoot = re.search(r'"shoot": <one of (.+?)>', user)
    if shoot:
        options = re.findall(r'"([^"]+)"', shoot.group(1))
        props["shoot"] = {"type": "string", "enum": options}
        props["aim"] = {"type": "string", "enum": ["head", "hip"]}
        required.append("shoot")
    extras = re.search(r"Optional extra fields, at most one of each: (.+)", user)
    if extras:
        line = extras.group(1)
        if '"whisper"' in line:
            props["whisper"] = {"type": "object", "properties": {"to": {"type": "string", "enum": others},
                                "text": {"type": "string", "minLength": 1, "maxLength": 160}},
                                "required": ["to", "text"], "additionalProperties": False}
        if '"reveal"' in line:
            props["reveal"] = {"type": "object", "properties": {"to": {"type": "string", "enum": others},
                               "card": {"type": "string", "enum": ["friend", "enemy"]}}, "required": ["to", "card"],
                               "additionalProperties": False}
        if '"give"' in line:
            props["give"] = {"type": "string", "enum": others}
        pledge = re.search(r'"pledge": <[^>]*any of ([^>]+)>', line)
        if pledge:
            props["pledge"] = {"type": "string", "enum": [n.strip() for n in pledge.group(1).split(",") if n.strip()]}
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


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
    def __init__(self, modal_url, key, run_dir, parley_port, thinking, thinking_budget, gpu):
        self.base = modal_url.rstrip("/")
        self.gpu = gpu
        self.auth = {"Authorization": f"Bearer {key}"}
        self.run_dir = Path(run_dir)
        self.thinking = thinking
        self.retry_seconds = 600  # covers a Modal container preemption and cold restart
        self.sampling = SAMPLING[thinking]
        self.routes = {}  # seat slot -> served model name (for mixed tables)
        # Extra output tokens for the thinking text, on top of the game's cap
        # for the JSON answer. If the thinking runs past it, the answer is
        # empty and the game rejects the reply.
        self.thinking_budget = thinking_budget if thinking else 0
        self.calls = []
        self.queues = set()
        self.session = None
        self.meta = {
            "model": HF_ID,
            "gpu": gpu,
            "gpu_dollars_per_hour": GPU_DOLLARS_PER_HOUR[gpu],
            "thinking": thinking,
            "thinking_budget": self.thinking_budget,
            "sampling": self.sampling,
            "decoding": "JSON schema per call (legal names only; vLLM response_format json_schema)",
            "parley_port": parley_port,
            "status": "recorder started",
            "gpu_started": None,
            "game": None,
        }
        self.replay = None
        # After a restart, show the calls and status already recorded for this run.
        saved = self.run_dir / "calls.jsonl"
        if saved.exists():
            latest = {}
            for line in saved.read_text().splitlines():
                call = json.loads(line)
                latest[call["n"]] = call
            self.calls = [latest[n] for n in sorted(latest)]
        if (self.run_dir / "meta.json").exists():
            self.meta.update(json.loads((self.run_dir / "meta.json").read_text()))
            self.meta["parley_port"] = parley_port

    def publish(self, kind, payload):
        data = json.dumps({"type": kind, kind: payload})
        for queue in list(self.queues):
            queue.put_nowait(data)

    def save(self, call):
        with (self.run_dir / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(call) + "\n")

    async def start(self, _app):
        # Modal answers a request that runs past 150 s with a redirect that
        # aiohttp follows, so long thinking replies still arrive.
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900, sock_read=300))

    async def stop(self, _app):
        await self.session.close()

    async def complete(self, request, call):
        """One streamed chat completion, retried across server restarts.

        A transport failure (connection error, 5xx, a stream cut off before
        [DONE]) means the model never answered, so the same request is sent
        again until it does or `retry_seconds` run out; the game sees one reply.
        Returns a dict shaped like a non-streamed reply: choices[0].message
        (content, reasoning), finish_reason, usage."""
        began = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self.complete_once(request, call)
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
                waited = time.monotonic() - began
                if waited > self.retry_seconds:
                    raise
                call["transport_retries"] = attempt
                call["status"] = f"GPU server unavailable ({type(error).__name__}); retrying, {waited:.0f} s so far"
                self.publish("call", call)
                await asyncio.sleep(min(30, 5 * attempt))

    async def complete_once(self, request, call):
        streamed = {**request, "stream": True, "stream_options": {"include_usage": True}}
        reasoning, content, finish, usage = [], [], None, None
        last_publish = 0.0
        async with self.session.post(self.base + "/v1/chat/completions",
                                     json=streamed, headers=self.auth) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"GPU server returned {response.status}: {text[:400]}")
            async for raw in response.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(f"GPU server error: {str(chunk['error'])[:200]}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        reasoning.append(piece)
                    if delta.get("content"):
                        content.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                now = time.monotonic()
                if now - last_publish > 0.5:
                    last_publish = now
                    call["reasoning_partial"] = "".join(reasoning)
                    call["output_partial"] = "".join(content)
                    self.publish("call", call)
        if finish is None:
            raise RuntimeError("stream ended before the server finished the reply")
        return {
            "choices": [{"message": {"content": "".join(content) or None,
                                     "reasoning": "".join(reasoning) or None},
                         "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            "streamed": True,
        }

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
            "max_tokens": body["max_tokens"] + self.thinking_budget,
            "temperature": body["temperature"],
            **self.sampling,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "parley_action", "schema": action_schema(system, user)}},
        }
        slot = int(request.headers.get("x-coworld-player-slot", "-1"))
        gpu_request["model"] = self.routes.get(slot, SERVED_NAME)
        call = {
            "n": len(self.calls) + 1,
            "slot": slot,
            "served_model": gpu_request["model"],
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
            gpu_response = await self.complete(gpu_request, call)
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
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if self.thinking and choice.get("finish_reason") == "length" and not output.strip():
            # The thinking ran past its budget. Qwen's documented thinking-budget
            # method: close the thinking with a fixed sentence and let the model
            # write the answer from what it has so far. The game sees one reply.
            call["thinking_cut"] = True
            closed = ("<think>\n" + (reasoning or "") + "\n\nConsidering the limited time by the user, "
                      "I have to give the solution based on the thinking directly now.\n</think>\n\n")
            continuation = {
                **gpu_request,
                "messages": gpu_request["messages"] + [{"role": "assistant", "content": closed}],
                # Room for a long spoken line; the schema bounds strings to 160 characters.
                # Qwen's own sampling settings for this short answer: under plain sampling the
                # grammar-constrained answer sometimes degenerates into endless whitespace.
                # (A repetition penalty is worse: it penalizes the closing braces.)
                "max_tokens": max(body["max_tokens"], 600),
                **SAMPLING[True],
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            call["continuation_request"] = continuation
            try:
                second = await self.complete(continuation, call)
            except Exception as error:
                call.update(status="error", error=str(error),
                            latency_ms=round((time.monotonic() - began) * 1000))
                self.publish("call", call)
                self.save(call)
                return web.json_response(
                    {"type": "error", "error": {"type": "api_error", "message": str(error)}},
                    status=502)
            call["continuation_response"] = second
            choice = second["choices"][0]
            message = choice["message"]
            # The reasoning parser may file the forced answer under either field.
            output = message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
            usage = {"prompt_tokens": usage["prompt_tokens"],
                     "completion_tokens": usage["completion_tokens"] + second["usage"]["completion_tokens"]}
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
        call.pop("reasoning_partial", None)
        call.pop("output_partial", None)
        call.update(
            status="done",
            latency_ms=round((time.monotonic() - began) * 1000),
            output=output,
            parsed=parsed,
            reasoning=reasoning,
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
                    "model": request["model"], "messages": request["messages"],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": request["chat_template_kwargs"]}) as response:
                tokens = (await response.json())["tokens"]
            async with self.session.post(self.base + "/detokenize", headers=self.auth, json={
                    "model": request["model"], "tokens": tokens}) as response:
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

    async def replay_page(self, _request):
        return web.FileResponse(HERE / "replay.html")

    async def replay_frame(self, _request):
        return web.FileResponse(HERE / "replay_frame.html")

    async def parley_client(self, request):
        """Parley's own renderer and stylesheet, from the pinned checkout."""
        name = request.match_info["name"]
        if name not in ("renderer.js", "chrome.css"):
            raise web.HTTPNotFound()
        return web.FileResponse(HERE / "upstream" / "client" / name)

    async def parley_asset(self, request):
        name = request.match_info["name"]
        if "/" in name or name.startswith("."):
            raise web.HTTPNotFound()
        return web.FileResponse(HERE / "upstream" / "data" / name)

    async def replay_data(self, _request):
        """Replay payload plus, for every decision, the model call behind it.

        The payload comes from the Parley server in replay mode, which adds the
        table state after each event. Decisions are joined to recorded calls
        by the exact user message the game sent."""
        if self.replay is None:
            async with self.session.ws_connect(
                    f"ws://127.0.0.1:{self.meta['parley_port']}/replay", max_msg_size=0) as ws:
                payload = json.loads((await ws.receive()).data)
            refs = json.loads((self.run_dir / "replay.json").read_text())["decisionRefs"]
            prompts = {}
            for line in (self.run_dir / "trajectory.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record.get("event_type") != "decision":
                    continue
                for attempt in record["attempts"]:
                    if attempt["attempt_id"] == record["selected_attempt_id"]:
                        prompts[str(record["decision_id"])] = attempt["request"]["messages"][0]["content"]
            by_user = {call["game_request"]["messages"][0]["content"]: call["n"] for call in self.calls}
            decisions = [{**ref, "n": by_user.get(prompts.get(str(ref["id"])))} for ref in refs]
            self.replay = {"payload": payload, "decisions": decisions,
                           "calls": {call["n"]: call for call in self.calls}, "meta": self.meta}
        return web.json_response(self.replay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-url", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--parley-port", type=int, default=8890)
    parser.add_argument("--thinking", action="store_true", help="let the model think before the JSON")
    parser.add_argument("--thinking-budget", type=int, default=3000)
    parser.add_argument("--gpu", choices=sorted(GPU_DOLLARS_PER_HOUR), default="H100")
    parser.add_argument("--sampling", choices=["qwen", "rl"], default="qwen")
    parser.add_argument("--route", action="append", default=[],
                        help="slot=model: serve this seat from another model name (repeatable)")
    args = parser.parse_args()
    key = dotenv_values(HERE / ".env")["VLLM_API_KEY"]
    Path(args.run_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    recorder = Recorder(args.modal_url, key, args.run_dir, args.parley_port,
                        args.thinking, args.thinking_budget, args.gpu)
    if args.sampling == "rl":
        recorder.sampling = SAMPLING["rl"]
        recorder.meta["sampling"] = recorder.sampling
    for route in args.route:
        slot, model = route.split("=", 1)
        recorder.routes[int(slot)] = model
    recorder.meta["routes"] = {str(k): v for k, v in recorder.routes.items()}
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.on_startup.append(recorder.start)
    app.on_cleanup.append(recorder.stop)
    app.router.add_get("/", recorder.page)
    app.router.add_get("/events", recorder.events)
    app.router.add_post("/v1/messages", recorder.messages)
    app.router.add_post("/api/status", recorder.status)
    app.router.add_get("/replay", recorder.replay_page)
    app.router.add_get("/replay-frame", recorder.replay_frame)
    app.router.add_get("/parley/client/assets/{name}", recorder.parley_asset)
    app.router.add_get("/parley/client/{name}", recorder.parley_client)
    app.router.add_get("/api/replay", recorder.replay_data)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
