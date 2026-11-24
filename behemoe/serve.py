"""OpenAI-compatible HTTP API over an app.Engine (standard library only).

    GET  /health, /v1/models
    POST /v1/chat/completions   messages, max_tokens | max_completion_tokens, temperature, top_p, top_k, seed, stop,
                                stream, tools (passed to the chat template; tool calls come back as text),
                                chat_template_kwargs (e.g. {"enable_thinking": false}, as vLLM / llama-server take it)
    POST /v1/completions        prompt (a string), same sampling fields

One generation at a time (Engine's lock); other requests wait. Streaming uses server-sent events in OpenAI's chunk
format, ending with "data: [DONE]"; a client that disconnects cancels its generation. Missing sampling fields take
the server's defaults (the serve command's --temperature etc.).
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _sampling(body: dict, defaults: dict) -> dict:
    out = {k: body.get(k, defaults.get(k)) for k in ("temperature", "top_p", "top_k", "seed")}
    out = {k: v for k, v in out.items() if v is not None}
    stop = body.get("stop")
    if stop:
        out["stop"] = [stop] if isinstance(stop, str) else list(stop)
    return out


def _max_tokens(body: dict, defaults: dict) -> int:
    for k in ("max_completion_tokens", "max_tokens"):
        if body.get(k) is not None:
            return int(body[k])
    return int(defaults.get("max_new_tokens", 1024))


def make_handler(engine, name: str, defaults: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # one line per request on stderr
            super().log_message(fmt, *args)

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str, kind: str = "invalid_request_error") -> None:
            self._json(code, {"error": {"message": msg, "type": kind}})

        def do_GET(self):  # noqa: N802
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": name, "object": "model",
                                                                    "owned_by": "behemoe"}]})
            self._error(404, f"no route {self.path}")

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                return self._error(400, f"bad JSON: {e}")
            if self.path == "/v1/chat/completions":
                if not isinstance(body.get("messages"), list) or not body["messages"]:
                    return self._error(400, "messages: a non-empty list is required")
                try:
                    ids = engine.encode_chat(body["messages"], tools=body.get("tools"),
                                             template_kwargs=body.get("chat_template_kwargs"))
                except Exception as e:  # noqa: BLE001 - the chat template rejected the conversation
                    return self._error(400, f"chat template: {e}")
                return self._run(body, ids, chat=True)
            if self.path == "/v1/completions":
                p = body.get("prompt")
                if isinstance(p, list) and len(p) == 1:
                    p = p[0]
                if not isinstance(p, str):
                    return self._error(400, "prompt: one string is required")
                return self._run(body, engine.encode_text(p), chat=False)
            self._error(404, f"no route {self.path}")

        def _run(self, body: dict, ids: list[int], chat: bool) -> None:
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            obj = "chat.completion" if chat else "text_completion"
            kw = _sampling(body, defaults)
            mx = _max_tokens(body, defaults)
            if not body.get("stream"):
                try:
                    text, finish, u = engine.complete(ids, mx, **kw)
                except ValueError as e:
                    return self._error(400, str(e))
                choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
                          if chat else {"index": 0, "text": text, "finish_reason": finish})
                return self._json(200, {"id": rid, "object": obj, "created": created, "model": name,
                                        "choices": [choice], "usage": _usage(u)})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            cobj = "chat.completion.chunk" if chat else "text_completion"

            def send(payload) -> bool:
                try:
                    self.wfile.write(b"data: " + (payload if isinstance(payload, bytes)
                                                  else json.dumps(payload).encode()) + b"\n\n")
                    self.wfile.flush()
                    return True
                except OSError:  # the client went away: cancel
                    return False

            def chunk(delta_text=None, finish=None, role=False):
                if chat:
                    delta = {"role": "assistant", "content": ""} if role else (
                        {"content": delta_text} if delta_text is not None else {})
                    c = {"index": 0, "delta": delta, "finish_reason": finish}
                else:
                    c = {"index": 0, "text": delta_text or "", "finish_reason": finish}
                return {"id": rid, "object": cobj, "created": created, "model": name, "choices": [c]}

            if chat and not send(chunk(role=True)):
                return
            try:
                _text, finish, u = engine.complete(ids, mx, on_text=lambda s: send(chunk(s)), **kw)
            except ValueError as e:
                send({"error": {"message": str(e), "type": "invalid_request_error"}})
                return
            if finish == "cancel":
                return
            last = chunk(finish=finish)
            if body.get("stream_options", {}).get("include_usage"):
                last["usage"] = _usage(u)
            send(last)
            send(b"[DONE]")

    return Handler


def _usage(u) -> dict:
    return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "total_tokens": u.prompt_tokens + u.completion_tokens}


def make_server(engine, host: str = "127.0.0.1", port: int = 8000, name: str = "model", defaults: dict | None = None):
    return ThreadingHTTPServer((host, port), make_handler(engine, name, defaults or {}))


def serve(engine, host: str = "127.0.0.1", port: int = 8000, name: str = "model", defaults: dict | None = None):
    srv = make_server(engine, host, port, name, defaults)
    print(f"serving {name} on http://{host}:{srv.server_address[1]}/v1 (OpenAI-compatible; Ctrl+C to stop)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
