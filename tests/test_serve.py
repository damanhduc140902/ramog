"""app.Engine.complete and the OpenAI-compatible server, CPU only: a fake generator replays fixed token ids, so the
text handling (incremental decode, multi-byte hold-back, stop strings, stop ids, cancel) and the HTTP layer
(/v1/models, chat and text completions, streaming SSE, errors) are tested without a model.

python tests/test_serve.py [--tokenizer DIR]   (DIR: a real Hugging Face tokenizer, e.g. Qwen3's; default: a
                                                built-in byte tokenizer)
"""

import argparse
import http.client
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_tiering import app, serve  # noqa: E402


class ByteTok:
    """UTF-8 bytes as tokens (a multi-byte character spans several tokens, as with byte-level BPE); 256 = EOS."""
    eos_token_id = 256

    def __call__(self, text):
        return {"input_ids": list(text.encode())}

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
        s = "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>"
        return list(s.encode())


class FakeGen:
    def __init__(self, script):
        self.script = script

    def generate(self, ids, n, temperature=0.0, top_p=1.0, top_k=0, stop_ids=(), seed=None, on_token=None):
        out, finish = [], "length"
        for t in self.script[:n]:
            out.append(t)
            if on_token is not None and on_token(t) is False:
                return out, "cancel", 0.01
            if t in stop_ids:
                return out, "stop", 0.01
        return out, finish, 0.01


def make_engine(tok, text, eos=True):
    e = object.__new__(app.Engine)
    e.tok, e.name, e.ctx, e.lock = tok, "fake", 4096, threading.Lock()
    e.stop_ids = [tok.eos_token_id]
    e.gen = FakeGen(tok(text)["input_ids"] + ([tok.eos_token_id] if eos else []))
    return e


TEXT = "Xin chào thế giới! 你好, world. STOP here and more"


def check_complete(tok) -> dict:
    r = {}
    pieces = []
    e = make_engine(tok, TEXT)
    text, finish, u = e.complete([1, 2, 3], 1000, on_text=lambda s: pieces.append(s))
    r["full_text"] = text == TEXT and finish == "stop" and "".join(pieces) == TEXT
    r["no_replacement_char_streamed"] = all("�" not in p for p in pieces)
    r["streamed_in_pieces"] = len(pieces) > 5
    pieces = []
    text, finish, _ = e.complete([1], 1000, stop=["STOP"], on_text=lambda s: pieces.append(s))
    r["stop_string"] = text == TEXT[:TEXT.index("STOP")] and finish == "stop" and "".join(pieces) == text
    text, finish, u = e.complete([1], 7)
    r["length"] = finish == "length" and u.completion_tokens == 7
    pieces = []
    _, finish, _ = e.complete([1], 1000, on_text=lambda s: pieces.append(s) or len(pieces) < 2)
    r["cancel"] = finish == "cancel" and len(pieces) == 2
    r["ok"] = all(r.values())
    return r


def check_http(tok) -> dict:
    r = {}
    e = make_engine(tok, TEXT)
    srv = serve.make_server(e, "127.0.0.1", 0, name="fake", defaults={"max_new_tokens": 1000, "temperature": 0.0})
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    def req(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body=None if body is None else json.dumps(body),
                  headers={"Content-Type": "application/json"})
        resp = c.getresponse()
        data = resp.read()
        c.close()
        return resp.status, resp.getheader("Content-Type"), data

    s, _, d = req("GET", "/v1/models")
    r["models"] = s == 200 and json.loads(d)["data"][0]["id"] == "fake"
    msgs = [{"role": "user", "content": "hi"}]
    s, _, d = req("POST", "/v1/chat/completions", {"model": "fake", "messages": msgs})
    j = json.loads(d)
    r["chat"] = (s == 200 and j["object"] == "chat.completion" and j["choices"][0]["message"]["content"] == TEXT
                 and j["choices"][0]["finish_reason"] == "stop" and j["usage"]["completion_tokens"] > 0)
    s, ct, d = req("POST", "/v1/chat/completions", {"model": "fake", "messages": msgs, "stream": True,
                                                     "stream_options": {"include_usage": True}})
    events = [x[6:] for x in d.decode().split("\n\n") if x.startswith("data: ")]
    chunks = [json.loads(x) for x in events[:-1]]
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    r["chat_stream"] = (s == 200 and ct == "text/event-stream" and events[-1] == "[DONE]"
                        and chunks[0]["choices"][0]["delta"].get("role") == "assistant" and content == TEXT
                        and chunks[-1]["choices"][0]["finish_reason"] == "stop" and "usage" in chunks[-1])
    s, _, d = req("POST", "/v1/chat/completions", {"messages": msgs, "stop": "STOP", "max_tokens": 1000})
    r["chat_stop"] = json.loads(d)["choices"][0]["message"]["content"] == TEXT[:TEXT.index("STOP")]
    s, _, d = req("POST", "/v1/completions", {"prompt": "once upon", "max_tokens": 5})
    j = json.loads(d)
    r["completion"] = (s == 200 and j["object"] == "text_completion" and j["choices"][0]["finish_reason"] == "length"
                       and j["usage"]["completion_tokens"] == 5)
    s, _, d = req("POST", "/v1/chat/completions", {"messages": []})
    r["bad_request"] = s == 400 and "error" in json.loads(d)
    s, _, _ = req("GET", "/nope")
    r["not_found"] = s == 404
    srv.shutdown()
    srv.server_close()
    r["ok"] = all(r.values())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer")
    a = ap.parse_args()
    toks = [("bytes", ByteTok())]
    if a.tokenizer:
        from transformers import AutoTokenizer

        toks.append((a.tokenizer, AutoTokenizer.from_pretrained(a.tokenizer)))
    ok = True
    for name, tok in toks:
        c = check_complete(tok)
        h = check_http(tok)
        print(name, json.dumps({"complete": c, "http": h}), flush=True)
        ok = ok and c["ok"] and h["ok"]
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
