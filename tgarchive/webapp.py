"""Локальный веб-просмотрщик: stdlib http.server + одна страница на vanilla JS."""

import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import db as dbm
from . import viewdata as VD


def _int(qs, key):
    v = qs.get(key)
    return int(v) if v not in (None, "") else None


def make_handler(db_path: Path, archive_dir: Path):
    local = threading.local()
    web_dir = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        @property
        def conn(self):
            if not hasattr(local, "conn"):
                local.conn = dbm.connect(db_path)
            return local.conn

        def _json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype=None):
            if not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            start, end = 0, size - 1
            status = 200
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes=") and size:
                try:
                    a, _, b = rng[6:].split(",")[0].partition("-")
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    else:
                        start = max(0, size - int(b))
                    end = min(end, size - 1)
                    if start <= end:
                        status = 206
                    else:
                        start, end = 0, size - 1
                except ValueError:
                    start, end = 0, size - 1
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _media(self, path: str):
            # /media/<chat_id>/<relpath внутри папки чата>
            parts = path.split("/", 3)
            if len(parts) < 4:
                return self.send_error(404)
            _, _, chat_id, rel = parts
            base = (archive_dir / chat_id).resolve()
            target = (base / urllib.parse.unquote(rel)).resolve()
            if not str(target).startswith(str(base) + "/"):
                return self.send_error(403)
            self._file(target)

        def _messages(self, qs):
            chat_id = _int(qs, "chat")
            if chat_id is None:
                return self._json({"error": "нужен параметр chat"}, 400)
            rows = VD.page_messages(
                self.conn, chat_id,
                topic_id=_int(qs, "topic"),
                sender=qs.get("sender") or None,
                before_id=_int(qs, "before_id"),
                after_id=_int(qs, "after_id"),
                around_id=_int(qs, "around_id"),
                date=qs.get("date") or None,
                limit=min(_int(qs, "limit") or 100, 500),
            )
            self._json({"messages": VD.serialize(self.conn, rows)})

        def _search(self, qs):
            q = (qs.get("q") or "").strip()
            if not q:
                return self._json({"error": "пустой запрос"}, 400)
            try:
                total, rows = VD.search_messages(
                    self.conn, q,
                    any_mode=qs.get("any") == "1",
                    chat_id=_int(qs, "chat"),
                    topic_id=_int(qs, "topic"),
                    sender=qs.get("sender") or None,
                    date_from=qs.get("from") or None,
                    date_to=qs.get("to") or None,
                    limit=min(_int(qs, "limit") or 100, 1000),
                    offset=_int(qs, "offset") or 0,
                )
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            self._json({"total": total, "messages": VD.serialize(self.conn, rows)})

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            try:
                if path == "/":
                    return self._file(web_dir / "index.html", "text/html; charset=utf-8")
                if path == "/api/chats":
                    return self._json(VD.list_chats(self.conn))
                if path == "/api/topics":
                    return self._json(VD.list_topics(self.conn, _int(qs, "chat")))
                if path == "/api/messages":
                    return self._messages(qs)
                if path == "/api/search":
                    return self._search(qs)
                if path.startswith("/media/"):
                    return self._media(path)
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # не роняем тред просмотрщика
                try:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:
                    pass

    return Handler


def serve(root: Path, cfg: dict, host="127.0.0.1", port=8377, open_browser=True):
    db_path = root / cfg["general"].get("db", "data/index.db")
    archive_dir = root / cfg["general"].get("archive_dir", "archive")
    httpd = ThreadingHTTPServer((host, port), make_handler(db_path, archive_dir))
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Просмотрщик: {url}   (Ctrl+C — остановить)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено")
