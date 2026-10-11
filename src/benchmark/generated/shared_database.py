"""One CouchDB process, with an authenticated database namespace per execution.

The gateway preserves logical database names for frozen MCP implementations.
Only it knows the server-admin credential. Case credentials cannot address a
different namespace, even through _all_dbs or an encoded path.
"""
from contextlib import contextmanager
import atexit
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
import json
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import threading
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

import requests

from .auth import private_json

_DB = re.compile(r"[a-z][a-z0-9_$()+-]*\Z")
_lock = threading.RLock()
_service = None
_users = 0


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, backend, auth):
        self.backend, self.auth = backend, auth
        self.cases = {}
        self.sessions = {}
        self.guard = threading.RLock()
        super().__init__(("127.0.0.1", 0), Handler)

    def request(self, method, path, **kwargs):
        return requests.request(method, self.backend + path, auth=self.auth,
                                timeout=120, **kwargs)

    def register(self):
        password = secrets.token_hex(32)
        prefix = "case_" + uuid4().hex + "__"
        with self.guard:
            self.cases[password] = prefix
        return {"url": f"http://host.docker.internal:{self.server_port}",
                "username": "evaluation", "password": password, "namespace": prefix}

    def unregister(self, lease):
        with self.guard:
            self.cases.pop(lease["password"], None)
            self.sessions = {k: v for k, v in self.sessions.items() if v != lease["password"]}
        response = self.request("GET", "/_all_dbs")
        response.raise_for_status()
        for name in response.json():
            if name.startswith(lease["namespace"]):
                self.request("DELETE", "/" + quote(name, safe="")).raise_for_status()

    def restore(self, lease, snapshot):
        for source in sorted(Path(snapshot).glob("*.json")):
            if not _DB.fullmatch(source.stem):
                raise ValueError("Unsupported snapshot database name")
            path = "/" + quote(lease["namespace"] + source.stem, safe="")
            self.request("PUT", path).raise_for_status()
            # Defense in depth: even direct backend access cannot use a case login.
            self.request("PUT", path + "/_security", json={
                "admins": {"names": [], "roles": ["_admin"]},
                "members": {"names": [], "roles": ["_admin"]}}).raise_for_status()
            docs = json.loads(source.read_text())
            for start in range(0, len(docs), 500):
                response = self.request("POST", path + "/_bulk_docs", json={"docs": docs[start:start + 500]})
                response.raise_for_status()
                if any("error" in row for row in response.json()):
                    raise ValueError("Failed to restore isolated database")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass  # Never log credentials, request bodies, or case data.

    def reply(self, status, body, headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def route(self):
        try:
            url = urlsplit(self.path)
            if url.scheme or url.netloc:
                return self.reply(400, b'{"error":"invalid_path"}')
            header = self.headers.get("Authorization", "")
            username, password = "", ""
            if header.startswith("Basic "):
                username, password = base64.b64decode(header[6:], validate=True).decode().split(":", 1)
            else:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                with self.server.guard:
                    password = self.server.sessions.get(cookie["AuthSession"].value, "") if "AuthSession" in cookie else ""
                username = "evaluation" if password else ""
            with self.server.guard:
                prefix = self.server.cases.get(password) if username == "evaluation" else None
            if prefix is None:
                self.close_connection = True
                return self.reply(401, b'{"error":"unauthorized"}')
            if url.path == "/_session":
                headers = {"Content-Type": "application/json"}
                if self.command == "POST":
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    token = secrets.token_hex(32)
                    with self.server.guard:
                        self.server.sessions[token] = password
                    headers["Set-Cookie"] = f"AuthSession={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600"
                elif self.command == "DELETE":
                    headers["Set-Cookie"] = "AuthSession=; Path=/; Max-Age=0"
                    with self.server.guard:
                        cookie = SimpleCookie(self.headers.get("Cookie", ""))
                        if "AuthSession" in cookie:
                            self.server.sessions.pop(cookie["AuthSession"].value, None)
                elif self.command != "GET":
                    return self.reply(405, b'{"error":"method_not_allowed"}')
                return self.reply(200, json.dumps({"ok": True, "name": "evaluation", "roles": ["_admin"],
                    "userCtx": {"name": "evaluation", "roles": ["_admin"]}}).encode(), headers)
            parts = url.path.lstrip("/").split("/")
            logical = unquote(parts[0])
            if self.command == "GET" and url.path == "/_all_dbs":
                response = self.server.request("GET", "/_all_dbs")
                response.raise_for_status()
                names = [n[len(prefix):] for n in response.json() if n.startswith(prefix)]
                return self.reply(200, json.dumps(names).encode(), {"Content-Type": "application/json"})
            if url.path in ("/", "/_up") and self.command in ("GET", "HEAD"):
                path = url.path
            elif _DB.fullmatch(logical) and all(unquote(p) not in (".", "..") for p in parts[1:]):
                if len(parts) > 1 and unquote(parts[1]) == "_security":
                    return self.reply(403, b'{"error":"forbidden"}')
                path = "/" + quote(prefix + logical, safe="")
                if len(parts) > 1:
                    path += "/" + "/".join(parts[1:])
            else:
                return self.reply(403, b'{"error":"forbidden"}')
            if url.query:
                path += "?" + url.query
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {key: self.headers[key] for key in ("Content-Type", "If-Match", "If-None-Match", "Destination") if key in self.headers}
            # COPY's destination must remain in this database; remote/global operations are never forwarded.
            if "Destination" in headers and ("/" in unquote(headers["Destination"]) or ":" in headers["Destination"]):
                return self.reply(403, b'{"error":"forbidden"}')
            response = self.server.request(self.command, path, data=body, headers=headers)
            if self.command == "PUT" and len(parts) == 1 and response.status_code == 201:
                secured = self.server.request("PUT", path.split("?", 1)[0] + "/_security", json={
                    "admins": {"names": [], "roles": ["_admin"]},
                    "members": {"names": [], "roles": ["_admin"]}})
                secured.raise_for_status()
            payload = response.content
            if len(parts) == 1 and self.command == "GET" and response.ok and _DB.fullmatch(logical):
                info = response.json()
                if isinstance(info, dict) and info.get("db_name") == prefix + logical:
                    info["db_name"] = logical
                    payload = json.dumps(info).encode()
            forwarded = {k: v for k, v in response.headers.items() if k.lower() in {"content-type", "etag", "cache-control"}}
            return self.reply(response.status_code, payload, forwarded)
        except (ValueError, UnicodeError):
            self.close_connection = True
            return self.reply(400, b'{"error":"invalid_request"}')
        except requests.RequestException:
            self.close_connection = True
            return self.reply(503, b'{"error":"database_unavailable"}')

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_COPY = route


class Service:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="assetops-database-")
        self.folder = Path(self.temporary.name)
        self.name = "aob-shared-" + uuid4().hex[:12]
        self.gateway = None
        password = secrets.token_hex(32)
        private_json(self.folder / "compose.json", {"name": self.name, "services": {"database": {
            "image": "couchdb:3.5", "environment": {"COUCHDB_USER": "evaluation_admin", "COUCHDB_PASSWORD": password,
                "ERL_FLAGS": "+S 2:2 +SDcpu 1 +SDio 2 +A 4"},
            "ports": ["127.0.0.1::5984"],
            "entrypoint": ["tini", "--", "sh", "-c", "printf '[couchdb]\\nsingle_node=true\\n' > /opt/couchdb/etc/local.d/evaluation.ini; exec /docker-entrypoint.sh /opt/couchdb/bin/couchdb"],
            "healthcheck": {"test": ["CMD", "curl", "-fsS", "http://localhost:5984/_up"], "interval": "1s", "timeout": "2s", "retries": 40}}}})
        try:
            self.compose("up", "-d", "--wait")
            port = self.compose("port", "database", "5984").stdout.strip().rsplit(":", 1)[-1]
            self.gateway = Gateway("http://127.0.0.1:" + port, ("evaluation_admin", password))
            self.thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def compose(self, *args):
        return subprocess.run(["docker", "compose", "-f", str(self.folder / "compose.json"), *args],
                              check=True, capture_output=True, text=True, timeout=120)

    def close(self):
        if self.gateway:
            self.gateway.shutdown()
            self.gateway.server_close()
            self.thread.join(timeout=5)
        try:
            self.compose("down", "--volumes", "--remove-orphans")
        finally:
            self.temporary.cleanup()


@contextmanager
def case_database(snapshot):
    global _service, _users
    with _lock:
        if _service is None:
            _service = Service()
        service = _service
        _users += 1
    lease = service.gateway.register()
    try:
        service.gateway.restore(lease, snapshot)
        yield lease
    finally:
        try:
            service.gateway.unregister(lease)
        finally:
            with _lock:
                _users -= 1
                if not _users:
                    _service = None
                    service.close()


def _shutdown():
    global _service
    with _lock:
        if _service is not None:
            service, _service = _service, None
            service.close()


atexit.register(_shutdown)
