"""Server locale per lavorare sulle slide: ricostruisce index.html a ogni richiesta E ogni volta che si salva un file del deck, e la pagina
si ricarica da sola (resta sulla stessa slide). uso: python3 serve.py [porta]  poi aprire http://localhost:8765  (Ctrl+C per fermare).
Per la presentazione finale basta aprire index.html (nessun server)."""
import http.server, sys, functools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
state = {"sig": None}


def refresh():
    sig = build.signature()
    if sig != state["sig"]:
        build.build(quiet=True); state["sig"] = sig
    return sig


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/__version":                      # la pagina lo interroga: se cambia, si ricarica
            body = refresh().encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            refresh()
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


root = str(Path(__file__).resolve().parent)
refresh()
print(f"Slide su http://localhost:{PORT}  (ricostruzione e ricarica automatiche quando salvi; Ctrl+C per fermare)")
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(H, directory=root)).serve_forever()
