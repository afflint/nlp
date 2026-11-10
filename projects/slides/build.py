"""Costruisce index.html concatenando slides/*.html (ordinati per nome) dentro template.html.
Le slide si modificano in slides/*.html, NON in index.html (che viene rigenerato). uso: python3 build.py"""
import sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
idx = root / "index.html"


def build(quiet=False):
    parts = sorted((root / "slides").glob("*.html"))
    body = "\n".join(f"    <!-- {p.name} -->\n" + p.read_text() for p in parts)
    out = (root / "template.html").read_text().replace("{{SLIDES}}", body)
    idx.write_text(out)
    if not quiet:
        print("index.html:", len(parts), "file di slide:", ", ".join(p.name for p in parts) or "(nessuno)")


def signature():
    """Impronta dei file che compongono il deck: cambia quando si salva una slide, il tema, il template o il JS."""
    files = [*root.glob("slides/*.html"), root / "template.html", *root.glob("css/*.css"), *root.glob("js/*.js")]
    return ";".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in sorted(files))


def watch(interval=0.5):
    """Ricostruisce index.html ogni volta che un file del deck viene salvato (basta poi ricaricare il browser)."""
    last = None
    print("Controllo i file di slide: salva una slide e ricarica il browser (Ctrl+C per fermare).")
    while True:
        sig = signature()
        if sig != last:
            build(); last = sig
        time.sleep(interval)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        build()
