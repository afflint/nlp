/* Inizializzazione del deck: 16:9 a 1920x1080, nessuna nota del relatore, logo gestito via CSS (nascosto nella copertina). */
const deck = new Reveal({
  width: 1920, height: 1080, margin: 0, minScale: 0.1, maxScale: 3,
  hash: true, fragmentInURL: true, controls: true, progress: true, slideNumber: 'c/t', center: false,
  transition: 'fade', transitionSpeed: 'fast', backgroundTransition: 'none', showNotes: false, pdfSeparateFragments: false
});
const syncCover = () => { const s = deck.getCurrentSlide(); const h = document.documentElement.classList; h.toggle('is-cover', !!(s && s.hasAttribute('data-cover'))); h.toggle('on-blue', !!(s && s.hasAttribute('data-section'))); };
deck.on('ready', syncCover); deck.on('slidechanged', syncCover);
// tasto "t": tema chiaro/scuro
document.addEventListener('keydown', e => { if (e.key === 't' && !e.ctrlKey && !e.metaKey) { const r = document.documentElement; r.dataset.theme = r.dataset.theme === 'dark' ? 'light' : 'dark'; } });
// ?static nell'URL: nasconde frecce, numero di slide e barra di avanzamento (per l'esportazione in PDF)
if (/[?&]static\b/.test(location.search)) document.documentElement.classList.add('static-export');
deck.initialize();
// Solo in modalita' server (serve.py): se un file del deck cambia, ricarica la pagina restando sulla stessa slide e allo stesso click
if (location.protocol.startsWith('http')) {
  let v = null;
  const poll = async () => { try { const r = await fetch('/__version', { cache: 'no-store' }); if (!r.ok) return; const t = await r.text(); if (v === null) v = t; else if (t !== v) { location.reload(); return; } } catch (e) { return; } setTimeout(poll, 800); };
  poll();
}
