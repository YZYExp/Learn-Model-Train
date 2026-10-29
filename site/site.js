// Progressive enhancement: text and code remain readable if a CDN is unavailable.
(async () => {
  const blocks = document.querySelectorAll('pre code.language-mermaid');
  if (blocks.length) {
    try {
      const { default: mermaid } = await import('https://cdn.jsdelivr.net/npm/mermaid@11.12.2/dist/mermaid.esm.min.mjs');
      mermaid.initialize({startOnLoad:false, securityLevel:'strict', theme:'neutral'});
      for (let i = 0; i < blocks.length; i++) {
        const source = blocks[i].textContent;
        try {
          const {svg} = await mermaid.render('diagram-' + i, source);
          const container = document.createElement('div');
          container.className = 'mermaid';
          container.innerHTML = svg;
          blocks[i].parentElement.replaceWith(container);
        } catch (err) { console.warn('Diagram retained as source:', err); }
      }
    } catch (err) { console.warn('Diagram renderer unavailable; showing source.', err); }
  }
})();
