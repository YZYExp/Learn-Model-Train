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

// Navigation and reading controls enhance the static content, never replace it.
(() => {
  const sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    const nav = sidebar.querySelector('nav');
    if (nav) {
      nav.id = 'book-navigation';
      const toggle = document.createElement('button');
      toggle.type = 'button'; toggle.className = 'menu-toggle';
      toggle.setAttribute('aria-controls', nav.id);
      sidebar.prepend(toggle);
      const media = window.matchMedia('(max-width: 760px)');
      let expanded = false;
      const update = () => {
        nav.hidden = media.matches && !expanded;
        toggle.setAttribute('aria-expanded', String(!nav.hidden));
        toggle.textContent = nav.hidden ? '展开全书目录 ＋' : '收起全书目录 −';
      };
      toggle.addEventListener('click', () => { expanded = !expanded; update(); });
      media.addEventListener('change', update); update();
    }
  }
  const article = document.querySelector('.article');
  if (!article) return;
  const headings = [...article.querySelectorAll('h2, h3')].filter(h => !h.closest('.footnotes,.page-references'));
  if (headings.length > 2) {
    const details = document.createElement('details'); details.className = 'chapter-tools';
    const summary = document.createElement('summary'); summary.textContent = `本章导览 · ${headings.filter(h=>h.tagName==='H2').length} 节`;
    const nav = document.createElement('nav'); nav.setAttribute('aria-label', '本章目录');
    headings.forEach((heading, i) => {
      if (!heading.id) heading.id = `section-${i + 1}`;
      const a = document.createElement('a'); a.href = '#' + heading.id;
      a.textContent = heading.textContent; if (heading.tagName === 'H3') a.className = 'toc-sub';
      nav.append(a);
    });
    details.append(summary, nav);
    const h1 = article.querySelector('h1');
    if (h1) h1.after(details); else article.prepend(details);
  }
  article.querySelectorAll('pre').forEach(pre => {
    const code = pre.querySelector('code');
    if (!code || code.classList.contains('language-mermaid') || !navigator.clipboard) return;
    const button = document.createElement('button'); button.className = 'copy-code'; button.type = 'button'; button.textContent = '复制代码';
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(code.textContent); button.textContent = '已复制'; }
      catch (_) { button.textContent = '请手动复制'; }
      setTimeout(() => {button.textContent = '复制代码';}, 1800);
    });
    pre.append(button);
  });
  const progress = document.createElement('div'); progress.className = 'reading-progress'; progress.setAttribute('aria-hidden', 'true'); document.body.append(progress);
  let scheduled = false;
  const update = () => {
    scheduled = false;
    const start = article.getBoundingClientRect().top + window.scrollY;
    const distance = Math.max(1, article.offsetHeight - window.innerHeight);
    progress.style.width = Math.max(0, Math.min(100, (window.scrollY - start) / distance * 100)) + '%';
  };
  const schedule = () => {if (!scheduled) {scheduled = true; requestAnimationFrame(update);}};
  window.addEventListener('scroll', schedule, {passive: true}); window.addEventListener('resize', schedule); update();
})();
