#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const MarkdownIt = require('markdown-it');
const footnote = require('markdown-it-footnote');
const hljs = require('highlight.js');
const mdEngine = new MarkdownIt({html:true,linkify:true,highlight(code,lang) {
 if (lang && lang !== 'mermaid' && hljs.getLanguage(lang)) {
   return hljs.highlight(code,{language:lang,ignoreIllegals:true}).value;
 }
 return '';
}}).use(footnote);
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'build', 'site');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'book.json'), 'utf8'));
const chapters = manifest.chapters;
const escape = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const args = process.argv.slice(2);
const i = args.indexOf('--base-path');
let base = i >= 0 ? args[i+1] : '/';
if (!base || !/^\/(?:[A-Za-z0-9._~-]+\/)*$/.test(base)) throw new Error('base-path must look like / or /repository/');
if (chapters.length === 0) throw new Error('No chapters in book.json');
const seen = new Set();
for (const c of chapters) {
  if (!/^[a-z0-9-]+$/.test(c.slug) || seen.has(c.slug)) throw new Error('Invalid or duplicate slug: ' + c.slug);
  seen.add(c.slug);
  if (!fs.existsSync(path.join(root,c.path))) throw new Error('Missing chapter: '+c.path);
}
fs.rmSync(output,{recursive:true,force:true});
fs.mkdirSync(path.join(output,'chapters'),{recursive:true});
fs.copyFileSync(path.join(root,'site/style.css'),path.join(output,'style.css'));
fs.copyFileSync(path.join(root,'site/site.js'),path.join(output,'site.js'));
fs.copyFileSync(require.resolve('highlight.js/styles/github-dark.css'),path.join(output,'highlight.css'));
for (const dir of ['assets','scripts']) {
  if (fs.existsSync(path.join(root,dir))) fs.cpSync(path.join(root,dir),path.join(output,dir),{recursive:true,filter:src => !src.includes('__pycache__')});
}
function nav(active) {
 return `<aside class="sidebar"><h2>目录 · ${chapters.length} 章</h2><nav>${chapters.map((c,n)=>`${(!n || c.part!==chapters[n-1].part) && c.part ? `<h3>${escape(c.part)}</h3>` : ''}<a ${c.slug===active?'class="active" aria-current="page"':''} href="${base}chapters/${c.slug}.html">${String(n+1).padStart(2,'0')} ${escape(c.title)}</a>`).join('')}</nav></aside>`;
}
function shell(title, content, active) {
 return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)} · ${escape(manifest.title)}</title><meta name="description" content="从训练目标、数据和方法出发，理解大模型训练的工程实践。"><link rel="stylesheet" href="${base}style.css"><link rel="stylesheet" href="${base}highlight.css"><script>window.MathJax={tex:{displayMath:[['$$','$$'],['\\\\[','\\\\]']],inlineMath:[['$','$'],['\\\\(','\\\\)']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script><script defer src="${base}site.js"></script></head><body><header class="top"><div class="top-inner"><a class="brand" href="${base}">大模型训练 · 从目标到系统</a><small>一本面向工程实践的开放书稿</small></div></header><div class="layout">${nav(active)}<main>${content}<footer>书稿持续更新 · 技术与工具版本请以引用的官方资料为准</footer></main></div></body></html>`;
}
function referencePanel(html) {
 const refs = new Map();
 html = html.replace(/<a\s+href="(https?:[^"<>]+)"([^>]*)>([\s\S]*?)<\/a>/g, (whole,url,attrs,label) => {
   let ref=refs.get(url);
   if (!ref) {ref={number:refs.size+1,label,url,uses:0};refs.set(url,ref);}
   ref.uses++;
   const id=`source-use-${ref.number}-${ref.uses}`;
   return `${whole}<sup class="source-marker" id="${id}"><a href="#source-${ref.number}" aria-label="参考资料 ${ref.number}">[${ref.number}]</a></sup>`;
 });
 const panel = refs.size ? `<section class="page-references" aria-labelledby="references-title"><h2 id="references-title">本页参考资料</h2><p>编号对应正文中的来源标记。链接标题沿用书稿标注；官方文档可能随版本更新。原创算例与未实测配置按正文说明理解。</p><ol>${Array.from(refs.values(),r=>`<li id="source-${r.number}"><strong>${r.label}</strong><br><a href="${r.url}" rel="noopener noreferrer">${r.url}</a> <a class="source-back" href="#source-use-${r.number}-1" aria-label="返回首次引用">↩ 返回正文</a></li>`).join('')}</ol></section>` : `<section class="page-references"><h2>本页参考资料</h2><p>本页为原创导读或教学说明；具体技术论述的来源见相应章节。本页没有外部资料链接。</p></section>`;
 return html + panel;
}
function markdown(md,c) {
 // Protect TeX from Markdown interpreting underscores or angle brackets.
 const maths = [];
 md = md.replace(/```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g, x => {if (x.startsWith('`')) return x; const n=maths.length;maths.push(escape(x));return `MATHPLACEHOLDER${n}END`;});
 let html = mdEngine.render(md);
 html = html.replace(/MATHPLACEHOLDER(\d+)END/g,(_,n)=>maths[Number(n)]);
 // The chapter source and output both live one directory below the site root.
 html = html.replace(/(href|src)="([^"#?]+)([^\"]*)"/g,(all,attr,raw,suffix)=>{
   if (/^(?:https?:|mailto:|\/)/.test(raw)) return all;
   const target = path.normalize(path.join(path.dirname(c.path), decodeURI(raw)));
   const match = chapters.find(x=>path.normalize(x.path)===target);
   if (match) return `${attr}="${base}chapters/${match.slug}.html${suffix}"`;
   return all;
 });
 return referencePanel(html.replace(/<table>/g,'<div class="table-wrap"><table>').replace(/<\/table>/g,'</table></div>'));
}
const parts = new Map();
for (const c of chapters) {const part=c.part||'章节';if(!parts.has(part))parts.set(part,[]);parts.get(part).push(c);}
const routes=[['先看全局','验收、数据、显存、算力与上线成本。','16-engineering-design'],['深入计算','注意力、损失、梯度、优化器与通信。','17-computation-depth'],['动手训练','HF 单卡训练、NeMo 集群训练与 verl 在线 RL。','18-hf-workflow']];
const intro=`<section class="hero"><div class="eyebrow">工程与原理 · 从目标到系统</div><h1>${escape(manifest.title)}</h1><p>先设计完整项目，再理解计算过程，最后沿典型工具链进入实现。</p><a class="button" href="${base}chapters/16-engineering-design.html">从完整项目开始 →</a></section><section class="reading-paths"><h2>选择阅读路径</h2><div class="cards">${routes.map(([title,desc,slug])=>`<a class="path-card" href="${base}chapters/${slug}.html"><strong>${title}</strong><p>${desc}</p><span>开始阅读 →</span></a>`).join('')}</div></section>${Array.from(parts,([part,list])=>`<section class="part-section chapter-grid"><h2>${escape(part)}</h2>${list.map(c=>{const text=fs.readFileSync(path.join(root,c.path),'utf8');const summary=text.split(/\n\s*\n/).find(x=>!x.startsWith('#'))||'';return `<a class="chapter-row" href="${base}chapters/${c.slug}.html"><strong>${escape(c.title)}</strong><p>${escape(summary.replace(/[*#]/g,'').slice(0,100))}…</p><small>约 ${Math.max(1,Math.ceil(text.length/650))} 分钟 · 阅读 →</small></a>`}).join('')}</section>`).join('')}<p class="note">阅读时长为粗略估计。教学算例与未实测 GPU 配置均在正文标明；引用可从正文编号跳转至本页参考资料。</p>`;
fs.writeFileSync(path.join(output,'index.html'),shell('首页',intro,null));
chapters.forEach((c,n)=>{
 const md=fs.readFileSync(path.join(root,c.path),'utf8');
 const previous=n?`<a href="${base}chapters/${chapters[n-1].slug}.html">← ${escape(chapters[n-1].title)}</a>`:'<span></span>';
 const next=n+1<chapters.length?`<a href="${base}chapters/${chapters[n+1].slug}.html">${escape(chapters[n+1].title)} →</a>`:'<span></span>';
 fs.writeFileSync(path.join(output,'chapters',c.slug+'.html'),shell(c.title,`<article class="article">${markdown(md,c)}</article><nav class="pager" aria-label="章节导航">${previous}${next}</nav>`,c.slug));
});
fs.writeFileSync(path.join(output,'.nojekyll'),'');
console.log(`Built ${chapters.length} chapters in ${output} (base ${base})`);
