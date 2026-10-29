#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {marked} = require('marked');
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
for (const dir of ['assets','scripts']) {
  if (fs.existsSync(path.join(root,dir))) fs.cpSync(path.join(root,dir),path.join(output,dir),{recursive:true,filter:src => !src.includes('__pycache__')});
}
function nav(active) {
 return `<aside class="sidebar"><h2>目录 · ${chapters.length} 章</h2><nav>${chapters.map((c,n)=>`${(!n || c.part!==chapters[n-1].part) && c.part ? `<h3>${escape(c.part)}</h3>` : ''}<a ${c.slug===active?'class="active" aria-current="page"':''} href="${base}chapters/${c.slug}.html">${String(n+1).padStart(2,'0')} ${escape(c.title)}</a>`).join('')}</nav></aside>`;
}
function shell(title, content, active) {
 return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)} · ${escape(manifest.title)}</title><meta name="description" content="从训练目标、数据和方法出发，理解大模型训练的工程实践。"><link rel="stylesheet" href="${base}style.css"><script>window.MathJax={tex:{displayMath:[['$$','$$'],['\\\\[','\\\\]']],inlineMath:[['$','$'],['\\\\(','\\\\)']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script><script defer src="${base}site.js"></script></head><body><header class="top"><div class="top-inner"><a class="brand" href="${base}">大模型训练 · 从目标到系统</a><small>一本面向工程实践的开放书稿</small></div></header><div class="layout">${nav(active)}<main>${content}<footer>书稿持续更新 · 技术与工具版本请以引用的官方资料为准</footer></main></div></body></html>`;
}
function footnotes(md) {
 const notes = new Map();
 md = md.replace(/^\[\^([^\]]+)\]:\s*(.+)$/gm, (_,key,val)=>{notes.set(key,val);return '';});
 md = md.replace(/\[\^([^\]]+)\]/g, (m,key)=>notes.has(key)?`<sup id="fnref-${escape(key)}"><a href="#fn-${escape(key)}">[${escape(key)}]</a></sup>`:m);
 if (notes.size) md += '\n\n<div class="footnotes">\n\n### 注释与资料\n\n' + Array.from(notes,([key,val])=>`<p id="fn-${escape(key)}"><strong>[${escape(key)}]</strong> ${marked.parseInline(val)} <a href="#fnref-${escape(key)}" aria-label="返回正文">↩</a></p>`).join('\n') + '\n</div>\n';
 return md;
}
function markdown(md,c) {
 // Protect TeX from Markdown interpreting underscores or angle brackets.
 const maths = [];
 md = md.replace(/```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g, x => {if (x.startsWith('`')) return x; const n=maths.length;maths.push(escape(x));return `MATHPLACEHOLDER${n}END`;});
 let html = marked.parse(footnotes(md), {gfm:true});
 html = html.replace(/MATHPLACEHOLDER(\d+)END/g,(_,n)=>maths[Number(n)]);
 // The chapter source and output both live one directory below the site root.
 html = html.replace(/(href|src)="([^"#?]+)([^\"]*)"/g,(all,attr,raw,suffix)=>{
   if (/^(?:https?:|mailto:|\/)/.test(raw)) return all;
   const target = path.normalize(path.join(path.dirname(c.path), decodeURI(raw)));
   const match = chapters.find(x=>path.normalize(x.path)===target);
   if (match) return `${attr}="${base}chapters/${match.slug}.html${suffix}"`;
   return all;
 });
 return html.replace(/<table>/g,'<div class="table-wrap"><table>').replace(/<\/table>/g,'</table></div>');
}
const intro=`<section class="hero"><div class="eyebrow">AN ENGINEER'S GUIDE TO MODEL TRAINING</div><h1>${escape(manifest.title)}</h1><p>从任务目标出发，贯通数据、训练、蒸馏、系统与评测。每一步既讲工程选择，也为想深入的读者保留原理与资料。</p><a class="button" href="${base}chapters/${chapters[0].slug}.html">开始阅读 →</a></section><section class="chapter-grid"><h2>全书目录</h2><p class="note">本书持续写作。示例的验证状态与工具版本请参见各章说明。</p><div class="cards">${chapters.map((c,n)=>`<a class="card" href="${base}chapters/${c.slug}.html"><span>CHAPTER ${String(n+1).padStart(2,'0')}</span><strong>${escape(c.title)}</strong></a>`).join('')}</div></section>`;
fs.writeFileSync(path.join(output,'index.html'),shell('首页',intro,null));
chapters.forEach((c,n)=>{
 const md=fs.readFileSync(path.join(root,c.path),'utf8');
 const previous=n?`<a href="${base}chapters/${chapters[n-1].slug}.html">← ${escape(chapters[n-1].title)}</a>`:'<span></span>';
 const next=n+1<chapters.length?`<a href="${base}chapters/${chapters[n+1].slug}.html">${escape(chapters[n+1].title)} →</a>`:'<span></span>';
 fs.writeFileSync(path.join(output,'chapters',c.slug+'.html'),shell(c.title,`<article class="article">${markdown(md,c)}</article><nav class="pager" aria-label="章节导航">${previous}${next}</nav>`,c.slug));
});
fs.writeFileSync(path.join(output,'.nojekyll'),'');
console.log(`Built ${chapters.length} chapters in ${output} (base ${base})`);
