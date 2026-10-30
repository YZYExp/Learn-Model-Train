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
 return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)} · ${escape(manifest.title)}</title><meta name="description" content="从训练目标、数据和方法出发，理解大模型训练的工程实践。"><link rel="stylesheet" href="${base}style.css"><link rel="stylesheet" href="${base}highlight.css"><script>window.MathJax={tex:{displayMath:[['$$','$$'],['\\\\[','\\\\]']],inlineMath:[['$','$'],['\\\\(','\\\\)']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script><script defer src="${base}site.js"></script></head><body class="${active===null?'home-page':'reader-page'}"><header class="top"><div class="top-inner"><a class="brand" href="${base}">大模型训练 · 从目标到系统</a><a class="github-link" href="https://github.com/YZYExp/Learn-Model-Train" aria-label="在 GitHub 查看本书源码">GitHub <span aria-hidden="true">↗</span></a></div></header><div class="layout">${active===null?'':nav(active)}<main>${content}<footer>书稿持续更新 · 技术与工具版本请以引用的官方资料为准 <a href="https://github.com/YZYExp/Learn-Model-Train">GitHub ↗</a></footer></main></div></body></html>`;
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
const routes=[['01','从项目出发','如何规划数据、算力与验收？','16-engineering-design'],['02','理解训练原理','一次参数更新里发生了什么？','17-computation-depth'],['03','进入工程实践','如何用工具把训练真正跑起来？','18-hf-workflow']];
const partDescriptions=['从目标、约束和数据出发，设计一次训练。','理解模型、损失、优化和知识迁移。','将训练方法落到单卡、多卡和集群。','分析性能、评估结果并交付模型。','按需查阅方法、框架与工具。'];
const intro=`<div class="home-shell"><section class="home-hero"><div class="hero-copy"><div class="hero-kicker"><span class="kicker-dot"></span> 一 本 开 放 的 工 程 实 践 书</div><h1>大模型训练<span>从目标到系统<span class="title-period">.</span></span></h1><p>从一项真实任务开始，走过数据、训练、评测与部署，理解模型背后的工程决策与计算原理。</p><div class="home-actions"><a class="home-button" href="${base}chapters/16-engineering-design.html">开始阅读 <span aria-hidden="true">↗</span></a><a class="home-text-link" href="#contents">浏览全部章节 <span aria-hidden="true">↓</span></a></div></div><div class="hero-art" aria-label="训练流程示意：数据、模型、评估、迭代"><div class="art-top"><span>TRAINING SYSTEM</span><span>01 — 04</span></div><div class="art-core"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="art-center"><span>MODEL</span><strong>训练</strong><small>从数据到决策</small></div><span class="art-node node-data">数据 <b>01</b></span><span class="art-node node-eval">评测 <b>03</b></span><span class="art-node node-loop">迭代 <b>04</b></span><span class="art-node node-system">系统 <b>02</b></span></div><div class="art-bottom"><span>DATA → MODEL → EVALUATION</span><span>↗</span></div></div></section><section class="home-facts" aria-label="书籍信息"><div><strong>${chapters.length}</strong><span>章节<br>从基础到实践</span></div><div><strong>03</strong><span>阅读路径<br>按照你的目标开始</span></div><div><strong>∞</strong><span>开放书稿<br>持续完善与更新</span></div><a href="https://github.com/YZYExp/Learn-Model-Train">查看 GitHub 仓库 <span aria-hidden="true">↗</span></a></section><section class="home-routes"><div class="home-section-head"><div><span class="home-section-label">WHERE TO START / 从这里开始</span><h2>选择你的阅读路径</h2></div><p>先看全局，深入原理，或者直接动手。</p></div><div class="route-grid">${routes.map(([number,title,desc,slug])=>`<a class="route-item" href="${base}chapters/${slug}.html"><span class="route-number">${number} /</span><div><h3>${title}</h3><p>${desc}</p></div><span class="route-arrow" aria-hidden="true">↗</span></a>`).join('')}</div></section><section class="home-contents" id="contents"><div class="home-section-head"><div><span class="home-section-label">THE BOOK / 全书目录</span><h2>一本书，贯通训练全过程</h2></div><p>按主题探索；每章都可独立阅读。</p></div><div class="contents-list">${Array.from(parts,([part,list],i)=>`<details class="contents-part" ${i===0?'open':''}><summary><span class="part-index">${String(i+1).padStart(2,'0')}</span><span class="part-title"><strong>${escape(part.replace(/^[一二三四五六七八九十]+、/,''))}</strong><small>${escape(partDescriptions[i]||'探索本部分的训练知识。')}</small></span><span class="part-count">${list.length} 章</span><span class="part-chevron" aria-hidden="true">＋</span></summary><div class="contents-entries">${list.map(c=>`<a href="${base}chapters/${c.slug}.html"><span>${escape(c.title)}</span><span aria-hidden="true">↗</span></a>`).join('')}</div></details>`).join('')}</div><p class="home-note">本书持续更新。教学算例和未实测配置在正文标明，参考资料可从章节内查看。</p></section></div>`;
fs.writeFileSync(path.join(output,'index.html'),shell('首页',intro,null));
chapters.forEach((c,n)=>{
 const md=fs.readFileSync(path.join(root,c.path),'utf8');
 const previous=n?`<a href="${base}chapters/${chapters[n-1].slug}.html">← ${escape(chapters[n-1].title)}</a>`:'<span></span>';
 const next=n+1<chapters.length?`<a href="${base}chapters/${chapters[n+1].slug}.html">${escape(chapters[n+1].title)} →</a>`:'<span></span>';
 fs.writeFileSync(path.join(output,'chapters',c.slug+'.html'),shell(c.title,`<article class="article">${markdown(md,c)}</article><nav class="pager" aria-label="章节导航">${previous}${next}</nav>`,c.slug));
});
fs.writeFileSync(path.join(output,'.nojekyll'),'');
console.log(`Built ${chapters.length} chapters in ${output} (base ${base})`);
