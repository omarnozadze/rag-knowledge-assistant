// Minimal, safe Markdown for model answers.
//
// Supports what answers actually use: paragraphs and line breaks, **bold**, *italic*,
// `code`, "- " / "* " bullet lists, "1." numbered lists, # headings, and citations [1].
//
// Safe by construction: parse() turns text into a plain tree, render() builds DOM nodes
// with createElement/createTextNode only. Nothing is ever assigned to innerHTML, so
// "<script>" or "<img onerror=...>" in an answer (or in a document the answer quotes)
// shows up as text and can't execute. Markdown links [text](url) are deliberately not
// supported, so no URL from the model ever becomes an href; the only links rendered are
// citations pointing at this page's own #source-N anchors.
(function () {
  const BULLET = /^\s*[-*•]\s+(.*)$/;
  const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
  const HEADING = /^\s*(#{1,3})\s+(.*)$/;
  // Order matters: ** before *, so "**bold**" isn't read as two italics
  const INLINE = /\*\*(.+?)\*\*|`([^`]+)`|\*([^*\s](?:[^*]*[^*\s])?)\*|\[(\d+)\]/g;

  function parseInline(text) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(INLINE)) {
      if (m.index > last) out.push({type: 'text', text: text.slice(last, m.index)});
      if (m[1] !== undefined) out.push({type: 'strong', children: parseInline(m[1])});
      else if (m[2] !== undefined) out.push({type: 'code', text: m[2]});
      else if (m[3] !== undefined) out.push({type: 'em', children: parseInline(m[3])});
      else out.push({type: 'cite', n: Number(m[4])});
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push({type: 'text', text: text.slice(last)});
    return out;
  }

  function parse(markdown) {
    const blocks = [];
    let paragraph = null;   // lines of the paragraph being built
    let list = null;        // the list being built

    const endParagraph = () => {
      if (paragraph) {
        const children = [];
        paragraph.forEach((line, i) => {
          if (i > 0) children.push({type: 'br'});
          children.push(...parseInline(line));
        });
        blocks.push({type: 'p', children});
      }
      paragraph = null;
    };
    const endList = () => { if (list) blocks.push(list); list = null; };

    for (const line of markdown.replace(/\r\n?/g, '\n').split('\n')) {
      let m;
      if (!line.trim()) {
        endParagraph(); endList();
      } else if ((m = line.match(HEADING))) {
        endParagraph(); endList();
        blocks.push({type: 'h', level: m[1].length, children: parseInline(m[2])});
      } else if ((m = line.match(BULLET)) || (m = line.match(NUMBERED))) {
        endParagraph();
        const ordered = m.length === 3;
        const kind = ordered ? 'ol' : 'ul';
        if (!list || list.type !== kind) {
          endList();
          list = {type: kind, items: []};
          if (ordered) list.start = Number(m[1]);
        }
        list.items.push(parseInline(ordered ? m[2] : m[1]));
      } else if (list && /^\s+\S/.test(line)) {
        // Indented continuation of the previous list item
        const item = list.items[list.items.length - 1];
        item.push({type: 'text', text: ' '}, ...parseInline(line.trim()));
      } else {
        endList();
        (paragraph = paragraph || []).push(line.trim());
      }
    }
    endParagraph(); endList();
    return blocks;
  }

  function renderInline(nodes, parent, doc) {
    for (const node of nodes) {
      if (node.type === 'text') parent.appendChild(doc.createTextNode(node.text));
      else if (node.type === 'br') parent.appendChild(doc.createElement('br'));
      else if (node.type === 'code') {
        const el = doc.createElement('code');
        el.textContent = node.text;
        parent.appendChild(el);
      } else if (node.type === 'cite') {
        const a = doc.createElement('a');
        a.className = 'cite';
        a.href = '#source-' + node.n;   // node.n is a parsed integer, never model text
        a.textContent = '[' + node.n + ']';
        parent.appendChild(a);
      } else {
        const el = doc.createElement(node.type);  // 'strong' | 'em' only
        renderInline(node.children, el, doc);
        parent.appendChild(el);
      }
    }
  }

  // Replaces container's content with the rendered markdown
  function render(markdown, container) {
    const doc = container.ownerDocument;
    container.replaceChildren();
    for (const block of parse(markdown)) {
      let el;
      if (block.type === 'p') {
        el = doc.createElement('p');
        renderInline(block.children, el, doc);
      } else if (block.type === 'h') {
        el = doc.createElement('h' + (block.level + 3));  // h4-h6 under the card's h3
        renderInline(block.children, el, doc);
      } else {
        el = doc.createElement(block.type);
        if (block.start && block.start !== 1) el.start = block.start;
        for (const item of block.items) {
          const li = doc.createElement('li');
          renderInline(item, li, doc);
          el.appendChild(li);
        }
      }
      container.appendChild(el);
    }
  }

  const api = {parse, parseInline, render};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;  // node tests
  else window.Markdown = api;
})();
