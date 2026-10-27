// node --test "tests/js/*.test.js"   (also run by tests/test_frontend.py under pytest)
const test = require('node:test');
const assert = require('node:assert/strict');
const {parse, parseInline, render} = require('../../app/static/markdown.js');

// --- parsing ------------------------------------------------------------------------

test('bold, italic, code and citations', () => {
  assert.deepEqual(parseInline('At least **5 working days** before [1].'), [
    {type: 'text', text: 'At least '},
    {type: 'strong', children: [{type: 'text', text: '5 working days'}]},
    {type: 'text', text: ' before '},
    {type: 'cite', n: 1},
    {type: 'text', text: '.'},
  ]);
  assert.deepEqual(parseInline('*only* `FIN-207`'), [
    {type: 'em', children: [{type: 'text', text: 'only'}]},
    {type: 'text', text: ' '},
    {type: 'code', text: 'FIN-207'},
  ]);
  assert.deepEqual(parseInline('[1][2]'), [{type: 'cite', n: 1}, {type: 'cite', n: 2}]);
});

test('Georgian text with bold', () => {
  assert.deepEqual(parseInline('მინიმუმ **5 სამუშაო დღით ადრე** [3]'), [
    {type: 'text', text: 'მინიმუმ '},
    {type: 'strong', children: [{type: 'text', text: '5 სამუშაო დღით ადრე'}]},
    {type: 'text', text: ' '},
    {type: 'cite', n: 3},
  ]);
});

test('unclosed bold mid-stream stays literal until it closes', () => {
  assert.deepEqual(parseInline('at least **5 work'), [{type: 'text', text: 'at least **5 work'}]);
});

test('paragraphs, line breaks, bullet and numbered lists, headings', () => {
  const blocks = parse('## Rules\nFirst line\nsecond line\n\n- one **a**\n* two\n\n3. third\n4. fourth\n\nEnd.');
  assert.deepEqual(blocks.map(b => b.type), ['h', 'p', 'ul', 'ol', 'p']);
  assert.equal(blocks[0].level, 2);
  assert.deepEqual(blocks[1].children.map(c => c.type), ['text', 'br', 'text']);
  assert.equal(blocks[2].items.length, 2);
  assert.equal(blocks[3].start, 3);
  assert.equal(blocks[3].items.length, 2);
});

test('a line starting with bold is not a bullet', () => {
  assert.deepEqual(parse('**No.** It is not returnable.').map(b => b.type), ['p']);
});

test('HTML and markdown links stay text', () => {
  const nodes = parseInline('<img src=x onerror=alert(1)> [click](javascript:alert(1))');
  assert.deepEqual(nodes, [{type: 'text', text: '<img src=x onerror=alert(1)> [click](javascript:alert(1))'}]);
});

// --- rendering (against a minimal DOM that records what gets created) ---------------

function fakeDocument() {
  const doc = {
    createElement(tag) {
      const el = {tag, children: [], attrs: {}, ownerDocument: doc,
        appendChild(child) { this.children.push(child); return child; },
        replaceChildren() { this.children = []; }};
      Object.defineProperty(el, 'textContent', {set(v) { el.children = [{text: String(v)}]; }});
      for (const prop of ['href', 'className', 'start']) {
        Object.defineProperty(el, prop, {set(v) { el.attrs[prop] = v; }});
      }
      return el;
    },
    createTextNode(text) { return {text}; },
  };
  return doc;
}

function tagsIn(node, out = []) {
  if (node.tag) out.push(node.tag);
  (node.children || []).forEach(c => tagsIn(c, out));
  return out;
}

function textIn(node) {
  return node.text !== undefined ? node.text : (node.children || []).map(textIn).join('');
}

test('render builds only whitelisted elements, and HTML in the answer stays text', () => {
  const container = fakeDocument().createElement('div');
  const answer = 'Hi <script>alert(1)</script> **<img src=x onerror=alert(1)>**\n\n- <b>item</b> [2]';
  render(answer, container);

  const tags = tagsIn(container).slice(1);  // skip the container itself
  assert.deepEqual(tags, ['p', 'strong', 'ul', 'li', 'a']);
  assert.ok(textIn(container).includes('<script>alert(1)</script>'));
  assert.ok(textIn(container).includes('<img src=x onerror=alert(1)>'));
});

test('citation links only point at #source-N on this page', () => {
  const container = fakeDocument().createElement('div');
  render('See [12] and [x](javascript:alert(1))', container);
  const link = container.children[0].children.find(c => c.tag === 'a');
  assert.equal(link.attrs.href, '#source-12');
  assert.equal(tagsIn(container).filter(t => t === 'a').length, 1);
});

test('render replaces previous content (re-render on every streamed delta)', () => {
  const container = fakeDocument().createElement('div');
  render('**5 work', container);
  render('**5 working days**', container);
  assert.equal(container.children.length, 1);
  assert.deepEqual(tagsIn(container).slice(1), ['p', 'strong']);
});
