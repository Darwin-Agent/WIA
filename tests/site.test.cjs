const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const html = read('index.html');
const paperTitle = 'World Intelligence Agent: Unifying World Modeling and Action Learning for Digital Agents';
const messages = JSON.parse(read('i18n.js').match(/const MESSAGES = (\{[\s\S]*?\n\});/)[1]);
const text = value => value.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();

test('every score column highlights only the highest and second-highest distinct scores', () => {
  const tables = [...html.matchAll(/<table class="result-table">([\s\S]*?)<\/table>/g)];
  assert.ok(tables.length > 0);
  for (const [, table] of tables) {
    const headers = [...table.matchAll(/<th>(.*?)<\/th>/g)].map(match => text(match[1]));
    const rows = [...table.matchAll(/<tr\b([^>]*)>([\s\S]*?)<\/tr>/g)]
      .filter(match => !/class="group"/.test(match[1]) && /<td\b/.test(match[2]))
      .map(match => [...match[2].matchAll(/<td\b([^>]*)>(.*?)<\/td>/g)]
        .map(cell => ({ attributes: cell[1], text: text(cell[2]) })));
    assert.ok(rows.length > 0);
    for (const row of rows) assert.equal(row.length, headers.length, row[0].text);
    for (let column = 1; column < headers.length; column++) {
      const scores = rows.map(row => Number(row[column].text));
      assert.ok(scores.every(Number.isFinite), `${headers[column]} must contain numeric scores`);
      const ranked = [...new Set(scores)].sort((a, b) => b - a);
      for (const row of rows) {
        const cell = row[column];
        const classes = (cell.attributes.match(/class="([^"]*)"/)?.[1] || '').split(/\s+/);
        const actual = classes.filter(value => value === 'best' || value === 'second');
        const expected = Number(cell.text) === ranked[0] ? ['best']
          : Number(cell.text) === ranked[1] ? ['second'] : [];
        assert.deepEqual(actual, expected, `${row[0].text}, ${headers[column]} (${cell.text})`);
      }
    }
  }
});

test('the official paper title is consistent in page metadata, hero and paper resource', () => {
  assert.equal(text(html.match(/<title>(.*?)<\/title>/)[1]), paperTitle);
  assert.equal(text(html.match(/<h1[^>]*>([\s\S]*?)<\/h1>/)[1]), paperTitle);
  assert.equal(text(messages.en['index.hero.title']), paperTitle);
  for (const lang of ['en', 'zh']) {
    assert.equal(messages[lang]['index.title'], paperTitle);
    assert.equal(messages[lang]['index.resources.paper.desc'], paperTitle);
  }
  assert.ok(messages.en['index.meta_description'].includes(paperTitle));
});

test('Darwin Agent Team is credited in the hero, page metadata and both footers', () => {
  const teamLink = /<a\b[^>]*href="https:\/\/github\.com\/Darwin-Agent"[^>]*data-i18n-skip[^>]*>Darwin Agent Team<\/a>/;
  const hero = html.match(/<header class="hero">([\s\S]*?)<\/header>/)[1];
  assert.match(hero, /class="hero-authors"/);
  assert.match(hero, teamLink);
  assert.equal(messages.en['common.authored_by'], 'By');
  assert.equal(messages.zh['common.authored_by'], '作者：');
  for (const file of ['index.html', 'demos.html']) {
    const page = read(file);
    assert.match(page, /<meta name="author" content="Darwin Agent Team" \/>/);
    const footer = page.match(/<footer\b[^>]*>([\s\S]*?)<\/footer>/)[1];
    assert.match(footer, teamLink, `${file} credits the team in its footer`);
  }
});

test('all explicitly translated homepage and demo labels exist in both languages', () => {
  for (const file of ['index.html', 'demos.html']) {
    for (const [, key] of read(file).matchAll(/data-i18n(?:-(?:html|content|aria-label|title-key))?="([^"]+)"/g)) {
      for (const lang of ['en', 'zh']) assert.ok(messages[lang][key], `${file}: ${lang}.${key}`);
    }
  }
});

test('the overview video is embedded without autoplay, and its assets are deployable', () => {
  const video = html.match(/<video\b([^>]*id="promoVideo"[^>]*)>([\s\S]*?)<\/video>/);
  assert.ok(video, 'overview video is present');
  assert.match(video[1], /\bcontrols\b/);
  assert.match(video[1], /\bplaysinline\b/);
  assert.match(video[1], /preload="metadata"/);
  assert.doesNotMatch(video[1], /\bautoplay\b/);
  assert.match(video[2], /src="demos\/WIA_promo_1080p\.mp4"/);
  assert.match(video[1], /poster="demos\/WIA_cover_1920x1080\.png"/);
  const workflow = read('.github/workflows/pages.yml');
  for (const asset of ['WIA_promo_1080p.mp4', 'WIA_cover_1920x1080.png']) {
    const stat = fs.lstatSync(path.join(root, 'demos', asset));
    assert.ok(stat.isFile() && !stat.isSymbolicLink());
    assert.ok(stat.size > 0 && stat.size < 100000000, `${asset} fits the publication limit`);
    assert.ok(workflow.includes(asset), `${asset} is in the publication manifest`);
  }
  const bytes = fs.readFileSync(path.join(root, 'demos', 'WIA_promo_1080p.mp4'));
  const atoms = [];
  for (let offset = 0; offset + 8 <= bytes.length;) {
    const size = bytes.readUInt32BE(offset);
    assert.ok(size >= 8 && offset + size <= bytes.length, 'valid MP4 atom');
    atoms.push(bytes.toString('ascii', offset + 4, offset + 8));
    offset += size;
  }
  assert.equal(atoms[0], 'ftyp');
  assert.ok(atoms.includes('moov') && atoms.includes('mdat'));
  assert.ok(atoms.indexOf('moov') < atoms.indexOf('mdat'), 'fast-start metadata precedes media data');
});
