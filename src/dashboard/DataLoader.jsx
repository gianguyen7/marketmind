// DataLoader.jsx — data fetching utilities for the MarketMind dashboard
const { useState, useEffect } = React;

const DATA_DIR = '../../outputs/tables';
const _cache = {};

/* ---------- CSV parser ---------- */
function _parseLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') { inQuotes = !inQuotes; }
    else if (c === ',' && !inQuotes) { values.push(current.trim()); current = ''; }
    else { current += c; }
  }
  values.push(current.trim());
  return values;
}

function parseCSV(text) {
  const lines = text.trim().split('\n');
  const headers = _parseLine(lines[0]);
  return lines.slice(1).filter(l => l.trim()).map(line => {
    const values = _parseLine(line);
    const row = {};
    headers.forEach((h, i) => {
      const v = (values[i] || '').trim();
      if (v === '' || v === 'inf' || v === 'nan' || v === 'NaN') { row[h] = null; }
      else if (!isNaN(v) && v !== '') { row[h] = parseFloat(v); }
      else { row[h] = v; }
    });
    return row;
  });
}

/* ---------- Fetch helpers with caching ---------- */
async function fetchCSV(filename) {
  if (_cache[filename]) return _cache[filename];
  const res = await fetch(`${DATA_DIR}/${filename}`);
  if (!res.ok) throw new Error(`Failed to load ${filename} (${res.status})`);
  const data = parseCSV(await res.text());
  _cache[filename] = data;
  return data;
}

async function fetchJSON(filename) {
  if (_cache[filename]) return _cache[filename];
  const res = await fetch(`${DATA_DIR}/${filename}`);
  if (!res.ok) throw new Error(`Failed to load ${filename} (${res.status})`);
  const data = await res.json();
  _cache[filename] = data;
  return data;
}

/* ---------- React hook ---------- */
function useData(loadFn) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let cancelled = false;
    loadFn().then(data => {
      if (!cancelled) setState({ data, error: null, loading: false });
    }).catch(error => {
      console.error('Data load error:', error);
      if (!cancelled) setState({ data: null, error, loading: false });
    });
    return () => { cancelled = true; };
  }, []);
  return state;
}

/* ---------- Category name mapping ---------- */
const CAT_NAMES = {
  'recession_economy': 'Recession / economy',
  'sports': 'Sports',
  'entertainment': 'Entertainment',
  'crypto_finance': 'Crypto / finance',
  'fed_monetary_policy': 'Fed monetary policy',
  'politics_elections': 'Politics / elections',
  'geopolitics': 'Geopolitics',
  'government_policy': 'Government policy',
  'science_technology': 'Science / tech',
  'social_media': 'Social media',
  'other': 'Other',
};

function catName(raw) {
  return CAT_NAMES[raw] || raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/* ---------- Number formatters ---------- */
function fmtDelta(d) {
  return (d >= 0 ? '+' : '\u2212') + Math.abs(d).toFixed(4);
}

function fmtPct(d) {
  return (d >= 0 ? '+' : '\u2212') + Math.abs(d * 100).toFixed(1) + '%';
}

function fmtMoney(d) {
  const prefix = d >= 0 ? '+$' : '\u2212$';
  return prefix + Math.abs(Math.round(d)).toLocaleString();
}

/* ---------- Loading / error states ---------- */
function LoadingState() {
  return (
    <div className="mm-page" style={{ padding: '80px 0', textAlign: 'center' }}>
      <div style={{ fontFamily: 'var(--mm-font-mono)', color: 'var(--mm-fg-3)', fontSize: 13 }}>
        Loading data{'\u2026'}
      </div>
    </div>
  );
}

function ErrorState({ error }) {
  return (
    <div className="mm-page">
      <Callout kind="danger" eyebrow="Error" title="Failed to load data">
        {error.message}. Make sure to serve the dashboard with{' '}
        <code>python src/dashboard/serve.py</code> from the project root.
      </Callout>
    </div>
  );
}

Object.assign(window, {
  fetchCSV, fetchJSON, useData, catName,
  fmtDelta, fmtPct, fmtMoney,
  LoadingState, ErrorState, DATA_DIR,
});
