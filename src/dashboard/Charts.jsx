// Charts.jsx — SVG chart primitives for the MarketMind dashboard

function ReliabilityChart({ data, width = 600, height = 340 }) {
  const pad = { l: 68, r: 24, t: 20, b: 52 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const xy = (x, y) => [pad.l + x * innerW, pad.t + (1 - y) * innerH];
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="mm-chart">
      {ticks.map(t => {
        const [,y] = xy(0, t);
        const [x2] = xy(1, t);
        return <line key={"h" + t} x1={pad.l} y1={y} x2={x2} y2={y} stroke="#EAE8E0"/>;
      })}
      {ticks.map(t => {
        const [x,] = xy(t, 0);
        return <line key={"v" + t} x1={x} y1={pad.t} x2={x} y2={pad.t + innerH} stroke="#EAE8E0"/>;
      })}
      <line x1={xy(0,0)[0]} y1={xy(0,0)[1]} x2={xy(1,1)[0]} y2={xy(1,1)[1]} stroke="#9B988C" strokeDasharray="4 3" strokeWidth="1.5"/>
      <polyline
        points={data.map(d => xy(d.x, d.y).join(",")).join(" ")}
        fill="none" stroke="#4F5BD5" strokeWidth="2.5"
      />
      {data.map((d, i) => {
        const [cx, cy] = xy(d.x, d.y);
        return <circle key={i} cx={cx} cy={cy} r="4" fill="#4F5BD5"/>;
      })}
      {ticks.map(t => (
        <text key={"yt" + t} x={pad.l - 10} y={xy(0, t)[1] + 4} textAnchor="end" className="mm-axis">{t.toFixed(2)}</text>
      ))}
      {ticks.map(t => (
        <text key={"xt" + t} x={xy(t, 0)[0]} y={pad.t + innerH + 20} textAnchor="middle" className="mm-axis">{t.toFixed(2)}</text>
      ))}
      <text x={pad.l + innerW/2} y={height - 8} textAnchor="middle" className="mm-axis-lbl">Market price</text>
      <text x={18} y={pad.t + innerH/2} textAnchor="middle" transform={`rotate(-90, 18, ${pad.t + innerH/2})`} className="mm-axis-lbl">Resolution rate</text>
    </svg>
  );
}

function BiasBars({ data, width = 720, height = 400 }) {
  const pad = { l: 80, r: 20, t: 16, b: 110 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const max = Math.max(...data.map(d => Math.abs(d.v)));
  const mid = pad.t + innerH / 2;
  const bw = innerW / data.length * 0.65;
  const gap = innerW / data.length * 0.35;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="mm-chart">
      <line x1={pad.l} y1={mid} x2={pad.l + innerW} y2={mid} stroke="#D9D6CB" strokeWidth="1.2"/>
      {[0.25, 0.5, 0.75, 1].map(t => {
        const yUp = mid - (innerH / 2) * t;
        const yDn = mid + (innerH / 2) * t;
        return (
          <g key={t}>
            <line x1={pad.l} y1={yUp} x2={pad.l + innerW} y2={yUp} stroke="#EAE8E0"/>
            <line x1={pad.l} y1={yDn} x2={pad.l + innerW} y2={yDn} stroke="#EAE8E0"/>
            <text x={pad.l - 10} y={yUp + 4} textAnchor="end" className="mm-axis">+{(t * max * 100).toFixed(0)}%</text>
            <text x={pad.l - 10} y={yDn + 4} textAnchor="end" className="mm-axis">{"\u2212"}{(t * max * 100).toFixed(0)}%</text>
          </g>
        );
      })}
      <text x={pad.l - 10} y={mid + 4} textAnchor="end" className="mm-axis">0%</text>
      {data.map((d, i) => {
        const x = pad.l + gap / 2 + i * (bw + gap);
        const h = (Math.abs(d.v) / max) * (innerH / 2);
        const y = d.v >= 0 ? mid - h : mid;
        const fill = d.v >= 0 ? "#2B9E6B" : "#E44D3A";
        const valLabel = (d.v >= 0 ? "+" : "\u2212") + Math.abs(d.v * 100).toFixed(1) + "%";
        const valY = d.v >= 0 ? y - 6 : y + h + 14;
        return (
          <g key={i}>
            <rect x={x} y={y} width={bw} height={Math.max(h, 1)} fill={fill} opacity="0.85" rx="2"/>
            <text x={x + bw / 2} y={valY} textAnchor="middle" className="mm-axis mm-axis--val" style={{fontSize: 10, fontFamily: 'var(--mm-font-mono)'}}>{valLabel}</text>
            <text
              x={x + bw / 2}
              y={pad.t + innerH + 16}
              textAnchor="end"
              transform={`rotate(-40, ${x + bw / 2}, ${pad.t + innerH + 16})`}
              className="mm-axis mm-axis--cat"
              style={{fontSize: 11}}
            >{d.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function SplitBars({ data, width = 660, height = 320 }) {
  const pad = { l: 64, r: 20, t: 16, b: 100 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const max = Math.max(...data.flatMap(d => [d.train, d.test])) * 1.15;
  const groupW = innerW / data.length * 0.75;
  const groupGap = innerW / data.length * 0.25;
  const bw = groupW / 2 - 2;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="mm-chart">
      {[0, 0.25, 0.5, 0.75, 1].map(t => {
        const y = pad.t + (1 - t) * innerH;
        return (
          <g key={t}>
            <line x1={pad.l} y1={y} x2={pad.l + innerW} y2={y} stroke={t === 0 ? "#9B988C" : "#EAE8E0"} strokeWidth={t === 0 ? 1.2 : 1}/>
            <text x={pad.l - 10} y={y + 4} textAnchor="end" className="mm-axis">{(t * max).toFixed(3)}</text>
          </g>
        );
      })}
      {data.map((d, i) => {
        const gx = pad.l + groupGap / 2 + i * (groupW + groupGap);
        const h1 = (d.train / max) * innerH;
        const h2 = (d.test / max) * innerH;
        return (
          <g key={i}>
            <rect x={gx} y={pad.t + innerH - h1} width={bw} height={h1} fill="#4F5BD5" rx="2"/>
            <rect x={gx + bw + 4} y={pad.t + innerH - h2} width={bw} height={h2} fill="#E44D3A" rx="2"/>
            <text
              x={gx + groupW / 2}
              y={pad.t + innerH + 16}
              textAnchor="end"
              transform={`rotate(-40, ${gx + groupW / 2}, ${pad.t + innerH + 16})`}
              className="mm-axis mm-axis--cat"
              style={{fontSize: 11}}
            >{d.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function CategoryBar({ data, width = 620, height = 340 }) {
  const pad = { l: 128, r: 60, t: 20, b: 44 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const rawMax = Math.max(...data.map(d => d.v));
  const niceMax = Math.ceil(rawMax / 500) * 500;
  const tickCount = 5;
  const ticks = Array.from({length: tickCount + 1}, (_, i) => (niceMax / tickCount) * i);
  const rowH = innerH / data.length;
  const bh = Math.min(rowH * 0.62, 22);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="mm-chart">
      {ticks.map((t, i) => {
        const x = pad.l + (t / niceMax) * innerW;
        return (
          <g key={i}>
            <line x1={x} y1={pad.t} x2={x} y2={pad.t + innerH} stroke={i === 0 ? "#9B988C" : "#EAE8E0"} strokeWidth={i === 0 ? 1.2 : 1}/>
            <text x={x} y={pad.t + innerH + 16} textAnchor="middle" className="mm-axis">{t.toLocaleString()}</text>
          </g>
        );
      })}
      <text x={pad.l + innerW / 2} y={height - 6} textAnchor="middle" className="mm-axis-lbl">Number of markets</text>
      {data.map((d, i) => {
        const y = pad.t + i * rowH + (rowH - bh) / 2;
        const w = (d.v / niceMax) * innerW;
        return (
          <g key={i}>
            <text x={pad.l - 10} y={y + bh / 2 + 4} textAnchor="end" className="mm-axis mm-axis--cat">{d.label}</text>
            <rect x={pad.l} y={y} width={w} height={bh} fill="#4F5BD5" opacity={0.9} rx="2"/>
            <text x={pad.l + w + 6} y={y + bh / 2 + 4} className="mm-axis mm-axis--val" style={{fontFamily: "var(--mm-font-mono)", fontSize: 11}}>
              {d.v.toLocaleString()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function VolumeOverTime({ data, width = 560, height = 220 }) {
  const pad = { l: 48, r: 48, t: 16, b: 32 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const maxN = Math.max(...data.map(d => d.markets)) * 1.15;
  const maxV = Math.max(...data.map(d => d.volume)) * 1.15;
  const bw = innerW / data.length * 0.6;
  const gap = innerW / data.length * 0.4;
  const linePts = data.map((d, i) => {
    const x = pad.l + gap/2 + i * (bw + gap) + bw/2;
    const y = pad.t + (1 - d.volume / maxV) * innerH;
    return [x, y];
  });
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="mm-chart">
      {[0.25, 0.5, 0.75, 1].map(t => {
        const y = pad.t + (1 - t) * innerH;
        return (<g key={t}>
          <line x1={pad.l} y1={y} x2={pad.l + innerW} y2={y} stroke="#EAE8E0"/>
          <text x={pad.l - 8} y={y + 3} textAnchor="end" className="mm-axis">{Math.round(t * maxN).toLocaleString()}</text>
          <text x={pad.l + innerW + 8} y={y + 3} textAnchor="start" className="mm-axis">${(t * maxV / 1e9).toFixed(1)}B</text>
        </g>);
      })}
      {data.map((d, i) => {
        const x = pad.l + gap/2 + i * (bw + gap);
        const h = (d.markets / maxN) * innerH;
        const y = pad.t + innerH - h;
        return (<g key={i}>
          <rect x={x} y={y} width={bw} height={h} fill="#4F5BD5" opacity={0.85} rx="2"/>
          <text x={x + bw/2} y={pad.t + innerH + 18} textAnchor="middle" className="mm-axis">{d.year}</text>
        </g>);
      })}
      <polyline points={linePts.map(p => p.join(",")).join(" ")} fill="none" stroke="#E44D3A" strokeWidth="2.5"/>
      {linePts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3.5" fill="#E44D3A"/>)}
    </svg>
  );
}

Object.assign(window, { ReliabilityChart, BiasBars, SplitBars, CategoryBar, VolumeOverTime });
