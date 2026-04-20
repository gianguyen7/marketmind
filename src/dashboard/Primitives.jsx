// Primitives.jsx — small building blocks for the dashboard

function MetricTile({ label, value, sub, delta, deltaType }) {
  return (
    <div className="mm-tile">
      <div className="mm-tile-label">{label}</div>
      <div className="mm-tile-value">
        {value}
        {delta && <span className={"mm-delta mm-delta--" + (deltaType || "up")}>{delta}</span>}
      </div>
      {sub && <div className="mm-tile-sub">{sub}</div>}
    </div>
  );
}

function Callout({ kind = "info", eyebrow, title, children }) {
  return (
    <div className={"mm-callout mm-callout--" + kind}>
      {eyebrow && <div className="mm-callout-eb">{eyebrow}</div>}
      {title && <div className="mm-callout-t">{title}</div>}
      <div className="mm-callout-body">{children}</div>
    </div>
  );
}

function Badge({ kind = "neutral", children }) {
  return <span className={"mm-badge mm-badge--" + kind}>{children}</span>;
}

function Button({ kind = "primary", children, ...rest }) {
  return <button className={"mm-btn mm-btn--" + kind} {...rest}>{children}</button>;
}

function DataTable({ columns, rows }) {
  return (
    <div className="mm-table-wrap">
      <table className="mm-table">
        <thead>
          <tr>{columns.map(c => (
            <th key={c.key} className={c.numeric ? "is-num" : ""}>
              <span className="mm-th-label">
                {c.label}
                {c.hint && (
                  <span className="mm-tt" tabIndex={0} aria-label={c.hint}>
                    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
                      <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.2"/>
                      <circle cx="8" cy="5" r="0.9" fill="currentColor"/>
                      <path d="M8 7.5v4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                    </svg>
                    <span className="mm-tt-body">{c.hint}</span>
                  </span>
                )}
              </span>
            </th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map(c => {
                const v = r[c.key];
                const cls = (c.numeric ? "is-num " : "") + (c.signed && typeof v === "number" ? (v >= 0 ? "is-pos" : "is-neg") : "");
                return <td key={c.key} className={cls.trim()}>{c.format ? c.format(v, r) : v}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TabBar({ tabs, active, onChange }) {
  return (
    <div className="mm-tabs">
      {tabs.map(t => (
        <button
          key={t.key}
          className={"mm-tab" + (active === t.key ? " is-active" : "")}
          onClick={() => onChange(t.key)}
        >{t.label}</button>
      ))}
    </div>
  );
}

function SectionHeader({ eyebrow, title, description, right }) {
  return (
    <div className="mm-section-h">
      <div>
        {eyebrow && <div className="mm-eyebrow">{eyebrow}</div>}
        <h2 className="mm-h2" style={{margin: eyebrow ? "6px 0 4px" : "0 0 4px"}}>{title}</h2>
        {description && <p className="mm-section-desc">{description}</p>}
      </div>
      {right}
    </div>
  );
}

Object.assign(window, { MetricTile, Callout, Badge, Button, DataTable, TabBar, SectionHeader });
