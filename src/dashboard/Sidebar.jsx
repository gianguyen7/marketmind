// Sidebar.jsx — MarketMind dashboard primary navigation
const { useState } = React;

const NAV_ITEMS = [
  { key: "overview", label: "Overview", icon: (
    <><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></>
  )},
  { key: "calibration", label: "Calibration explorer", icon: (
    <><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></>
  )},
  { key: "fl", label: "Favourite-longshot", icon: (
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  )},
  { key: "exploit", label: "Exploitation attempts", icon: (
    <><path d="M10 2v7.5L4 19a2 2 0 0 0 1.7 3h12.6A2 2 0 0 0 20 19l-6-9.5V2"/><line x1="10" y1="2" x2="14" y2="2"/></>
  )},
  { key: "data", label: "Data deep dive", icon: (
    <><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></>
  )},
];

function Sidebar({ active, onNavigate }) {
  return (
    <aside className="mm-sidebar">
      <div className="mm-brand">
        <img src="assets/logo-mark.svg" width="32" height="32" alt="" />
        <div>
          <div className="mm-brand-t">MarketMind</div>
          <div className="mm-brand-s">Calibration research</div>
        </div>
      </div>
      <nav className="mm-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.key}
            className={"mm-nav-item" + (active === item.key ? " is-active" : "")}
            onClick={() => onNavigate(item.key)}
          >
            <svg className="mm-nav-icon" viewBox="0 0 24 24">{item.icon}</svg>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="mm-sidebar-foot">
        <div className="mm-caption">4,538 markets &middot; 639,807 snapshots</div>
        <div className="mm-caption">Jan 2022 to Apr 2026</div>
      </div>
    </aside>
  );
}

window.Sidebar = Sidebar;
