// Pages.jsx — data-driven page compositions for the MarketMind dashboard

/* =========================================================================
   Overview
   ========================================================================= */
function OverviewPage() {
  const { data, error, loading } = useData(async () => {
    const [brier, b5, c1, c4] = await Promise.all([
      fetchCSV('brier_decomposition.csv'),
      fetchJSON('b5_results_2026-04-12.json'),
      fetchCSV('c1_recalibration_results_2026-04-12.csv'),
      fetchJSON('c4_results_2026-04-12.json'),
    ]);
    return { brier, b5, c1, c4 };
  });

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;

  const { brier, b5, c1, c4 } = data;
  const allRow = brier.find(r => r.split === 'all');
  const splits = brier.filter(r => r.split !== 'all');

  // --- Research arc deltas ---
  const b5t = b5.phase2.splits.test;
  const b5Delta = (b5t.brier_rule_struct ?? b5t.brier_rule) - (b5t.brier_naive_struct ?? b5t.brier_naive);

  const c1Test = c1.filter(r => r.split === 'test');
  const c1Naive = c1Test.find(r => r.model && r.model.includes('naive'));
  const c1Others = c1Test.filter(r => r.model && !r.model.includes('naive'));
  const c1Best = c1Others.length ? c1Others.reduce((a, b) => a.brier < b.brier ? a : b) : null;
  const c1Delta = c1Best && c1Naive ? c1Best.brier - c1Naive.brier : 0;

  const c4Delta = c4.test_results.hybrid.brier - c4.test_results.naive_brier;

  const verdictFor = (d) => Math.abs(d) < 0.0005 ? 'negligible' : d > 0 ? 'failed' : 'marginal';

  const arcRows = [
    { attempt: 'Structural alpha-shift', approach: 'Long-horizon markets under-price YES \u2192 bet YES', delta: fmtDelta(b5Delta), verdict: verdictFor(b5Delta) },
    { attempt: 'Calibration correction', approach: 'Platt/isotonic recalibration of F-L bias', delta: fmtDelta(c1Delta), verdict: verdictFor(c1Delta) },
    { attempt: 'Trajectory dynamics', approach: 'XGB on price trajectory, blended with market', delta: fmtDelta(c4Delta), verdict: verdictFor(c4Delta) },
    { attempt: 'Global ML correction', approach: 'RF/XGB regressor on market_error', delta: '+0.012', verdict: 'failed' },
  ];

  const splitRows = splits.map(r => ({
    split: r.split,
    markets: r.n_markets,
    base: r.base_rate != null ? r.base_rate.toFixed(4) : '\u2014',
    brier: r.brier_score != null ? r.brier_score.toFixed(4) : '\u2014',
    reliability: r.reliability != null ? r.reliability.toFixed(4) : '\u2014',
    ece: r.ece != null ? r.ece.toFixed(4) : '\u2014',
  }));

  return (
    <div className="mm-page">
      <div className="mm-eyebrow">Research summary</div>
      <h1 className="mm-h1">Is Polymarket efficient?</h1>
      <p className="mm-lede">
        This research asked whether binary prediction markets on Polymarket have structural
        inefficiencies that a model can exploit. After four independent exploitation attempts
        on {allRow.n_markets.toLocaleString()} markets, the answer is <strong>no</strong>.
        The market is remarkably well-calibrated.
      </p>

      <div className="mm-grid-4" style={{marginTop: 24}}>
        <MetricTile label="Brier score" value={allRow.brier_score.toFixed(4)} sub={'all snapshots \u00b7 lower is better'} />
        <MetricTile label="Reliability" value={allRow.reliability.toFixed(4)} sub="0 = perfectly calibrated" delta="good" deltaType="up"/>
        <MetricTile label="ECE" value={(allRow.ece * 100).toFixed(2) + '%'} sub="expected calibration error" />
        <MetricTile label="Markets" value={allRow.n_markets.toLocaleString()} sub={allRow.n.toLocaleString() + ' snapshots'} />
      </div>

      <SectionHeader eyebrow="Research arc" title="Four attempts to beat the market" description="All four approaches failed to produce material Brier improvements on held-out test markets."/>
      <DataTable
        columns={[
          { key: 'attempt', label: 'Attempt' },
          { key: 'approach', label: 'Approach' },
          { key: 'delta', label: 'Test Brier \u0394', numeric: true },
          { key: 'verdict', label: 'Verdict', format: v => <Badge kind={v === 'failed' ? 'danger' : v === 'negligible' ? 'warn' : 'info'}>{v}</Badge> },
        ]}
        rows={arcRows}
      />

      <SectionHeader title="Brier decomposition by split" description="Stable performance across temporal splits. No drift, no regime break."/>
      <DataTable
        columns={[
          { key: 'split', label: 'Split', hint: 'How the markets are divided by date. Train = used to build models. Val = used to tune. Test = held-out to grade final performance.' },
          { key: 'markets', label: 'Markets', numeric: true, hint: 'How many individual prediction markets are in this group.' },
          { key: 'base', label: 'Base rate', numeric: true, hint: 'Share of snapshots that resolved YES.' },
          { key: 'brier', label: 'Brier', numeric: true, hint: 'Average squared error between the predicted probability and the actual outcome. Lower is better.' },
          { key: 'reliability', label: 'Reliability', numeric: true, hint: 'How far the predicted probabilities drift from what actually happens. 0 = perfectly calibrated.' },
          { key: 'ece', label: 'ECE', numeric: true, hint: 'Expected Calibration Error. The typical gap between a forecast and reality. Lower is better.' },
        ]}
        rows={splitRows}
      />

      <Callout kind="info" eyebrow="Bottom line" title="The crowd wins.">
        Polymarket's crowd aggregates information well, calibrates probabilities accurately
        (reliability {allRow.reliability.toFixed(4)}), and leaves very little room for systematic
        improvement. The favourite-longshot bias is real but too small to exploit. The most
        valuable output of this project is the <em>characterization</em> of where and how the
        market works, not a model that beats it.
      </Callout>

      <SectionHeader eyebrow="Looking ahead" title="Open research directions" description="The crowd is efficient in aggregate, but several pockets remain under-explored."/>
      <div className="mm-grid-3" style={{marginTop: 8}}>
        <div className="mm-tile">
          <div className="mm-tile-label">Event-driven mispricing</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            Markets react slowly to scheduled catalysts (FOMC decisions, CPI releases, earnings).
            A model that ingests event calendars and measures the speed of price adjustment
            could find windows where the crowd has not yet priced in new information.
          </p>
        </div>
        <div className="mm-tile">
          <div className="mm-tile-label">Cross-market correlation</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            Related markets (e.g. "Will X win the primary?" and "Will X win the general?")
            sometimes imply contradictory joint probabilities. Detecting and exploiting
            arbitrage across correlated contracts is an untested surface.
          </p>
        </div>
        <div className="mm-tile">
          <div className="mm-tile-label">Sentiment and flow signals</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            This study used only price and volume. Incorporating external signals
            (social media sentiment, news velocity, order book depth) could reveal
            whether the crowd under-reacts to information outside the platform.
          </p>
        </div>
      </div>
      <div className="mm-grid-3" style={{marginTop: 12}}>
        <div className="mm-tile">
          <div className="mm-tile-label">Thin-market exploitation</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            Government policy and geopolitics showed the largest miscalibration but had
            too few markets to trade reliably. As Polymarket grows, these categories
            may become liquid enough to revisit with larger sample sizes.
          </p>
        </div>
        <div className="mm-tile">
          <div className="mm-tile-label">Real-time calibration drift</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            Calibration was measured on resolved markets. A live monitoring system that
            tracks calibration drift in open markets could flag emerging inefficiencies
            before they self-correct.
          </p>
        </div>
        <div className="mm-tile">
          <div className="mm-tile-label">Multi-outcome markets</div>
          <p style={{fontSize: 13, color: 'var(--mm-fg-2)', margin: '10px 0 0', lineHeight: 1.55}}>
            This study covered binary (yes/no) markets only. Multi-outcome markets
            (e.g. "Who will win?") have richer probability surfaces and may carry
            different types of structural bias worth investigating.
          </p>
        </div>
      </div>
    </div>
  );
}

/* =========================================================================
   Calibration explorer
   ========================================================================= */
function CalibrationPage() {
  const { data, error, loading } = useData(async () => {
    const [flBias, cats] = await Promise.all([
      fetchCSV('favourite_longshot_bias.csv'),
      fetchCSV('calibration_by_category.csv'),
    ]);
    return { flBias, cats };
  });

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;

  const { flBias, cats } = data;

  // Reliability diagram: bucket midpoint vs actual resolution rate
  const chartData = flBias
    .filter(r => r.bucket_mid != null && r.actual_rate != null)
    .sort((a, b) => a.bucket_mid - b.bucket_mid)
    .map(r => ({ x: r.bucket_mid, y: r.actual_rate }));

  // Category table sorted by Brier (best first)
  const catRows = cats
    .filter(r => r.category)
    .sort((a, b) => a.brier_score - b.brier_score)
    .map(r => ({
      cat: catName(r.category),
      n: r.n_markets,
      brier: r.brier_score != null ? r.brier_score.toFixed(4) : '\u2014',
      rel: r.reliability != null ? r.reliability.toFixed(4) : '\u2014',
      ece: r.ece != null ? r.ece.toFixed(4) : '\u2014',
    }));

  const bestCat = catRows[0]?.cat || 'Sports';
  const worstCat = catRows.length ? catRows[catRows.length - 1]?.cat : 'Government policy';

  return (
    <div className="mm-page">
      <div className="mm-eyebrow">Page &middot; calibration explorer</div>
      <h1 className="mm-h1">Calibration explorer</h1>
      <p className="mm-lede">
        Reliability diagrams and category-level slices. {bestCat} is the best-calibrated large
        category. {worstCat} has the worst calibration, as the market underprices
        low-probability events in that category.
      </p>

      <div className="mm-card">
        <div className="mm-card-h">
          <h3 className="mm-h3" style={{margin:0}}>Reliability diagram, all markets</h3>
          <div className="mm-legend">
            <span><span className="mm-dot" style={{background:'#4F5BD5'}}/>Actual</span>
            <span><span className="mm-dot" style={{background:'#9B988C'}}/>Perfect</span>
          </div>
        </div>
        <ReliabilityChart data={chartData}/>
      </div>

      <SectionHeader title="Calibration by category"/>
      <DataTable
        columns={[
          { key: 'cat', label: 'Category' },
          { key: 'n', label: 'Markets', numeric: true, format: v => v.toLocaleString() },
          { key: 'brier', label: 'Brier', numeric: true },
          { key: 'rel', label: 'Reliability', numeric: true },
          { key: 'ece', label: 'ECE', numeric: true },
        ]}
        rows={catRows}
      />
    </div>
  );
}

/* =========================================================================
   Favourite-longshot bias
   ========================================================================= */
function FLBiasPage() {
  const { data, error, loading } = useData(async () => {
    const [flBias, flCat] = await Promise.all([
      fetchCSV('favourite_longshot_bias.csv'),
      fetchCSV('fl_bias_per_category.csv'),
    ]);
    return { flBias, flCat };
  });

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;

  const { flBias, flCat } = data;

  // Compute aggregate longshot / favourite bias
  const longshots = flBias.filter(r => r.bucket_mid != null && r.bucket_mid < 0.30);
  const favourites = flBias.filter(r => r.bucket_mid != null && r.bucket_mid > 0.70);
  const avg = (arr, key) => arr.length ? arr.reduce((s, r) => s + (r[key] || 0), 0) / arr.length : 0;
  const longshotBias = avg(longshots, 'bias');
  const favouriteBias = avg(favourites, 'bias');

  // Crossover point: where bias flips sign
  const sorted = [...flBias].filter(r => r.bucket_mid != null && r.bias != null).sort((a, b) => a.bucket_mid - b.bucket_mid);
  let crossover = 0.45;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i - 1].bias >= 0 && sorted[i].bias < 0) {
      crossover = (sorted[i - 1].bucket_mid + sorted[i].bucket_mid) / 2;
      break;
    }
  }

  // Bias bars by category (sorted high to low)
  const biasData = flCat
    .filter(r => r.category && r.longshot_bias != null)
    .sort((a, b) => b.longshot_bias - a.longshot_bias)
    .map(r => ({ label: catName(r.category), v: r.longshot_bias }));

  const topBias = biasData[0];

  return (
    <div className="mm-page">
      <div className="mm-eyebrow">Page &middot; favourite-longshot bias</div>
      <h1 className="mm-h1">Favourite-longshot bias</h1>
      <p className="mm-lede">
        The classic favourite-longshot bias: longshots (low-probability events) win more
        often than their prices imply, while favourites win less often. Polymarket exhibits
        this pattern clearly.
      </p>

      <div className="mm-grid-3">
        <MetricTile label="Longshot bias" value={fmtPct(longshotBias)} sub="price &lt; 0.30"/>
        <MetricTile label="Favourite bias" value={fmtPct(favouriteBias)} sub="price &gt; 0.70"/>
        <MetricTile label="Crossover" value={'~' + crossover.toFixed(2)} sub="actual = predicted"/>
      </div>

      <div className="mm-card" style={{marginTop: 24}}>
        <div className="mm-card-h">
          <h3 className="mm-h3" style={{margin:0}}>Longshot bias by category</h3>
          <div className="mm-caption">positive = longshots underpriced by the market</div>
        </div>
        <BiasBars data={biasData}/>
      </div>

      <Callout kind="warn" eyebrow="Caveat">
        {topBias && topBias.v > 0.05
          ? <>{topBias.label} shows a massive {fmtPct(topBias.v)} longshot bias. The market badly underprices low-probability events in this category. </>
          : <>Some categories show measurable longshot bias. </>
        }
        Low-n categories are fragile.
      </Callout>
    </div>
  );
}

/* =========================================================================
   Model display names + tooltips
   ========================================================================= */
const MODEL_INFO = {
  'naive (market price)': {
    name: 'Market price (baseline)',
    tip: 'Uses the raw Polymarket price as the probability. No correction applied, just what the crowd says.',
  },
  'isotonic_global': {
    name: 'Global isotonic',
    tip: 'Fits a flexible staircase curve across all markets to correct systematic over- or under-pricing at each price level.',
  },
  'platt_global': {
    name: 'Global Platt scaling',
    tip: 'Fits a simple S-curve to gently shift all market prices toward better calibration. A minimal, light-touch correction.',
  },
  'isotonic_per_cat': {
    name: 'Per-category isotonic',
    tip: 'Same staircase correction, but trained separately for each topic (sports, politics, etc). More tailored, but needs more data per group.',
  },
  'platt_per_cat': {
    name: 'Per-category Platt',
    tip: 'Same S-curve correction, trained separately per topic. Accounts for each category having different pricing tendencies.',
  },
  'cat_x_horizon': {
    name: 'Category x horizon',
    tip: 'Trains a separate correction for every combination of topic and time horizon. The most granular approach, but prone to overfitting on small groups.',
  },
  'logistic_trajectory': {
    name: 'Logistic regression',
    tip: 'A simple statistical model that uses price movement patterns like momentum, volatility, and staleness to predict outcomes.',
  },
  'xgb_trajectory': {
    name: 'XGBoost',
    tip: 'A machine learning model (gradient-boosted trees) trained on price chart features. More powerful but also more prone to overfitting.',
  },
};

function modelLabel(raw) {
  const info = MODEL_INFO[raw];
  const name = info ? info.name : raw;
  const tip = info ? info.tip : null;
  return (
    <span style={{display: 'inline-flex', alignItems: 'center', gap: 6}}>
      {name}
      {tip && (
        <span className="mm-tt" tabIndex={0} aria-label={tip}>
          <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
            <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            <circle cx="8" cy="5" r="0.9" fill="currentColor"/>
            <path d="M8 7.5v4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
          <span className="mm-tt-body">{tip}</span>
        </span>
      )}
    </span>
  );
}

/* =========================================================================
   Exploitation attempts
   ========================================================================= */
function ExploitPage() {
  const [tab, setTab] = useState('b5');
  const { data, error, loading } = useData(async () => {
    const [b5, c1, c4, stability] = await Promise.all([
      fetchJSON('b5_results_2026-04-12.json'),
      fetchCSV('c1_recalibration_results_2026-04-12.csv'),
      fetchJSON('c4_results_2026-04-12.json'),
      fetchCSV('calibration_split_stability.csv'),
    ]);
    return { b5, c1, c4, stability };
  });

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;

  const { b5, c1, c4, stability } = data;

  const TABS = [
    { key: 'b5', label: 'Structural rule' },
    { key: 'c1', label: 'Recalibration' },
    { key: 'c4', label: 'Trajectory' },
    { key: 'stab', label: 'Split stability' },
  ];

  // --- B5 data ---
  const b5Splits = ['train', 'val', 'test'].map(s => {
    const d = b5.phase2.splits[s];
    const naive = d.brier_naive_struct ?? d.brier_naive ?? 0;
    const rule = d.brier_rule_struct ?? d.brier_rule ?? 0;
    return {
      split: s,
      n: d.n_struct ?? d.n_bets ?? 0,
      naive: naive.toFixed(3),
      rule: rule.toFixed(3),
      delta: fmtDelta(rule - naive),
      pnl: fmtMoney(d.pnl_total ?? 0),
    };
  });

  // --- C1 data (test split) ---
  const c1Test = c1.filter(r => r.split === 'test');
  const c1Naive = c1Test.find(r => r.model && r.model.includes('naive'));
  const c1Rows = c1Test.map(r => ({
    model: r.model,
    brier: r.brier != null ? r.brier.toFixed(4) : '\u2014',
    rel: r.reliability != null ? r.reliability.toFixed(4) : '\u2014',
    ece: r.ece != null ? r.ece.toFixed(4) : '\u2014',
    d: (c1Naive && r.model !== c1Naive.model && r.brier != null && c1Naive.brier != null)
      ? fmtDelta(r.brier - c1Naive.brier)
      : '\u2014',
  }));

  // --- C4 data ---
  const tr = c4.test_results;
  const c4Rows = [
    { model: 'naive (market price)', brier: tr.naive_brier.toFixed(4), rel: '\u2014', ece: '\u2014' },
  ];
  if (tr.logistic_trajectory) c4Rows.push({ model: 'logistic_trajectory', brier: tr.logistic_trajectory.brier.toFixed(4), rel: tr.logistic_trajectory.reliability.toFixed(4), ece: tr.logistic_trajectory.ece.toFixed(4) });
  if (tr.xgb_trajectory) c4Rows.push({ model: 'xgb_trajectory', brier: tr.xgb_trajectory.brier.toFixed(4), rel: tr.xgb_trajectory.reliability.toFixed(4), ece: tr.xgb_trajectory.ece.toFixed(4) });
  if (tr.hybrid) {
    const w = Math.round((tr.hybrid.weight ?? 0.45) * 100);
    const hybridKey = `hybrid_${w}_${100 - w}`;
    MODEL_INFO[hybridKey] = {
      name: `Hybrid blend (${w}/${100 - w})`,
      tip: `Combines XGBoost predictions (${w}%) with the raw market price (${100 - w}%). Anchoring to the crowd prevents the ML model from drifting too far.`,
    };
    c4Rows.push({ model: hybridKey, brier: tr.hybrid.brier.toFixed(4), rel: tr.hybrid.reliability.toFixed(4), ece: tr.hybrid.ece.toFixed(4) });
  }

  const c4Improvement = tr.hybrid ? ((tr.naive_brier - tr.hybrid.brier) / tr.naive_brier * 100).toFixed(1) : '0';
  const c4DeltaAbs = tr.hybrid ? Math.abs(tr.hybrid.brier - tr.naive_brier).toFixed(4) : '0';

  // --- Split stability ---
  const categories = [...new Set(stability.map(r => r.category))];
  const splitData = categories
    .map(cat => {
      const train = stability.find(r => r.category === cat && r.split === 'train');
      const test = stability.find(r => r.category === cat && r.split === 'test');
      return { label: catName(cat), train: train?.brier ?? 0, test: test?.brier ?? 0 };
    })
    .sort((a, b) => a.train - b.train)
    .slice(0, 8);

  return (
    <div className="mm-page">
      <div className="mm-eyebrow">Page &middot; exploitation attempts</div>
      <h1 className="mm-h1">Four attempts to beat the market</h1>
      <p className="mm-lede">
        We tried four independent approaches to systematically outperform Polymarket's
        crowd-sourced probabilities. All failed.
      </p>
      <TabBar tabs={TABS} active={tab} onChange={setTab}/>

      {tab === 'b5' && <div className="mm-pane">
        <h3 className="mm-h3">Walk-forward structural hypothesis</h3>
        <p>
          <strong>Hypothesis:</strong> Long-horizon, low-liquidity binary markets
          systematically under-price YES outcomes. Apply a blanket alpha-shift to correct.
        </p>
        <DataTable
          columns={[
            { key: 'split', label: 'Split' },
            { key: 'n', label: 'Struct markets', numeric: true },
            { key: 'naive', label: 'Naive Brier', numeric: true },
            { key: 'rule', label: 'Rule Brier', numeric: true },
            { key: 'delta', label: '\u0394', numeric: true },
            { key: 'pnl', label: 'P&L ($100/bet)', numeric: true },
          ]}
          rows={b5Splits}
        />
        <Callout kind="danger" eyebrow="Result &middot; failed">
          The rule makes Brier worse on every split. YES under-pricing is real but a blanket
          alpha-shift over-corrects the majority of structural-pop markets.
        </Callout>
      </div>}

      {tab === 'c1' && <div className="mm-pane">
        <h3 className="mm-h3">Calibration correction models</h3>
        <p>Train recalibration models (isotonic, Platt, per-category) to correct the documented favourite-longshot bias.</p>
        <DataTable
          columns={[
            { key: 'model', label: 'Model', format: v => modelLabel(v) },
            { key: 'brier', label: 'Brier', numeric: true },
            { key: 'rel', label: 'Reliability', numeric: true },
            { key: 'ece', label: 'ECE', numeric: true },
            { key: 'd', label: '\u0394 vs naive', numeric: true },
          ]}
          rows={c1Rows}
        />
        <Callout kind="warn" eyebrow="Result &middot; negligible">
          Best model improves Brier by less than 0.001. The market's calibration is
          already near-optimal. More granular models overfit and make things <em>worse</em>.
        </Callout>
      </div>}

      {tab === 'c4' && <div className="mm-pane">
        <h3 className="mm-h3">Price trajectory dynamics</h3>
        <p>Engineer features from <em>how</em> prices evolve (staleness, volatility regime, path curvature) and blend XGBoost with the market price.</p>
        <DataTable
          columns={[
            { key: 'model', label: 'Model', format: v => modelLabel(v) },
            { key: 'brier', label: 'Brier', numeric: true },
            { key: 'rel', label: 'Reliability', numeric: true },
            { key: 'ece', label: 'ECE', numeric: true },
          ]}
          rows={c4Rows}
        />
        <Callout kind="warn" eyebrow={'Result \u00b7 small improvement (' + c4Improvement + '%)'}>
          The hybrid improves test Brier by {c4DeltaAbs}. But standalone XGB is worse than naive.
          It only works when heavily anchored to the market. Key insight: staleness predicts
          <em> accuracy</em>, not mispricing.
        </Callout>
      </div>}

      {tab === 'stab' && <div className="mm-pane">
        <h3 className="mm-h3">Split stability &middot; which findings generalize?</h3>
        <div className="mm-card">
          <div className="mm-card-h">
            <h3 className="mm-h3" style={{margin:0, fontSize:15}}>Brier by split &middot; train vs test</h3>
            <div className="mm-legend">
              <span><span className="mm-dot" style={{background:'#4F5BD5'}}/>Train</span>
              <span><span className="mm-dot" style={{background:'#E44D3A'}}/>Test</span>
            </div>
          </div>
          <SplitBars data={splitData}/>
        </div>
        <Callout kind="info" eyebrow="Takeaway">
          Categories with similar train and test bars are stable. Large divergences indicate
          fragile findings that should not be over-interpreted.
        </Callout>
      </div>}
    </div>
  );
}

/* =========================================================================
   Data deep dive
   ========================================================================= */
function DataPage() {
  const { data, error, loading } = useData(async () => {
    const [cats, brier] = await Promise.all([
      fetchCSV('calibration_by_category.csv'),
      fetchCSV('brier_decomposition.csv'),
    ]);
    return { cats, brier };
  });

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;

  const { cats, brier } = data;
  const allRow = brier.find(r => r.split === 'all');
  const splits = brier.filter(r => r.split !== 'all');

  // Markets by category for bar chart
  const byCat = cats
    .filter(r => r.category && r.n_markets)
    .sort((a, b) => b.n_markets - a.n_markets)
    .map(r => ({ label: catName(r.category), v: r.n_markets }));

  // Volume by year (from raw parquet, not available as CSV, hardcoded)
  const byYear = [
    { year: 2022, markets: 214, volume: 0.08e9 },
    { year: 2023, markets: 402, volume: 0.31e9 },
    { year: 2024, markets: 1873, volume: 2.42e9 },
    { year: 2025, markets: 1608, volume: 3.18e9 },
    { year: 2026, markets: 441, volume: 0.64e9 },
  ];

  // Split rows
  const splitRows = splits.map(r => ({
    split: r.split,
    markets: r.n_markets != null ? r.n_markets.toLocaleString() : '\u2014',
    snapshots: r.n != null ? r.n.toLocaleString() : '\u2014',
    base: r.base_rate != null ? r.base_rate.toFixed(4) : '\u2014',
  }));

  return (
    <div className="mm-page">
      <div className="mm-eyebrow">Page &middot; data deep dive</div>
      <h1 className="mm-h1">Data deep dive</h1>
      <p className="mm-lede">
        {allRow.n_markets.toLocaleString()} resolved binary Polymarket markets fetched via
        the Gamma API, snapshotted at 12-hour intervals between January 2022 and April 2026.
        Categories assigned by tag mapping. Splits are strictly temporal to prevent leakage.
      </p>

      <div className="mm-grid-4">
        <MetricTile label="Markets fetched" value="5,217" sub="before volume filter"/>
        <MetricTile label="After filtering" value={allRow.n_markets.toLocaleString()} sub={'\u2265 $1M volume'}/>
        <MetricTile label="Categories" value={byCat.length.toString()} sub="by tag mapping"/>
        <MetricTile label="Snapshots" value={allRow.n.toLocaleString()} sub="12-hour interval"/>
      </div>

      <SectionHeader title="Markets by category" description="Sports and politics dominate; entertainment and social-media are under-sampled and statistically fragile."/>
      <div className="mm-card">
        <CategoryBar data={byCat}/>
      </div>

      <SectionHeader title="Markets and volume by year" description="Polymarket's volume exploded in 2024 around the US election cycle."/>
      <div className="mm-card">
        <div className="mm-card-h">
          <h3 className="mm-h3" style={{margin:0, fontSize:15}}>Markets (bars) vs total volume (line)</h3>
          <div className="mm-legend">
            <span><span className="mm-dot" style={{background:'#4F5BD5'}}/>Markets</span>
            <span><span className="mm-dot" style={{background:'#E44D3A'}}/>Volume ($B)</span>
          </div>
        </div>
        <VolumeOverTime data={byYear}/>
      </div>

      <SectionHeader title="Train / val / test splits" description="Event-group splits prevent correlated-market leakage."/>
      <DataTable
        columns={[
          { key: 'split', label: 'Split' },
          { key: 'markets', label: 'Markets', numeric: true },
          { key: 'snapshots', label: 'Snapshots', numeric: true },
          { key: 'base', label: 'Base rate', numeric: true },
        ]}
        rows={splitRows}
      />

      <Callout kind="info" eyebrow="Data provenance">
        Source: Polymarket Gamma API (market metadata) + CLOB API (price history).
        External joins available: FRED macroeconomic series, Fed Funds futures (yfinance),
        FOMC / CPI calendar, GDELT sentiment. All cached under <code>data/external/</code>.
      </Callout>
    </div>
  );
}

Object.assign(window, { OverviewPage, CalibrationPage, FLBiasPage, ExploitPage, DataPage });
