export default function Sidebar({ config, setConfig, serverState, modelState, availableModels, onHealth, onTierFetch }) {
  const set = (k) => (e) => setConfig(c => ({ ...c, [k]: e.target.type === 'number' ? Number(e.target.value) : e.target.value }))

  const stateColor = {
    online: '#6fc46e',
    loading: '#ffb454',
    offline: '#f07178',
    unknown: '#5f6d80',
  }

  const modelLabel = {
    loaded: 'Loaded',
    empty: 'No model',
    none: 'No model',
  }

  return (
    <div style={S.wrap}>
      <div style={S.header}>
        <h1 style={S.logo}>Lamio</h1>
        <span style={S.stateDot(serverState)} title={`Server: ${serverState}`} />
      </div>

      <div style={S.statusRow}>
        <span style={S.statusLabel}>Server</span>
        <span style={{ ...S.statusValue, color: stateColor[serverState] }}>{serverState}</span>
      </div>
      <div style={S.statusRow}>
        <span style={S.statusLabel}>Model</span>
        <span style={{ ...S.statusValue, color: modelState === 'loaded' ? '#6fc46e' : '#5f6d80' }}>
          {modelLabel[modelState] || modelState}
        </span>
      </div>

      <div style={S.divider} />

      <button onClick={onHealth} style={S.btn}>Check server</button>
      <button onClick={onTierFetch} style={{ ...S.btn, marginTop: 4 }}>Refresh tier</button>

      <div style={S.divider} />

      <label style={S.label}>Model</label>
      {availableModels.length > 0 ? (
        <select value={config.model} onChange={set('model')} style={S.input}>
          {availableModels.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}
        </select>
      ) : (
        <input value={config.model} onChange={set('model')} style={S.input} placeholder="no model loaded" />
      )}

      <div style={S.divider} />

      <label style={S.label}>Threads</label>
      <input type="number" value={config.threads} onChange={set('threads')} style={S.input} min={1} max={64} />

      <label style={S.label}>GPU layers (-ngl)</label>
      <input type="number" value={config.ngl} onChange={set('ngl')} style={S.input} min={0} max={999} />

      <label style={S.label}>Context</label>
      <input type="number" value={config.ctx} onChange={set('ctx')} style={S.input} min={64} max={65536} />

      <div style={S.divider} />

      <label style={S.label}>Temperature</label>
      <input type="number" value={config.temperature} onChange={set('temperature')} style={S.input} step={0.1} min={0} max={2} />

      <label style={S.label}>Top-K</label>
      <input type="number" value={config.top_k} onChange={set('top_k')} style={S.input} min={1} max={200} />

      <label style={S.label}>Top-P</label>
      <input type="number" value={config.top_p} onChange={set('top_p')} style={S.input} step={0.05} min={0} max={1} />

      <label style={S.label}>Repeat penalty</label>
      <input type="number" value={config.repeat_penalty} onChange={set('repeat_penalty')} style={S.input} step={0.05} min={1} max={2} />

      <label style={S.label}>Max tokens</label>
      <input type="number" value={config.n_predict} onChange={set('n_predict')} style={S.input} min={1} max={4096} />

      <div style={S.divider} />

      <h3 style={S.sectionTitle}>MoE Tiering</h3>
      <label style={S.label}>Budget (MiB)</label>
      <input type="number" value={config.tier_budget} onChange={set('tier_budget')} style={S.input} min={0} max={65536} />
      <label style={S.label}>Expert top-K override</label>
      <input type="number" value={config.expert_k} onChange={set('expert_k')} style={S.input} min={0} max={256} />
      <span style={S.hint}>0 = model default</span>
    </div>
  )
}

const S = {
  wrap: { width: 240, minWidth: 240, background: '#131820', borderRight: '1px solid #1f2733', padding: 12, overflowY: 'auto' },
  header: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 },
  logo: { fontSize: 20, fontWeight: 700, color: '#39baec', letterSpacing: -0.5 },
  stateDot: (s) => ({
    width: 8, height: 8, borderRadius: '50%',
    background: { online: '#6fc46e', loading: '#ffb454', offline: '#f07178', unknown: '#5f6d80' }[s] || '#5f6d80',
  }),
  statusRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 0' },
  statusLabel: { fontSize: 11, color: '#5f6d80' },
  statusValue: { fontSize: 12, fontWeight: 600 },
  btn: { width: '100%', padding: '6px 0', fontSize: 12 },
  divider: { height: 1, background: '#1f2733', margin: '10px 0' },
  sectionTitle: { fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, color: '#5f6d80', marginBottom: 6 },
  label: { fontSize: 11, color: '#5f6d80', marginTop: 6, display: 'block' },
  input: { width: '100%', marginTop: 2, padding: '4px 8px', fontSize: 13 },
  hint: { fontSize: 10, color: '#3a4a5e' },
}
