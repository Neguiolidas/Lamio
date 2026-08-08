export default function Sidebar({ config, setConfig, connected, onHealth, onTierFetch }) {
  const set = (k) => (e) => setConfig(c => ({ ...c, [k]: e.target.type === 'number' ? Number(e.target.value) : e.target.value }))

  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <h1 style={styles.logo}>Lamio</h1>
        <span style={styles.badge(connected)}>
          {connected ? 'ON' : 'OFF'}
        </span>
      </div>

      <button onClick={onHealth} style={styles.btn}>Check server</button>
      <button onClick={onTierFetch} style={{ ...styles.btn, marginTop: 4 }}>Fetch tier stats</button>

      <div style={styles.divider} />

      <label style={styles.label}>Model</label>
      <input value={config.model} onChange={set('model')} style={styles.input} placeholder="model-id" />

      <label style={styles.label}>Threads</label>
      <input type="number" value={config.threads} onChange={set('threads')} style={styles.input} min={1} max={64} />

      <label style={styles.label}>GPU layers (-ngl)</label>
      <input type="number" value={config.ngl} onChange={set('ngl')} style={styles.input} min={0} max={999} />

      <label style={styles.label}>Context</label>
      <input type="number" value={config.ctx} onChange={set('ctx')} style={styles.input} min={64} max={65536} />

      <div style={styles.divider} />

      <label style={styles.label}>Temperature</label>
      <input type="number" value={config.temperature} onChange={set('temperature')} style={styles.input} step={0.1} min={0} max={2} />

      <label style={styles.label}>Top-K</label>
      <input type="number" value={config.top_k} onChange={set('top_k')} style={styles.input} min={1} max={200} />

      <label style={styles.label}>Top-P</label>
      <input type="number" value={config.top_p} onChange={set('top_p')} style={styles.input} step={0.05} min={0} max={1} />

      <label style={styles.label}>Repeat penalty</label>
      <input type="number" value={config.repeat_penalty} onChange={set('repeat_penalty')} style={styles.input} step={0.05} min={1} max={2} />

      <label style={styles.label}>Max tokens</label>
      <input type="number" value={config.n_predict} onChange={set('n_predict')} style={styles.input} min={1} max={4096} />

      <div style={styles.divider} />

      <h3 style={styles.sectionTitle}>MoE Tiering</h3>

      <label style={styles.label}>Budget (MiB)</label>
      <input type="number" value={config.tier_budget} onChange={set('tier_budget')} style={styles.input} min={0} max={65536} />

      <label style={styles.label}>Expert top-K override</label>
      <input type="number" value={config.expert_k} onChange={set('expert_k')} style={styles.input} min={0} max={256} />
      <span style={styles.hint}>0 = model default</span>
    </div>
  )
}

const styles = {
  wrap: {
    width: 240,
    minWidth: 240,
    background: '#131820',
    borderRight: '1px solid #1f2733',
    padding: 12,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  logo: {
    fontSize: 20,
    fontWeight: 700,
    color: '#39baec',
    letterSpacing: -0.5,
  },
  badge: (on) => ({
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 3,
    background: on ? '#1a3a1a' : '#3a1a1a',
    color: on ? '#6fc46e' : '#f07178',
  }),
  btn: {
    width: '100%',
    padding: '6px 0',
    fontSize: 12,
  },
  divider: {
    height: 1,
    background: '#1f2733',
    margin: '10px 0',
  },
  sectionTitle: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
    color: '#5f6d80',
    marginBottom: 6,
  },
  label: {
    fontSize: 11,
    color: '#5f6d80',
    marginTop: 6,
    display: 'block',
  },
  input: {
    width: '100%',
    marginTop: 2,
    padding: '4px 8px',
    fontSize: 13,
  },
  hint: {
    fontSize: 10,
    color: '#3a4a5e',
  },
}
