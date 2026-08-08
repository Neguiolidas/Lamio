function Stat({ label, value, color }) {
  return (
    <div style={styles.stat}>
      <span style={styles.statLabel}>{label}</span>
      <span style={{ ...styles.statValue, color: color || '#c5cdd8' }}>{value}</span>
    </div>
  )
}

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div style={styles.barBg}>
      <div style={{ ...styles.barFill, width: `${pct}%`, background: color || '#39baec' }} />
    </div>
  )
}

export default function Telemetry({ stats, onRefresh, connected }) {
  const hasData = stats && stats.hits !== undefined
  const total = hasData ? stats.hits + stats.misses : 0
  const hitRate = total > 0 ? ((stats.hits / total) * 100).toFixed(1) : '0.0'

  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <h2 style={styles.title}>Tier Cache</h2>
        <button onClick={onRefresh} style={styles.refreshBtn} title="Refresh">
          Refresh
        </button>
      </div>

      {!connected && (
        <p style={styles.empty}>Server offline</p>
      )}

      {connected && !hasData && (
        <p style={styles.empty}>No telemetry yet. Run inference or click refresh.</p>
      )}

      {hasData && (
        <>
          <div style={styles.section}>
            <Stat label="Hit rate" value={`${hitRate}%`} color={Number(hitRate) > 50 ? '#6fc46e' : '#ffb454'} />
            <Bar value={Number(hitRate)} max={100} color={Number(hitRate) > 50 ? '#50a14f' : '#ffb454'} />
          </div>

          <div style={styles.grid}>
            <Stat label="Hits" value={stats.hits} color="#6fc46e" />
            <Stat label="Misses" value={stats.misses} color="#f07178" />
            <Stat label="Evictions" value={stats.evictions || 0} color="#ffb454" />
          </div>

          <div style={styles.divider} />

          <div style={styles.section}>
            <Stat label="Bytes loaded" value={`${(stats.bytes_loaded / 1048576).toFixed(2)} MB`} />
            <Stat label="Load time" value={`${(stats.load_time_ms || 0).toFixed(1)} ms`} />
          </div>

          <div style={styles.divider} />

          <div style={styles.section}>
            <Stat label="Used" value={`${(stats.used_mb || 0).toFixed(1)} MB`} />
            <Stat label="Capacity" value={`${(stats.capacity_mb || 0).toFixed(1)} MB`} />
            <Bar value={stats.used_mb || 0} max={stats.capacity_mb || 1} color="#39baec" />
          </div>

          <div style={styles.divider} />

          <div style={styles.section}>
            <h3 style={styles.subTitle}>Cache breakdown</h3>
            <Stat label="Layers" value={stats.n_layers || '-'} />
            <Stat label="Experts/layer" value={stats.n_expert || '-'} />
            <Stat label="Active experts" value={stats.n_expert_used || '-'} />
            <Stat label="Tensor types" value={stats.n_type_idx || 3} />
          </div>
        </>
      )}
    </div>
  )
}

const styles = {
  wrap: {
    width: 240,
    minWidth: 240,
    background: '#131820',
    borderLeft: '1px solid #1f2733',
    padding: 12,
    overflowY: 'auto',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  title: {
    fontSize: 14,
    color: '#39baec',
    fontWeight: 600,
  },
  refreshBtn: {
    fontSize: 11,
    padding: '3px 8px',
  },
  empty: {
    color: '#5f6d80',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 24,
  },
  section: {
    marginBottom: 10,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: 6,
    marginTop: 8,
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    gap: 1,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 10,
    color: '#5f6d80',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  statValue: {
    fontSize: 15,
    fontWeight: 600,
    fontFamily: '"SF Mono", "Cascadia Code", "Consolas", monospace',
  },
  barBg: {
    height: 4,
    background: '#0d1117',
    borderRadius: 2,
    marginTop: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 2,
    transition: 'width 0.3s',
  },
  divider: {
    height: 1,
    background: '#1f2733',
    margin: '10px 0',
  },
  subTitle: {
    fontSize: 11,
    color: '#5f6d80',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 6,
  },
}
