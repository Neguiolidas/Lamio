import { useState, useRef, useEffect } from 'react'

export default function Chat({ messages, onSend, onStop, onClear, streaming, connected }) {
  const [input, setInput] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    onSend(input.trim())
    setInput('')
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(e)
    }
  }

  return (
    <div style={styles.wrap}>
      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <span style={styles.emptyIcon}>&#9889;</span>
            <p>Start a conversation with the model</p>
            <p style={styles.emptyHint}>Make sure the server is running and connected</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={styles.msg(m.role)}>
            <div style={styles.msgLabel(m.role)}>{m.role}</div>
            <pre style={styles.msgText}>{m.content}</pre>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} style={styles.bar}>
        {streaming && (
          <button type="button" onClick={onStop} style={styles.stopBtn}>Stop</button>
        )}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={connected ? 'Type a message...' : 'Connect to server first'}
          disabled={!connected || streaming}
          style={styles.textarea}
          rows={1}
        />
        <button type="submit" disabled={!connected || streaming || !input.trim()} style={styles.sendBtn}>
          Send
        </button>
        <button type="button" onClick={onClear} style={styles.clearBtn} title="Clear chat">
          Clear
        </button>
      </form>
    </div>
  )
}

const styles = {
  wrap: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minWidth: 0,
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: 16,
  },
  empty: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#5f6d80',
    gap: 4,
  },
  emptyIcon: { fontSize: 32 },
  emptyHint: { fontSize: 12, color: '#3a4a5e' },
  msg: (role) => ({
    marginBottom: 12,
    padding: '8px 12px',
    borderRadius: 6,
    background: role === 'user' ? '#1a2332' : '#131820',
    borderLeft: role === 'user' ? '2px solid #39baec' : '2px solid #2d3b4f',
  }),
  msgLabel: (role) => ({
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    color: role === 'user' ? '#39baec' : '#5f6d80',
    marginBottom: 4,
  }),
  msgText: {
    fontFamily: '"SF Mono", "Cascadia Code", "Consolas", monospace',
    fontSize: 13,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    margin: 0,
    lineHeight: 1.6,
  },
  bar: {
    display: 'flex',
    gap: 6,
    padding: 12,
    borderTop: '1px solid #1f2733',
    background: '#0d1117',
  },
  textarea: {
    flex: 1,
    resize: 'none',
    fontFamily: 'inherit',
    lineHeight: 1.4,
  },
  sendBtn: { padding: '6px 16px' },
  stopBtn: {
    padding: '6px 12px',
    background: '#3a1a1a',
    color: '#f07178',
    borderColor: '#5a2a2a',
  },
  clearBtn: { padding: '6px 10px', fontSize: 11 },
}
