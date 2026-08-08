import { useState, useRef, useEffect } from 'react'

export default function Chat({ messages, onSend, onStop, onClear, streaming, serverState, modelState }) {
  const [input, setInput] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const canChat = serverState === 'online' && modelState === 'loaded'

  const submit = (e) => {
    e.preventDefault()
    if (!input.trim() || !canChat) return
    onSend(input.trim())
    setInput('')
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(e) }
  }

  const emptyMsg = () => {
    if (serverState === 'offline') return 'Server is not running. Start llama-server and click Check server.'
    if (serverState === 'loading') return 'Model is loading...'
    if (serverState === 'unknown') return 'Click Check server to connect.'
    if (modelState !== 'loaded') return 'No model loaded. Load a model via /models/load to start chatting.'
    return 'Start a conversation'
  }

  return (
    <div style={S.wrap}>
      <div style={S.messages}>
        {messages.length === 0 && (
          <div style={S.empty}>
            <span style={S.emptyIcon}>{canChat ? '\u26A1' : '\u23F3'}</span>
            <p>{emptyMsg()}</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={S.msg(m.role)}>
            <div style={S.msgLabel(m.role)}>{m.role}</div>
            <pre style={S.msgText}>{m.content}</pre>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} style={S.bar}>
        {streaming && <button type="button" onClick={onStop} style={S.stopBtn}>Stop</button>}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={canChat ? 'Type a message...' : 'Waiting for model'}
          disabled={!canChat || streaming}
          style={S.textarea}
          rows={1}
        />
        <button type="submit" disabled={!canChat || streaming || !input.trim()} style={S.sendBtn}>Send</button>
        <button type="button" onClick={onClear} style={S.clearBtn} title="Clear">Clear</button>
      </form>
    </div>
  )
}

const S = {
  wrap: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 },
  messages: { flex: 1, overflowY: 'auto', padding: 16 },
  empty: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#5f6d80', gap: 8 },
  emptyIcon: { fontSize: 32 },
  msg: (r) => ({ marginBottom: 12, padding: '8px 12px', borderRadius: 6, background: r === 'user' ? '#1a2332' : '#131820', borderLeft: r === 'user' ? '2px solid #39baec' : '2px solid #2d3b4f' }),
  msgLabel: (r) => ({ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.8, color: r === 'user' ? '#39baec' : '#5f6d80', marginBottom: 4 }),
  msgText: { fontFamily: '"SF Mono", "Cascadia Code", "Consolas", monospace', fontSize: 13, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0, lineHeight: 1.6 },
  bar: { display: 'flex', gap: 6, padding: 12, borderTop: '1px solid #1f2733', background: '#0d1117' },
  textarea: { flex: 1, resize: 'none', fontFamily: 'inherit', lineHeight: 1.4 },
  sendBtn: { padding: '6px 16px' },
  stopBtn: { padding: '6px 12px', background: '#3a1a1a', color: '#f07178', borderColor: '#5a2a2a' },
  clearBtn: { padding: '6px 10px', fontSize: 11 },
}
