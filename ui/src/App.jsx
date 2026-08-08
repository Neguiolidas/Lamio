import { useState, useRef, useCallback, useEffect } from 'react'
import Sidebar from './Sidebar'
import Chat from './Chat'
import Telemetry from './Telemetry'

export default function App() {
  const [config, setConfig] = useState({
    model: '',
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    repeat_penalty: 1.1,
    n_predict: 256,
    tier_budget: 4096,
    expert_k: 0,
    ngl: 0,
    threads: 4,
    ctx: 2048,
  })
  const [serverState, setServerState] = useState('unknown')
  const [modelState, setModelState] = useState('none')
  const [availableModels, setAvailableModels] = useState([])
  const [tierStats, setTierStats] = useState(null)
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef(null)

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch('/v1/models')
      if (r.ok) {
        const d = await r.json()
        const models = d.data || []
        setAvailableModels(models)
        if (models.length > 0) {
          setModelState('loaded')
          if (!config.model) {
            setConfig(c => ({ ...c, model: models[0].id }))
          }
        } else {
          setModelState('empty')
        }
        setServerState('online')
      } else if (r.status === 503) {
        setServerState('loading')
      } else {
        setServerState('offline')
      }
    } catch {
      setServerState('offline')
    }
  }, [config.model])

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 5000)
    return () => clearInterval(id)
  }, [checkHealth])

  const fetchTier = useCallback(async () => {
    try {
      const r = await fetch('/lamio/tier-stats')
      if (r.ok) {
        const d = await r.json()
        if (d.enabled) {
          setTierStats(d)
        } else {
          setTierStats(null)
        }
      }
    } catch {}
  }, [])

  useEffect(() => {
    if (serverState === 'online') fetchTier()
    const id = setInterval(() => {
      if (serverState === 'online') fetchTier()
    }, 3000)
    return () => clearInterval(id)
  }, [serverState, fetchTier])

  const send = useCallback(async (content) => {
    if (streaming || !content.trim() || modelState !== 'loaded') return
    const userMsg = { role: 'user', content }
    const next = [...messages, userMsg]
    setMessages(next)
    setStreaming(true)

    const assistantMsg = { role: 'assistant', content: '' }
    setMessages([...next, assistantMsg])

    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      const r = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.model,
          messages: next.map(m => ({ role: m.role, content: m.content })),
          temperature: config.temperature,
          top_k: config.top_k,
          top_p: config.top_p,
          repeat_penalty: config.repeat_penalty,
          n_predict: config.n_predict,
          stream: true,
        }),
        signal: ctrl.signal,
      })

      if (!r.ok) throw new Error(`HTTP ${r.status}`)

      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') break
          try {
            const j = JSON.parse(data)
            const tok = j.choices?.[0]?.delta?.content
            if (tok) {
              assistantMsg.content += tok
              setMessages([...next, { ...assistantMsg }])
            }
          } catch {}
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        assistantMsg.content = `Error: ${e.message}`
        setMessages([...next, { ...assistantMsg }])
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
      fetchTier()
    }
  }, [messages, streaming, config, modelState, fetchTier])

  const stop = useCallback(() => { abortRef.current?.abort() }, [])
  const clear = useCallback(() => { setMessages([]) }, [])

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        config={config}
        setConfig={setConfig}
        serverState={serverState}
        modelState={modelState}
        availableModels={availableModels}
        onHealth={checkHealth}
        onTierFetch={fetchTier}
      />
      <Chat
        messages={messages}
        onSend={send}
        onStop={stop}
        onClear={clear}
        streaming={streaming}
        serverState={serverState}
        modelState={modelState}
      />
      <Telemetry
        stats={tierStats}
        onRefresh={fetchTier}
        serverState={serverState}
      />
    </div>
  )
}
