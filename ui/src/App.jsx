import { useState, useRef, useCallback, useEffect } from 'react'
import Sidebar from './Sidebar'
import Chat from './Chat'
import Telemetry from './Telemetry'

const REASONING_FORMATS = ['deepseek', 'deepseek-legacy', 'gpt-oss']
const REASONING_EFFORTS = ['none', 'low', 'medium', 'high']

export default function App() {
  const [config, setConfig] = useState({
    model: '',
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    repeat_penalty: 1.1,
    n_predict: 256,
    reasoning_format: 'deepseek',
    reasoning_effort: 'medium',
  })
  const [serverState, setServerState] = useState('unknown')
  const [modelStatus, setModelStatus] = useState({})
  const [availableModels, setAvailableModels] = useState([])
  const [serverCtx, setServerCtx] = useState(null)
  const [tierStats, setTierStats] = useState(null)
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [modelAction, setModelAction] = useState({ type: null, model: null })
  const abortRef = useRef(null)

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch('/v1/models')
      if (r.ok) {
        const d = await r.json()
        const models = d.data || []
        setAvailableModels(models)

        const status = {}
        let hasLoaded = false
        for (const m of models) {
          const st = m.status?.value || m.status || 'unknown'
          status[m.id] = st
          if (st === 'loaded' || st === 'running') hasLoaded = true
        }
        setModelStatus(status)
        setServerState(hasLoaded ? 'online' : 'ready')

        if (!config.model && models.length > 0) {
          const first = models.find(m => !m.id.startsWith('ggml-vocab')) || models[0]
          setConfig(c => ({ ...c, model: first.id }))
        }

        const ctx = models.find(m => m.id === config.model)?.meta?.n_ctx
        if (ctx) setServerCtx(ctx)
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
        setTierStats(d.enabled ? d : null)
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

  const loadModel = useCallback(async (modelId) => {
    if (!modelId || modelAction.type) return
    setModelAction({ type: 'load', model: modelId })
    try {
      const r = await fetch('/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      })
      if (!r.ok) {
        const e = await r.json().catch(() => ({}))
        console.error('load failed:', e.error?.message || r.status)
      } else {
        setTimeout(checkHealth, 1000)
      }
    } catch (e) {
      console.error('loadModel:', e.message)
    } finally {
      setModelAction({ type: null, model: null })
    }
  }, [modelAction, checkHealth])

  const unloadModel = useCallback(async (modelId) => {
    if (!modelId || modelAction.type) return
    setModelAction({ type: 'unload', model: modelId })
    try {
      const r = await fetch('/models/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      })
      if (!r.ok) {
        const e = await r.json().catch(() => ({}))
        console.error('unload failed:', e.error?.message || r.status)
      } else {
        setTimeout(checkHealth, 1000)
      }
    } catch (e) {
      console.error('unloadModel:', e.message)
    } finally {
      setModelAction({ type: null, model: null })
    }
  }, [modelAction, checkHealth])

  const currentModelStatus = modelStatus[config.model]
  const modelReady = currentModelStatus === 'loaded' || currentModelStatus === 'running'

  const send = useCallback(async (content) => {
    if (streaming || !content.trim() || !modelReady) return
    const userMsg = { role: 'user', content }
    const next = [...messages, userMsg]
    setMessages(next)
    setStreaming(true)

    const assistantMsg = { role: 'assistant', content: '', reasoning: '' }
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
          reasoning_format: config.reasoning_format,
          reasoning_effort: config.reasoning_effort,
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
            const delta = j.choices?.[0]?.delta
            if (!delta) continue
            const tok = delta.content
            const reasonTok = delta.reasoning_content || delta.reasoning
            if (reasonTok) assistantMsg.reasoning += reasonTok
            if (tok) assistantMsg.content += tok
            setMessages([...next, { ...assistantMsg }])
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
  }, [messages, streaming, config, modelReady, fetchTier])

  const stop = useCallback(() => { abortRef.current?.abort() }, [])

  const newSession = useCallback(() => {
    if (streaming) abortRef.current?.abort()
    setMessages([])
  }, [streaming])

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        config={config}
        setConfig={setConfig}
        serverState={serverState}
        modelStatus={modelStatus}
        availableModels={availableModels}
        onHealth={checkHealth}
        onTierFetch={fetchTier}
        onLoadModel={loadModel}
        onUnloadModel={unloadModel}
        modelAction={modelAction}
        serverCtx={serverCtx}
      />
      <Chat
        messages={messages}
        onSend={send}
        onStop={stop}
        onNewSession={newSession}
        streaming={streaming}
        modelReady={modelReady}
        serverState={serverState}
        modelName={config.model}
        serverCtx={serverCtx}
      />
      <Telemetry
        stats={tierStats}
        onRefresh={fetchTier}
        serverState={serverState}
      />
    </div>
  )
}
