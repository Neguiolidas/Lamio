let serverState = 'connecting';
let config = { model: '', temperature: 0.7, top_k: 40, top_p: 0.9, repeat_penalty: 1.1, n_predict: 128, seed: -1 };
let messageHistory = [];
let streaming = false;
let abortCtrl = null;

const $ = (id) => document.getElementById(id);

function toggleSidebar() {
  $('sidebar').classList.toggle('collapsed');
}

function switchView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  $('view-' + name).classList.add('active');
  document.querySelector(`.nav-item[data-view="${name}"]`).classList.add('active');
  $('topbarTitle').textContent = name.charAt(0).toUpperCase() + name.slice(1);
  if (name === 'telemetry') fetchTelemetry();
  if (name === 'models') fetchModels();
  if (name === 'settings') renderSettings();
  if (name === 'memory') fetchMemory();
}

function setServerState(state) {
  serverState = state;
  const pill = $('serverStatus');
  pill.className = 'status-pill ' + state;
  const label = pill.querySelector('.nav-label');
  if (label) label.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    if (r.ok) {
      const d = await r.json();
      setServerState(d.model_loaded ? 'online' : 'ready');
      if (d.model) {
        $('modelBadge').textContent = d.model;
        $('modelBadge').classList.add('active');
        config.model = d.model;
      }
    } else {
      setServerState('offline');
    }
  } catch {
    setServerState('offline');
  }
}

async function fetchModels() {
  try {
    const r = await fetch('/api/models');
    if (!r.ok) return;
    const models = await r.json();
    const list = $('modelList');
    list.innerHTML = '';
    if (models.length === 0) {
      list.innerHTML = '<p class="muted">No models found in the models directory.</p>';
      return;
    }
    for (const m of models) {
      const card = document.createElement('div');
      card.className = 'model-card' + (m.loaded ? ' loaded' : '');
      card.innerHTML = `
        <div class="model-info">
          <span class="model-name">${m.name}</span>
          <span class="model-meta">${formatBytes(m.size)} | ${m.arch || 'unknown'} | ${m.params || ''}</span>
        </div>
        <div class="model-actions">
          ${m.loaded
            ? `<button class="btn btn-danger" onclick="unloadModel('${m.name}')">Unload</button>`
            : `<button class="btn btn-primary" onclick="loadModel('${m.name}')">Load</button>`
          }
        </div>`;
      list.appendChild(card);
    }
  } catch (e) {
    console.error('fetchModels:', e);
  }
}

async function loadModel(name) {
  try {
    const r = await fetch('/api/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name })
    });
    if (r.ok) {
      checkHealth();
      fetchModels();
    } else {
      const e = await r.json().catch(() => ({}));
      alert(e.error || 'Failed to load model');
    }
  } catch (e) {
    alert('Load failed: ' + e.message);
  }
}

async function unloadModel(name) {
  try {
    const r = await fetch('/api/models/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name })
    });
    if (r.ok) {
      checkHealth();
      fetchModels();
    }
  } catch (e) {
    alert('Unload failed: ' + e.message);
  }
}

async function sendMessage() {
  const input = $('chatInput');
  const text = input.value.trim();
  if (!text || streaming || serverState !== 'online') return;

  messageHistory.push({ role: 'user', content: text });
  renderMessages();
  input.value = '';
  autoResize(input);

  messageHistory.push({ role: 'assistant', content: '' });
  streaming = true;
  $('sendBtn').classList.add('hidden');
  $('stopBtn').classList.remove('hidden');
  renderMessages();

  abortCtrl = new AbortController();
  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: text,
        history: messageHistory.filter(m => m.role !== 'assistant' || m.content).slice(0, -1).map(m => m.content),
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        repeat_penalty: config.repeat_penalty,
        n_predict: config.n_predict,
        seed: config.seed,
      }),
      signal: abortCtrl.signal,
    });

    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const assistant = messageHistory[messageHistory.length - 1];
    assistant.content = data.text || '';
    renderMessages();
  } catch (e) {
    if (e.name !== 'AbortError') {
      const assistant = messageHistory[messageHistory.length - 1];
      assistant.content = 'Error: ' + e.message;
      renderMessages();
    }
  } finally {
    streaming = false;
    abortCtrl = null;
    $('sendBtn').classList.remove('hidden');
    $('stopBtn').classList.add('hidden');
  }
}

function stopGeneration() {
  if (abortCtrl) abortCtrl.abort();
}

function renderMessages() {
  const container = $('chatMessages');
  container.innerHTML = '';
  for (const m of messageHistory) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + m.role;
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (m.role === 'assistant') {
      const role = document.createElement('div');
      role.className = 'msg-role';
      role.textContent = 'Assistant';
      div.appendChild(role);
    }
    const content = document.createElement('div');
    content.textContent = m.content;
    if (streaming && m.role === 'assistant' && m.content === '') {
      content.className = 'typing-cursor';
    }
    bubble.appendChild(content);
    div.appendChild(bubble);
    container.appendChild(div);
  }
  container.scrollTop = container.scrollHeight;
}

function onChatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

async function fetchTelemetry() {
  try {
    const r = await fetch('/api/telemetry');
    if (!r.ok) return;
    const d = await r.json();
    const grid = $('telemetryGrid');
    grid.innerHTML = '';
    const metrics = [
      { label: 'Model', value: d.model || '-', },
      { label: 'RAM (RSS)', value: d.rss_mb || '-', unit: 'MB' },
      { label: 'Tokens/s', value: d.tokens_per_sec?.toFixed(2) || '-' },
      { label: 'Total Tokens', value: d.total_tokens || 0 },
      { label: 'Expert Tensors', value: d.expert_tensors || 0 },
      { label: 'Trunk Buffer', value: d.trunk_mb || '-', unit: 'MB' },
      { label: 'Architecture', value: d.arch || '-' },
      { label: 'Layers', value: d.n_layers || '-' },
      { label: 'Experts', value: d.n_experts || '-', },
    ];
    for (const m of metrics) {
      const card = document.createElement('div');
      card.className = 'metric-card';
      card.innerHTML = `<span class="metric-label">${m.label}</span><span class="metric-value">${m.value}${m.unit ? ' <span class="metric-unit">' + m.unit + '</span>' : ''}</span>`;
      grid.appendChild(card);
    }
  } catch (e) {
    console.error('telemetry:', e);
  }
}

async function fetchMemory() {
  try {
    const r = await fetch('/api/memory');
    if (!r.ok) return;
    const d = await r.json();
    const container = $('memoryContent');
    container.innerHTML = '';
    if (!d.layers || d.layers.length === 0) {
      container.innerHTML = '<p class="muted">No model loaded.</p>';
      return;
    }
    for (const layer of d.layers) {
      const entry = document.createElement('div');
      entry.className = 'layer-entry';
      entry.innerHTML = `Layer ${layer.idx}: type=${layer.type}, KV cache=${layer.kv_entries || 0}, state=${layer.state_size || 0}B`;
      container.appendChild(entry);
    }
  } catch (e) {
    console.error('memory:', e);
  }
}

function renderSettings() {
  const panel = $('settingsPanel');
  panel.innerHTML = '';
  const settings = [
    { key: 'temperature', label: 'Temperature', hint: '0.0 = deterministic, 1.0 = creative', min: 0, max: 2, step: 0.05 },
    { key: 'top_k', label: 'Top-K', hint: 'Only sample from top K tokens (0 = disabled)', min: 0, max: 200, step: 1 },
    { key: 'top_p', label: 'Top-P', hint: 'Nucleus sampling threshold (0 = disabled)', min: 0, max: 1, step: 0.01 },
    { key: 'repeat_penalty', label: 'Repeat Penalty', hint: 'Penalize repeated tokens', min: 1, max: 2, step: 0.05 },
    { key: 'n_predict', label: 'Max Tokens', hint: 'Maximum tokens to generate per response', min: 1, max: 4096, step: 1 },
    { key: 'seed', label: 'Seed', hint: '-1 = random, 0+ = deterministic', min: -1, max: 999999, step: 1 },
  ];
  for (const s of settings) {
    const row = document.createElement('div');
    row.className = 'setting-row';
    row.innerHTML = `
      <label class="setting-label">${s.label}</label>
      <input class="setting-slider" type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${config[s.key]}" oninput="updateSetting('${s.key}', this.value, '${s.step}')">
      <div class="setting-hint"><span id="val-${s.key}">${config[s.key]}</span> | ${s.hint}</div>`;
    panel.appendChild(row);
  }
}

function updateSetting(key, val, step) {
  config[key] = step.includes('.') ? parseFloat(val) : parseInt(val);
  $('val-' + key).textContent = val;
}

function formatBytes(bytes) {
  if (!bytes) return '-';
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + 'KB';
  if (bytes < 1073741824) return (bytes / 1048576).toFixed(0) + 'MB';
  return (bytes / 1073741824).toFixed(2) + 'GB';
}

setInterval(checkHealth, 5000);
checkHealth();
