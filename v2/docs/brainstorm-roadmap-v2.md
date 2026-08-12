# Brainstorm: Lamio v2 -- Roadmap de Implementações Restantes

**Data:** 2026-08-12
**Status:** Forward pass funcional com KV cache + estado recurrente (commit 38ec8e337)
**Modelo de referência:** Qwen3.5-0.8B (13 GDN + 5 full attention, 18 layers)

## PROBLEMA

O Lamio v2 tem forward pass funcional mas não tem utilidade prática: só faz argmax greedy via CLI. Precisa das interfaces de uso (sampling, servidor, REPL) e suporte a modelos maiores (Ornith-9B/35B) para ser um runtime standalone útil.

## WHY

O objetivo do Lamio é ser um fork standalone do llama.cpp com MoE expert tiering. O v2 é a reescrita limpa do forward pass sem o peso do llama.cpp. Sem sampling, servidor e suporte a MoE, o v2 é só uma prova de conceito.

## IN SCOPE

- **Sampling:** temperature, top-k, top-p, repetition penalty
- **Prompt interativo:** REPL stdin (ler token, gerar, imprimir, repetir)
- **Servidor HTTP:** API compatível com OpenAI `/v1/completions`
- **Suporte MoE:** Ornith-9B (denso, baseline) e Ornith-35B (MoE MXFP4, 113 experts)

## OUT OF SCOPE

- Multi-GPU / backend non-CPU (cuda, metal, vulkan)
- Quantização dinâmica (só ler Q4_0/Q8_0/F16 já quantizado no GGUF)
- Fine-tuning / training
- Streaming de tokens via SSE no servidor (primeira versão só response completa)
- Sistema de plugins / extensões

## BANDAS (ordem de implementação)

### Banda 1 -- Usabilidade básica (sampling + REPL)

**Por que primeiro:** Sem sampling variável, não dá para testar qualidade de geração. Sem REPL, não dá para iterar. É o menor delta com maior retorno.

1. **Sampling (temperature/top-k/top-p):**
   - Implementar `ggml_sampler` do próprio ggml (o ggml já tem `ggml_sampler_init_softmax`, etc.)
   - Ou implementar manual: aplicar temperature aos logits, fazer top-k/top-p filter, samplear com distribuição categórica
   - CLI flags: `--temperature`, `--top-k`, `--top-p`, `--repeat-penalty`
   - ~50 linhas no main.cpp

2. **REPL interativo:**
   - Após prefill, ler tokens do stdin
   - A cada token gerado, adicionar ao contexto e continuar
   - Ctrl+C para parar
   - ~30 linhas no main.cpp

**Gate:** gerar 100 tokens com temperature=0.7 e obter texto coerente (não repetitivo).

### Banda 2 -- Servidor HTTP

**Por que segundo:** O servidor é a interface que permite plugar o Lamio em qualquer ferramenta (Hermes, frontend, tests). É o ponto de interoperabilidade.

1. **Servidor HTTP minimalista:**
   - Usar `httplib` (header-only, já no ggml? ou inline) ou raw socket
   - Endpoint `POST /v1/completions` com JSON: prompt, max_tokens, temperature
   - Endpoint `GET /v1/models` (lista modelo carregado)
   - Resposta JSON compatível com OpenAI: `choices[].text`, `usage.completion_tokens`
   - CLI flag: `--serve --port 8080`

2. **Sampling no servidor:**
   - Mesmas flags de sampling mas via JSON no request body

**Gate:** `curl -s http://localhost:8080/v1/completions -d '{"prompt":"Hello","max_tokens":20}'` retorna texto correto.

### Banda 3 -- Modelos maiores (Ornith-9B denso)

**Por que terceiro:** Ornith-9B é denso (não-MoE), mesma arquitetura do Qwen3.5 mas maior. Valida que o forward pass escala para modelos maiores sem mudanças arquiteturais.

1. **Generalizar hparams:**
   - O v2 já lê hparams do GGUF. Ornith-9B pode ter n_head, head_dim, n_layers diferentes
   - Verificar se o `parse_qwen35_hparams` lida com todos os tamanhos
   - Possivelmente ajustar max_ctx (4096 hardcoded)

2. **Testar com Ornith-9B-Q4_0.gguf:**
   - Verificar se carrega, gera, e produz tokens coerentes
   - Medir t/s em CPU (sem GPU)

**Gate:** Ornith-9B gera 20 tokens coerentes, sem crash.

### Banda 4 -- MoE (Ornith-35B)

**Por que por último:** MoE é o objetivo final do Lamio (expert tiering). Mas depende de tudo anterior funcionar. O Ornith-35B tem 113 experts em MXFP4. O `ggml_mul_mat_id` já existe no ggml. Preciso implementar o roteamento de experts no forward pass.

1. **Implementar `mul_mat_id` (expert routing):**
   - O peso `wq` vira um tensor de experts `[n_experts, n_embd, n_out]`
   - O gate (router) produz `expert_ids` E `expert_weights`
   - `ggml_mul_mat_id` faz a multiplicação seletiva

2. **MoE layer no forward:**
   - Detectar se a layer é MoE (pelo tensor de gate)
   - Roteamento: router_logits -> top-k experts -> mul_mat_id
   - Combinar outputs dos experts ponderados

3. **Expert tiering (opcional, se CPU+mmap):**
   - Flag `--lamio-tier-budget` (MiB) e `--lamio-expert-k`
   - Carregar só K experts na RAM, resto via mmap
   - Endpoint `/lamio/tier-stats`

**Gate:** Ornith-35B gera 20 tokens coerentes. Se tiering implementado: funciona com --lamio-tier-budget 2048 em máquina com <16GB RAM.

## RECORDED TRADE-OFFS

- **httplib vs raw socket:** httplib é mais fácil mas adiciona dependência. Preferir raw socket inline (~200 linhas) para manter zero-dependência.
- **ggml_sampler vs manual:** ggml já tem sampler. Usar o que existe no ggml em vez de reimplementar.
- **MoE tiering:** só necessário para CPU+mmap. Se rodando em GPU ou RAM suficiente, tier é desnecessário. Implementar como flag opcional.

## ACCEPTANCE CRITERIA (AC-001 a AC-006 do Conscio)

1. AC-001: Forward pass completa sem erros para Qwen3.5-0.8B E Ornith-9B
2. AC-002: Geração com temperature=0.7 produz texto coerente (não repetitivo)
3. AC-003: Inputs inválidos (prompt vazio, max_tokens=0) retornam erro gracefully
4. AC-004: Servidor HTTP responde `/v1/completions` com formato OpenAI
5. AC-005: Teste de regressão: Hello -> ,\n\nI have a question ainda passa
6. AC-006: Changelog atualizado a cada banda

## OPEN QUESTIONS (RESPONDIDAS)

1. **Ornith-35B usa `mul_mat_id` nativo?** SIM. `ggml_mul_mat_id` existe em `ggml.h:1445`. O ggml já suporta o op.
2. **httplib disponível?** SIM. `vendor/cpp-httplib/` já existe no repo Lamio. Posso reusar.
3. **Ornith-9B usa GDN+attn híbrido?** A confirmar quando o modelo GGUF estiver disponível. O Qwen3.5 usa híbrido (4n+3 = attention, resto GDN).
4. **ggml_sampler?** NÃO existe no ggml standalone. É parte do llama.cpp. Preciso implementar sampling manual (~50 linhas: temperature scale, top-k filter, top-p filter, distribuição categórica).
