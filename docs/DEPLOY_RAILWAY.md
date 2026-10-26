# Deploy no Railway — passo a passo

## 0. Pré-requisitos

1. Repositório do MQ-MAS no GitHub.
2. Conta no [Railway](https://railway.com).
3. Pelo menos uma chave de LLM gratuita:
   * Groq: <https://console.groq.com/keys>
   * Gemini: <https://aistudio.google.com/apikey>
   * OpenRouter: <https://openrouter.ai/keys>
4. **Somente para live:** uma API key da corretora com permissão **apenas de Spot Trading**. Deixe **desativados** saques, margem e futuros. Se possível, restrinja a chave por IP (veja o passo 6).

## 1. Criar o serviço

1. Railway → **New Project** → **Deploy from GitHub repo** → selecione `mq-mas`.
2. O Railway detecta o `railway.toml` e faz o build com o `Dockerfile`.

## 2. Região (crítico para a Binance)

A Binance bloqueia IPs dos EUA (HTTP 451). Na aba **Settings → Deploy → Region**, escolha uma região **fora dos EUA**, como `europe-west4` (Amsterdã) ou `asia-southeast1` (Singapura).

Depois do deploy, confirme no shell do serviço (`railway ssh` ou a aba *Shell*):

```bash
python -c "import urllib.request;print(urllib.request.urlopen('https://api.binance.com/api/v3/ping').status)"
```

A resposta precisa ser `200`. Se vier 451 ou 403, troque de região ou use `EXCHANGE_ID=bybit`. Verifique também os termos da corretora para a sua jurisdição.

## 3. Volume persistente para o SQLite

Sem volume, o banco (memória de trades, circuit breaker) **some a cada deploy**.

1. No serviço: **+ New → Volume**, com mount path **`/data`**.
2. O `Dockerfile` já define `DB_PATH=/data/mqmas.db`.

## 4. Variáveis de ambiente

Em **Variables → Raw Editor**, cole e ajuste (a referência completa está em `.env.example`):

```env
TRADING_MODE=paper
EXCHANGE_ID=binance
SYMBOL=BTC/USDT
INITIAL_CAPITAL_USDT=50
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
ADMIN_TOKEN=<string-aleatória-longa>
LOG_LEVEL=INFO
```

Não defina `PORT`: o Railway injeta essa variável sozinho.

## 5. Domínio público

Em **Settings → Networking → Generate Domain**, gere o domínio e abra `https://<seu-app>.up.railway.app`. O escritório pixel art conecta sozinho em `wss://…/ws`. O healthcheck usa `GET /health`.

## 6. Operação em paper (recomendado: semanas)

* Acompanhe pelos logs JSON do Railway e por `GET /api/state`, `/api/trades` e `/api/debates`.
* Os trades em paper alimentam a evidência estatística (n, p̂) usada pelas travas.

## 7. Migrar para live (só se os números em paper justificarem)

1. **Static outbound IP** (plano Pro): em *Settings → Networking*, ative o IP de saída estático e cadastre esse IP na whitelist da API key da corretora.
2. Adicione estas variáveis:
   ```env
   TRADING_MODE=live
   LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISKS
   EXCHANGE_API_KEY=...
   EXCHANGE_API_SECRET=...
   ```
3. Mantenha na conta **apenas o capital do mandato**. O sistema já limita o tamanho das ordens a `min(saldo livre, banca do mandato)`.
4. A Binance cobra taxa em BTC na compra, a menos que você pague em BNB. O broker desconta a taxa da quantidade vendável automaticamente.

## 8. Kill switch de emergência

```bash
curl -X POST "https://<seu-app>.up.railway.app/api/kill-switch?engage=true" \
     -H "X-Admin-Token: $ADMIN_TOKEN"
```

Esse comando bloqueia novas entradas e zera qualquer posição aberta a mercado no próximo ciclo (≤ 2s). Para liberar, use `engage=false`.

## 9. Checklist pós-deploy

- [ ] `/health` retorna `feed_stale: false`
- [ ] O escritório mostra telemetria atualizando e o pill **PAPER**
- [ ] O volume `/data` está montado (o banco persiste após um redeploy)
- [ ] O status dos provedores LLM aparece em `/health` → `llm`
- [ ] Há exatamente 1 réplica
