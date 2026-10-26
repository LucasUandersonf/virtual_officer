# Guia operacional do MQ-MAS

Este guia explica, em português e sem jargão desnecessário, o que é o sistema, quanto custa mantê-lo, e o passo a passo para sair do modo de simulação (paper) e operar com dinheiro real na Binance.

> **Leia antes de tudo.** O sistema foi desenhado para *preservar capital*: só opera spot (sem alavancagem, sem risco de liquidação) e limita a perda de cada operação a 0,6% da banca. Isso **não** garante lucro. Ele pode perder dinheiro, e com as regras atuais vai passar a maior parte do tempo **sem operar**. Esse comportamento é intencional.

---

## 1. O que foi construído

| Peça | O que faz | Onde está |
|---|---|---|
| **Pipeline quantitativo** | Calcula a cada minuto o Z-Score, a VWAP, o Volume Profile (POC), o desequilíbrio do livro de ofertas, a projeção de preço com intervalo de 95%, o ATR e o tamanho da posição (Kelly) | `mqmas/quant/` |
| **4 agentes de IA** | Ada (analista, *Tree of Thoughts*), Gauss (estatística), Vera (risco, *pré-mortem*) e Hex (execução). Debatem cada oportunidade; só opera com **unanimidade** | `mqmas/agents/` |
| **Travas rígidas** | 20 regras em código que nenhuma IA consegue contornar (ex.: risco máx. 0,6%, circuit breaker após 2 stops) | `mqmas/risk/constraints.py` |
| **Roteador de IAs** | Usa Groq; se cair, Gemini; se cair, OpenRouter. Se todos falharem, **não opera** | `mqmas/llm/router.py` |
| **Execução** | Modo `paper` (simulado com o livro real da Binance) ou `live` (ordens reais spot, com stop na própria corretora) | `mqmas/execution/` |
| **Memória** | Banco SQLite com todos os trades, debates e eventos de risco | `mqmas/persistence/` |
| **Escritório virtual** | A tela onde você assiste aos agentes debatendo ao vivo | `mqmas/web/static/` |

**Endereços:**

- App no ar: <https://mq-mas-production.up.railway.app>
- Código: <https://github.com/LucasUandersonf/virtual_officer>
- Hospedagem: projeto `virtual_officer` no Railway, serviço `mq-mas`, região Amsterdã

---

## 2. Quanto custa por mês

### 2.1 Railway (hospedagem)

Preços oficiais do Railway (out/2026): RAM **US$ 10/GB/mês**, CPU **US$ 20/vCPU/mês**, volume **US$ 0,15/GB/mês**, tráfego de saída **US$ 0,05/GB**. Os builds são gratuitos.

Consumo **medido** do serviço `mq-mas` nas primeiras horas no ar:

| Recurso | Consumo medido | Custo/mês estimado |
|---|---|---|
| Memória | ~0,36–0,46 GB contínuos | ~US$ 3,60–4,60 |
| CPU | ~0,003–0,01 vCPU em média | ~US$ 0,10–0,20 |
| Volume `/data` | ~0,08 GB | ~US$ 0,01 |
| Tráfego de saída | < 1 GB | < US$ 0,05 |
| **Uso total** | | **~US$ 4–5** |

O **plano Hobby custa US$ 5/mês e inclui US$ 5 de uso**. Como o uso estimado fica dentro disso:

> **Custo esperado: ~US$ 5/mês (≈ R$ 28–30, conforme o câmbio).** Se o uso passar de US$ 5, você paga só a diferença (ex.: uso de US$ 6 → conta de US$ 6).

Dica: no Railway, em **Workspace → Usage**, defina um *usage limit* (ex.: US$ 10) para nunca ter surpresa.

### 2.2 Outros custos

| Item | Custo |
|---|---|
| IAs (Groq / Gemini / OpenRouter, planos gratuitos) | **US$ 0**. O sistema economiza chamadas de propósito |
| GitHub | **US$ 0** |
| Binance: taxa de negociação | 0,1% por lado. Numa operação típica de ~US$ 35, ida e volta ≈ **US$ 0,07** |
| Binance: depósito via PIX | Normalmente sem taxa; a conversão BRL→USDT tem *spread* |
| **IP fixo** (opcional, ver seção 4.3) | Exige o plano **Pro** do Railway: **US$ 20/mês** |

---

## 3. Fase 1 — Simulação (paper). É onde estamos agora

O sistema já está rodando em modo **paper**: usa o preço e o livro de ofertas **reais** da Binance, mas as ordens são simuladas. Nenhum centavo seu é usado.

### 3.1 Falta uma coisa: a chave de IA

Hoje os agentes se abstêm em todo debate porque não há chave de IA configurada (o log mostra `no LLM API key configured`). Para ligar:

1. Crie uma conta gratuita em <https://console.groq.com> e gere uma chave em **API Keys**.
2. No Railway: projeto `virtual_officer` → serviço `mq-mas` → aba **Variables** → **New Variable**:
   - Nome: `GROQ_API_KEY`
   - Valor: a chave que começa com `gsk_`
3. Salve. O Railway faz o redeploy sozinho.
4. (Opcional, recomendado) Adicione também uma chave reserva:
   - `GEMINI_API_KEY`, gratuita em <https://aistudio.google.com/apikey>
   - `OPENROUTER_API_KEY`, gratuita em <https://openrouter.ai/keys>

### 3.2 Por quanto tempo ficar em paper?

**Recomendação: no mínimo 4 a 8 semanas.** Motivos:

- A trava `sample_size` exige pelo menos **20 operações** de evidência estatística. Os trades em paper contam como evidência.
- A trava `win_rate` exige **taxa de acerto ≥ 75%**. Se o setup não atingir isso, o sistema simplesmente não opera, e isso é a proteção funcionando.

### 3.3 Critérios para passar para dinheiro real

Só avance quando **todos** forem verdadeiros:

- [ ] Pelo menos 20 trades fechados em paper (`/api/trades`)
- [ ] Taxa de acerto real ≥ 75% nesses trades
- [ ] Resultado acumulado positivo (`/api/state` → `equity` > 50)
- [ ] Nenhum erro recorrente nos logs do Railway por 2 semanas
- [ ] Você entendeu a seção 6 (riscos)

---

## 4. Fase 2 — Preparar a Binance

### 4.1 Conta e depósito

1. Crie a conta em <https://www.binance.com> e conclua a **verificação de identidade (KYC)**.
2. **Ative a autenticação em dois fatores (2FA)** com um app autenticador (Google Authenticator ou similar).
3. Deposite reais via **PIX** (Carteira → Depositar → BRL).
4. Converta para **USDT** (Comprar/Converter, ou o par `USDT/BRL`). O sistema opera **BTC/USDT** e precisa de saldo em **USDT** na **carteira Spot**.
5. Valor sugerido: o equivalente a **US$ 50** (≈ R$ 300), que é a banca configurada (`INITIAL_CAPITAL_USDT=50`).

> O sistema nunca usa mais que a banca configurada, mesmo que a conta tenha mais saldo. Ainda assim, o ideal é deixar na conta **só** o valor destinado ao robô.

> **Não ative "pagar taxas com BNB"**, a menos que ajuste `FEE_RATE=0.00075`. Isso mantém a contabilidade de taxas do sistema exata.

### 4.2 Criar a chave de API

Na Binance: **Perfil → Gerenciamento de API → Criar API → "Gerado pelo sistema"**. Dê um nome (ex.: `mq-mas`).

Permissões, **exatamente assim**:

| Permissão | Marcar? |
|---|---|
| Habilitar leitura (*Enable Reading*) | ✅ Sim |
| Habilitar negociação Spot e Margem (*Enable Spot & Margin Trading*) | ✅ Sim. O sistema só faz Spot, nunca margem |
| Habilitar saques (*Enable Withdrawals*) | ❌ **NUNCA** |
| Futuros, Margem, Transferência universal, etc. | ❌ Não |

Você recebe duas informações: a **API Key** e a **Secret Key**. A Secret aparece **uma única vez**; copie para um lugar seguro. Nunca envie essas chaves por chat, e-mail ou print.

### 4.3 Restrição de IP: a decisão importante

A Binance recomenda restringir a chave a IPs confiáveis. O Railway só oferece **IP de saída fixo no plano Pro (US$ 20/mês)**. Você tem duas opções:

| Opção | Custo extra | Segurança |
|---|---|---|
| **A. Railway Pro + IP fixo** (Settings → Networking → Static Outbound IPs) e cadastrar esse IP na chave da Binance | +US$ 15/mês (Pro US$ 20 em vez de Hobby US$ 5) | Melhor: a chave só funciona a partir do servidor |
| **B. Chave sem restrição de IP** | US$ 0 | Aceitável **somente** porque saques ficam desabilitados e a conta tem só ~US$ 50. Quem roubasse a chave poderia negociar, mas não sacar |

Com a opção B, a Binance mostra avisos e pode limitar ou expirar chaves sem restrição de IP. Leia o aviso exibido na tela de criação da chave. Para uma banca de US$ 50, os US$ 15/mês extras da opção A custam **30% da banca por mês**. Por isso a opção B, com saque desligado, costuma ser a escolha racional nesse tamanho.

---

## 5. Fase 3 — Ligar o modo real

No Railway → serviço `mq-mas` → **Variables**, adicione ou altere:

| Variável | Valor |
|---|---|
| `EXCHANGE_API_KEY` | sua API Key da Binance |
| `EXCHANGE_API_SECRET` | sua Secret Key da Binance |
| `TRADING_MODE` | `live` |
| `LIVE_TRADING_CONFIRM` | `I_UNDERSTAND_THE_RISKS` |

Salve. Depois do redeploy:

1. Abra o app: o selo no topo deve mudar de **PAPER** para **LIVE** (vermelho).
2. Confira nos logs do Railway (serviço → **Deployments → View logs**) a linha `starting MQ-MAS in live mode`.
3. Se algo falhar na validação (ex.: faltou a frase de confirmação), o app **se recusa a iniciar**. É proposital.

**Para voltar ao modo simulado** a qualquer momento: `TRADING_MODE=paper`.

---

## 6. Riscos: leia com atenção

- **Sem garantia de lucro.** As travas limitam a perda **por operação** e por dia; não impedem uma sequência de perdas pequenas.
- **As taxas comem a vantagem.** Com 0,1% por lado, uma perda no stop custa ~0,85% da posição e um ganho no alvo rende ~0,6%. Para empatar, o sistema precisa acertar **~59%** das vezes; por isso a exigência de 75%.
- **Perda máxima por operação ≈ US$ 0,30** (0,6% de US$ 50). Perda máxima por dia ≈ 1,5% da banca. Após 2 stops seguidos no dia, o sistema **se bloqueia por 24 h**.
- **Falha técnica.** Depois de cada compra, uma ordem *stop-limit* fica registrada **na própria Binance**, protegendo a posição mesmo se o servidor cair. O alvo de lucro (take profit) é gerido pelo servidor.
- **Entrada órfã.** Se o servidor cair exatamente entre enviar a compra e registrar o preenchimento, o sistema avisa no próximo início (evento `orphan_pending_entry` na ata). Nesse caso, confira a conta na Binance manualmente.
- **Impostos.** Ganhos com criptoativos podem ter tributação e obrigação de declaração no Brasil (inclusive os saldos em "Bens e Direitos" do IR). Converse com um contador.

---

## 7. Operação do dia a dia

### 7.1 Lendo a tela

| Área | O que mostra |
|---|---|
| **Topo** | Preço do BTC (variação 24 h), banca, PAPER/LIVE, CIRCUIT BREAKER (só aparece se bloqueado), conexão |
| **Escritório** | Os 4 agentes nas mesas; num debate, vão para a Sala de Debate |
| **Cartão do debate** | Quem aprovou, quem está analisando, cronômetro e resultado |
| **Telemetria** | Z-Score (vermelho ≤ −2 = sobrevenda), pressão do livro, VWAP, POC, projeção com IC 95% |
| **Plano da ordem** | Quantidade, preço, stop, alvo e risco máximo do último debate ou da posição aberta |
| **Ata do debate** | Tudo o que cada agente disse, com horário e voto |

### 7.2 Endereços úteis

| Endereço | Para quê |
|---|---|
| `/` | Escritório virtual |
| `/health` | Saúde do sistema (o Railway usa para reiniciar se cair) |
| `/api/state` | Banca, trades, bloqueios, posição aberta |
| `/api/trades` | Histórico de trades fechados |
| `/api/debates` | Histórico de debates |

### 7.3 Botão de emergência (kill switch)

Bloqueia novas entradas e **vende imediatamente** qualquer posição aberta.

- **Pela tela:** botão **Kill switch** → confirmar → informar o `ADMIN_TOKEN`.
- **O `ADMIN_TOKEN`** está no Railway → `mq-mas` → **Variables** → `ADMIN_TOKEN` (ícone de olho). Guarde-o num gerenciador de senhas.
- Para liberar de novo, clique no mesmo botão (pede o token outra vez).

### 7.4 Logs e problemas

Railway → `mq-mas` → **Deployments** → **View logs**.

| Mensagem no log | Significado | O que fazer |
|---|---|---|
| `no LLM API key configured` | Sem chave de IA; os agentes se abstêm | Seção 3.1 |
| `circuit breaker armed until …` | 2 stops seguidos; bloqueado 24 h | Nada; é a proteção agindo |
| `provider groq failed, cascading` | Groq instável; usando reserva | Nada, se houver chave reserva |
| `exchange stop rejected` | A Binance recusou o stop nativo; o stop passou a ser monitorado pelo servidor | Verificar a posição na Binance |
| `pending position … found at boot` | Entrada órfã após queda | Conferir a conta na Binance |

---

## 8. Configuração atual (Railway)

| Variável | Valor | Significado |
|---|---|---|
| `TRADING_MODE` | `paper` | Simulação |
| `INITIAL_CAPITAL_USDT` | `50` | Banca do mandato |
| `EXCHANGE_ID` / `SYMBOL` | `binance` / `BTC/USDT` | Corretora e par |
| `MAX_RISK_PER_TRADE_PCT` | `0.006` | Perda máx. por trade: 0,6% |
| `STOP_LOSS_MIN_PCT` / `MAX` | `0.003` / `0.006` | Stop entre 0,3% e 0,6% (via ATR) |
| `TAKE_PROFIT_MIN_PCT` / `MAX` | `0.006` / `0.008` | Alvo entre 0,6% e 0,8% |
| `MAX_SLIPPAGE_PCT` | `0.0005` | Derrapagem máx. 0,05% |
| `CONSECUTIVE_LOSS_LIMIT` / `LOCKOUT_HOURS` | `2` / `24` | Circuit breaker |
| `MAX_DAILY_LOSS_PCT` | `0.015` | Perda máx. diária 1,5% |
| `ZSCORE_ENTRY` / `ZSCORE_PRECHECK` | `-2.0` / `-1.5` | Gatilho de entrada / de convocar debate |
| `MIN_WIN_RATE` / `MIN_SAMPLE_SIZE` | `0.75` / `20` | Evidência mínima exigida |
| `KELLY_FRACTION` | `0.25` | ¼ de Kelly no dimensionamento |
| `DB_PATH` | `/data/mqmas.db` | Banco no volume persistente |
| `RAILWAY_RUN_UID` / `PORT` | `0` / `8000` | Permissão de gravar no volume / porta do app |

As travas de risco **só podem ser apertadas**. Valores mais frouxos que o mandato (ex.: risco > 0,6%) fazem o app **recusar a inicialização**.

---

## 9. Glossário rápido

- **Spot**: compra o ativo de fato; não há alavancagem nem liquidação.
- **Paper trading**: simulação com dados reais e dinheiro fictício.
- **Z-Score**: quantos desvios-padrão o preço está da média recente. −2 indica preço "esticado para baixo".
- **VWAP**: preço médio ponderado pelo volume do dia; funciona como "preço justo" intradiário.
- **POC (Point of Control)**: faixa de preço com mais volume negociado; costuma atuar como suporte ou resistência.
- **Stop loss / take profit**: preço de saída na perda / no ganho.
- **Slippage**: diferença entre o preço esperado e o executado.
- **Kelly**: fórmula que sugere quanto arriscar dada a vantagem estatística; usamos ¼ dela por segurança.
- **Circuit breaker**: bloqueio automático após perdas seguidas.
- **API Key / Secret**: "login e senha" que permitem ao robô operar sua conta.
