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

### 3.1 Falta uma coisa: as chaves de IA

Sem chave de IA, os agentes se abstêm em todo debate (o log mostra `no LLM API key configured`) e o sistema não opera, nem em simulação. Basta **uma** chave para funcionar; as outras duas são reservas para quando a primeira atingir o limite grátis ou cair.

| Ordem | Provedor | Site para criar a chave | Variável no Railway | Formato da chave | Modelo (já configurado) |
|---|---|---|---|---|---|
| 1ª (principal) | **Groq** | <https://console.groq.com/keys> | `GROQ_API_KEY` | começa com `gsk_` | `openai/gpt-oss-120b` |
| 2ª (reserva) | **Google Gemini** | <https://aistudio.google.com/apikey> | `GEMINI_API_KEY` | começa com `AIza` | `gemini-flash-latest` |
| 3ª (reserva) | **OpenRouter** | <https://openrouter.ai/settings/keys> | `OPENROUTER_API_KEY` | começa com `sk-or-` | `openai/gpt-oss-120b:free` |

Nenhum dos três pede cartão de crédito para o plano gratuito.

**Groq (obrigatório)**
1. Acesse <https://console.groq.com> e entre com Google ou e-mail.
2. Menu **API Keys** → **Create API Key** → nome `mq-mas` → **Submit**.
3. Copie a chave (`gsk_...`). Ela aparece **uma única vez**.

**Google Gemini (recomendado)**
1. Acesse <https://aistudio.google.com/apikey> com sua conta Google e aceite os termos.
2. **Create API key** → escolha ou crie um projeto (ex.: `mq-mas`) → copie a chave (`AIza...`).
3. Não ative faturamento (*billing*): sem ele, a chave fica no plano gratuito e nunca gera cobrança.
4. Em **Rate limits** (no próprio AI Studio) você vê quais modelos Flash estão gratuitos e os limites do seu projeto.

**OpenRouter (opcional)**
1. Acesse <https://openrouter.ai> → **Sign in**.
2. Vá em <https://openrouter.ai/settings/keys> → **Create Key** → nome `mq-mas` → copie (`sk-or-...`).
3. Modelos `:free` têm limite de ~50 chamadas/dia. Depositar US$ 10 uma única vez sobe para ~1.000/dia (opcional; não é preciso para começar).

**Colar no Railway** (para cada chave)
1. <https://railway.com/dashboard> → projeto `virtual_officer` → serviço `mq-mas` → aba **Variables**.
2. **New Variable** → nome (ex.: `GROQ_API_KEY`) → valor (a chave) → **Add**.
3. Clique em **Deploy** na faixa roxa que aparece no topo. O redeploy leva ~2 minutos.
4. Confira em **Deployments → View logs**: devem aparecer mensagens dos agentes debatendo, sem `no LLM API key configured`.

> Nunca cole chaves em chats, e-mails ou no GitHub. O lugar delas é só a aba **Variables** do Railway.

> Os provedores trocam modelos com frequência (o Groq desativou o `llama-3.3-70b-versatile` em 16/08/2026). Se um modelo sair do ar, basta alterar `GROQ_MODEL`, `GEMINI_MODEL` ou `OPENROUTER_MODEL` no Railway; enquanto isso, o roteador usa o próximo provedor da fila.

### 3.2 A Liga de Setups: como o robô escolhe estratégias

O robô não fica preso a uma estratégia só. Ele mantém uma **biblioteca de 9 setups** e os põe para competir num torneio com **2 anos de histórico** do BTC/USDT:

| Família | Setups |
|---|---|
| Reversão | Reversão Z-Score · RSI-2 em tendência · Reentrada Bollinger · Capitulação com pavio · Divergência de RSI |
| Tendência | Pullback na EMA20 |
| Rompimento | Rompimento Donchian · Squeeze de Bollinger |
| Fluxo | Recuperação da VWAP |

**Como o torneio funciona** (roda sozinho a cada 24h, sem gastar IA, em menos de 1 segundo de CPU):

1. Cada setup é testado com as **mesmas regras da operação real**: entrada na abertura do candle seguinte, alvo de 0,6–0,8%, stop de 0,3–0,6%, saída por tempo após 8h, taxas de 0,1% em cada ponta e derrapagem. Se o stop e o alvo caírem no mesmo candle, conta o stop.
2. **Walk-forward:** o histórico é cortado em 5 partes. O melhor alvo e stop de cada setup é escolhido nas partes anteriores e o setup é avaliado na parte seguinte, **que ele nunca viu**. Só esse resultado "fora da amostra" decide a aprovação.
3. **Aprovação** exige tudo isto: pelo menos 20 trades fora da amostra, lucro líquido médio > 0, profit factor ≥ 1,15, taxa de acerto ≥ 75% (o seu mandato) e resultado positivo em pelo menos metade das janelas de teste.
4. Só setups **aprovados** convocam o debate. Quando dois ou mais disparam juntos, isso é apresentado aos agentes como **confluência**.
5. No debate, o Chief Quant avalia se o regime atual (tendência ou lateral) combina com o setup. O Econometrista confere as estatísticas fora da amostra do setup somadas às operações em papel do mesmo setup. O CRO e a Execução mantêm as travas de sempre.

O painel **Liga de Setups** mostra o ranking ao vivo. Setups reprovados aparecem com o motivo (passe o mouse). **Se nenhum setup for aprovado, o robô não opera.** Isso não é defeito: significa que nenhuma estratégia provou vantagem depois das taxas. Você pode afrouxar a meta de acerto (`MIN_WIN_RATE`, padrão 0,75), mas isso aumenta o risco. Decida olhando o relatório.

### 3.3 Carteira multi-moeda (o setup vencedor da pesquisa)

A pesquisa (`/api/research`) testou 544 configurações em 40 moedas com walk-forward. O único setup sólido foi a **confluência diária de rompimento**, que agora roda em simulação na **Carteira multi-moeda**:

| Camada | Regra |
|---|---|
| Universo | Todo mês: as 15 moedas USDT de maior volume em dólar (30 dias), com oscilação diária média de 2,5–15% |
| Obrigatório | BTC acima da média de 100 dias (dia anterior) · moeda acima da EMA200 · EMA50 acima da EMA200 |
| Gatilho | Fechamento diário acima da máxima dos 20 dias anteriores |
| Extras (2 de 3) | Força relativa entre os 40% melhores · volume acima de 1,3× a média · RSI(14) entre 50 e 70 |
| Saídas | Stop = entrada − 2,5×ATR · alvo = entrada + 4,5×ATR · saída por tempo em 20 dias |
| Risco | 0,6% da banca por operação (`BOOK_RISK_PER_TRADE`) · máx. 3 posições · kill switch, circuit breaker e perda diária compartilhados |

Resultado fora da amostra (jul/2023–out/2026): 102 trades, 56% de acerto, profit factor 2,29, drawdown máximo de 2,9%, ~2,5 trades por mês.

Como funciona ao vivo: a cada 30 segundos o robô vigia stops, alvos e saídas por tempo das posições abertas. A cada fechamento de candle do tempo gráfico do setup (00:00 UTC no diário), ele reavalia as 40 moedas. Cada sinal passa pelo pre-mortem da CRO (IA), que pode vetar a entrada. Sem IA disponível, não entra (fail-closed).

> Com 0,6% de risco e stops de ~14%, a maioria das posições fica abaixo do mínimo de US$ 5 da Binance. A simulação registra a operação e avisa quando isso acontece. No modo real seria preciso uma banca de ~US$ 120 ou um risco de ~1,5% por operação. A carteira ainda **não opera em modo LIVE**.

Para adicionar novos setups aprovados pela pesquisa, coloque as chaves de configuração separadas por `;` em `BOOK_STRATEGIES`.

### 3.4 Por quanto tempo ficar em paper?

**Recomendação: no mínimo 4 a 8 semanas.** Motivos:

- A trava `sample_size` exige pelo menos **20 operações** de evidência estatística. Os trades em paper contam como evidência.
- A trava `win_rate` exige **taxa de acerto ≥ 75%**. Se o setup não atingir isso, o sistema simplesmente não opera, e isso é a proteção funcionando.

### 3.5 Critérios para passar para dinheiro real

Só avance quando **todos** forem verdadeiros:

- [ ] Pelo menos 1 setup **aprovado** na Liga (`/api/league`)
- [ ] Pelo menos 20 trades fechados em paper (`/api/trades`)
- [ ] Taxa de acerto real ≥ 75% nesses trades
- [ ] Resultado acumulado positivo (`/api/state` → `equity` > 50)
- [ ] Nenhum erro recorrente nos logs do Railway por 2 semanas
- [ ] Você entendeu a seção 6 (riscos)

---

## 4. Fase 2 — Preparar a Binance

### 4.1 Conta e depósito

1. Crie a conta em <https://www.binance.com/pt-BR/register> (ou pelo app oficial **Binance**, na loja do seu celular) e conclua a **verificação de identidade (KYC)**.
2. **Ative a autenticação em dois fatores (2FA)** com um app autenticador (Google Authenticator ou similar).
3. Deposite reais via **PIX** (Carteira → Depositar → BRL).
4. Converta para **USDT** (Comprar/Converter, ou o par `USDT/BRL`). O sistema opera **BTC/USDT** e precisa de saldo em **USDT** na **carteira Spot**.
5. Valor sugerido: o equivalente a **US$ 50** (≈ R$ 300), que é a banca configurada (`INITIAL_CAPITAL_USDT=50`).

> O sistema nunca usa mais que a banca configurada, mesmo que a conta tenha mais saldo. Ainda assim, o ideal é deixar na conta **só** o valor destinado ao robô.

> **Não ative "pagar taxas com BNB"**, a menos que ajuste `FEE_RATE=0.00075`. Isso mantém a contabilidade de taxas do sistema exata.

### 4.2 Criar a chave de API

Na Binance (<https://www.binance.com/pt-BR/my/settings/api-management>): **Perfil → Gerenciamento de API → Criar API → "Gerado pelo sistema"**. Dê um nome (ex.: `mq-mas`).

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
| `/api/book` | Carteira multi-moeda: setups ativos, universo, últimos sinais, posições |
| `/api/research` | Relatório completo da pesquisa multi-moeda (walk-forward, estresse, Monte Carlo) |
| `/api/league` | Ranking completo da Liga de Setups (estatísticas fora da amostra de cada setup) |

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
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modelo da IA principal |
| `GEMINI_MODEL` | `gemini-flash-latest` | Modelo da 1ª reserva |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b:free` | Modelo da 2ª reserva |
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

## Execução real, alertas e backups

### Duas chaves para operar dinheiro real na carteira multi-moeda
Precisam estar **as duas** ligadas: `TRADING_MODE=live` (com `LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISKS` e as chaves da Binance) **e** `BOOK_LIVE_ENABLED=true`. Sem a segunda, o modo real registra os sinais e não envia ordens.

O que a execução real garante (`mqmas/execution/spot.py`):
- **Stop dentro da Binance** (STOP_LOSS_LIMIT) logo após a compra. Se o servidor cair, a posição continua protegida. Se a Binance recusar o stop, chega um alerta e o stop por software continua ativo.
- **ID próprio em cada ordem** (`clientOrderId`), gravado no banco antes do envio. Se houver queda ou timeout, a ordem é consultada pelo ID e nunca reenviada.
- **Reconciliação ao ligar:** entradas pendentes são conferidas na Binance, e stops que sumiram são recolocados.
- **Filtros do par:** lote, tick e mínimo de US$5. Ordens abaixo do mínimo são recusadas antes do envio, com alerta. Com US$50 a 0,6% de risco quase toda ordem fica abaixo do mínimo; veja a decisão A/B.
- **Relógio sincronizado** com a Binance (`adjustForTimeDifference`) e limitador de requisições ligado.

### Alertas no celular (Telegram)
1. No Telegram, fale com **@BotFather** → `/newbot` → copie o token.
2. Mande qualquer mensagem para o seu bot e abra `https://api.telegram.org/bot<TOKEN>/getUpdates` para ver o `chat.id`.
3. No Railway → Variables: `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`.

Você recebe: compras, vendas, circuit breaker, kill switch, stop ausente, ordem recusada, falha de reconciliação e o **backup diário do banco** (às 03:00 UTC, mantendo 7 cópias também no volume).

### Aviso se o robô parar (heartbeat)
Crie um check grátis em **healthchecks.io** com período de 10 min e coloque a URL de ping em `HEARTBEAT_URL`. Enquanto o robô estiver saudável, ele envia o ping a cada 5 min. Se os pings pararem, o healthchecks avisa você por e-mail ou Telegram.

### Revisão de relatórios pelas IAs
`POST /api/review` (cabeçalho `x-admin-token`) com `trades`, `win_rate`, `payoff`, `net_return_pct`, `max_consecutive_losses`, `max_drawdown_pct` e `patterns`. O Python calcula os fatos e Gauss, Ada e Vera debatem, com o histórico medido dos padrões. O debate aparece no escritório.
