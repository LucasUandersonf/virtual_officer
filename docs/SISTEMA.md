# Como o sistema funciona: gestores, comitê, execução e placar

Todos os números abaixo vêm de **backtests com dados reais da Binance**, com taxas (0,10%) e derrapagem (0,05%) por lado. Nenhum vem de operação com dinheiro real: o sistema ainda não operou ao vivo.

Viés comum a todos os resultados: as moedas testadas são as mais negociadas **hoje**, e as que morreram desde 2019 ficaram de fora. Isso infla qualquer estratégia que só compra, inclusive as referências (segurar BTC).

## 1. A visão geral

```
              Binance (candles diários de 40 moedas, a cada execução horária)
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
 GESTOR DE TENDÊNCIA (70% da banca)              CAÇADOR DE ONDAS (30% da banca)
 9 canais Donchian, top 20 moedas,               toda segunda-feira: as 5 moedas que mais
 tamanho por volatilidade; rebalanceia           subiram em 3 semanas, peso igual; fica em
 todo dia; sai quando a tendência quebra         USDT se o BTC estiver abaixo da média de 100 dias
          └─────────────────────────┬─────────────────────────┘
                                    ▼  cada COMPRA proposta
                         COMITÊ DE INVESTIMENTO
   técnico → fundamentalista → sentimento → risco (Vera, IA, lê os outros pareceres)
   cada um: confirmar · reduzir (metade) · vetar · abster
   só os especialistas ATIVOS mudam a ordem; os demais ficam "em sombra"
                                    ▼
                EXECUÇÃO (Binance spot, sem alavancagem)
   ordem limitada · stop de desastre na corretora (6 ATR, só sobe)
   trava de queda por gestor · kill switch · alertas no Telegram
                                    ▼
                                 PLACAR
   o que cada especialista disse × o que a moeda fez 7 e 30 dias depois
   → quem ajuda é ativado; quem atrapalha continua em sombra
```

**Regra de ouro:** só os gestores **propõem**, porque são as únicas partes com evidência forte fora da amostra. Os especialistas podem confirmar, reduzir ou vetar, mas nunca criar uma compra nem aumentar uma posição.

## 2. Resultados dos gestores

O caçador e a carteira 70/30 foram simulados como uma conta real de US$ 120 (ordem mínima de US$ 5, banda de 25%, stop na corretora). A linha do gestor de tendência vem da simulação por pesos.

| 2020-01 → 2026-10 | CAGR | Queda máx. | Sharpe | Desde jul/2023 |
|---|---|---|---|---|
| Gestor de tendência (top 20, vol. 60%) | +39% | 35% | 1,27 | +21%/ano, queda 33% |
| Caçador de ondas | +125% | 67% | 1,52 | +72%/ano, queda 67% |
| **Carteira 70% tendência + 30% ondas** | **+62%** | **38%** | **1,45** | **+36%/ano, queda 35%** |
| Segurar BTC | +42% | 77% | 0,89 | +34%/ano, queda 53% |

Ano a ano da carteira 70/30: 2020 +109% · 2021 +399% · **2022 −20%** · 2023 +87% · 2024 +50% · **2025 −11%** · 2026 (até out.) +32%.

Testes de robustez do caçador de ondas (`mqmas/research/waves.py`):

| Teste | CAGR | Queda máx. | Sharpe |
|---|---|---|---|
| Original | +135% | 67% | 1,57 |
| Custos dobrados | +126% | 68% | 1,51 |
| Sem as 3 moedas que mais contribuíram (FET, HBAR, ADA) | +92% | 67% | 1,30 |
| Sem as 5 que mais contribuíram | +73% | 70% | 1,16 |

**As ondas grandes, em números:**
- Depois de um dia de **+10%** (1.899 casos), a moeda rendeu em média **+4,4% acima do normal** nos 20 dias seguintes (t = 7,4).
- Mas a **mediana foi −2,4%** e só 47% dos casos subiram. Na maioria das vezes a moeda devolve parte da alta; o lucro vem de poucas ondas gigantes.
- Depois de um dia de **+30%**, não houve vantagem estatística.

Por isso o caçador não compra "o dia de +30%". Ele segura as moedas com a **alta mais consistente em 3 semanas** e troca toda semana. Ele estava posicionado em 30% dos dias de +20% do universo; o gestor de tendência (top 20), em 59%.

## 3. Catálogo de análises técnicas e o resultado medido de cada uma

### Validadas: usadas pelo técnico do comitê para confirmar

| Análise | Onde | Resultado fora da amostra |
|---|---|---|
| Rompimento clássico diário (máxima de 20 dias + volume 1,5×) | estudo clássico, zero parâmetros | 119 operações, acerto 46%, PF 2,11 (2,02 com custo dobrado), 75% das janelas de 6 meses positivas |
| Rompimento 1d (arena) | `research/arena.py` | n=274, acerto 42%, PF 1,62 |
| Rompimento + cruzamento de médias 1d | arena | n=73, acerto 48%, PF 2,12 |
| Rompimento 2d | arena | n=151, acerto 43%, PF 2,36 |
| Rompimento 3d | arena | n=112, acerto 50%, PF 2,23 |
| Triângulo descendente rompido para cima, 3d (Stanzione) | arena | n=65, acerto 49%, PF 3,24 |
| "Mean Reversion Ignition" 1d (worklist do time, nº 85) | estudo de padrões v5 | o **único de 350 sinais** que funcionou: 131 operações, acerto 57%, PF 2,28 |

Ressalva: na escolha "cega" (o sistema escolhendo sozinho, a cada janela, qual setup usar), a arena ficou com PF de 1,4 a 1,7 e um Sharpe deflacionado quase zero. A evidência individual é boa, mas não dá para descartar sorte.

### Funcionam como camada de contexto (informativo, sem poder de veto)

| Análise | Resultado |
|---|---|
| Score de confluência 0–100 (localização, padrões, tendência, momento, volume) | **Não validado** no 4h nem no diário. Aparece no parecer do técnico só como informação |
| Mapa de suportes e resistências | Regra "resistência forte a menos de 1 ATR → reduzir" em sombra; ainda sem validação |
| Fundo arredondado 1d (livro) | n=158, PF 1,72, estável nos dois períodos, mas sem significância na escolha cega |
| ≥2 estratégias do livro ao mesmo tempo (1d) | n=738, PF 1,53 no 2º período. Conferido com um motor independente (backtesting.py): PF 1,24 vs 1,24 |

### Reprovadas: não usadas

| Análise | Resultado |
|---|---|
| Mesa original de BTC em 15 min (9 setups, alvo de 0,6–0,8%) | Todos os 9 falharam depois das taxas |
| 61 padrões de candle da TA-Lib | Escolha cega: PF 0,69 (1d) e 1,11 (2d). Sozinhos, nenhum sobrevive |
| 340 de 350 sinais do estudo de padrões | Sem efeito |
| Pullback na EMA20 (clássico) | PF 0,95 |
| Bollinger (fecha fora, volta para dentro) | PF 0,19 |
| Fibonacci 50–61,8% | PF 1,21 em só 40 operações |
| Grid bot | PF 0,91; perde tudo numa queda |
| Divergência de RSI + engolfo | PF 0,63–0,97 |
| Reteste de resistência rompida | PF 1,09; com custo dobrado, 0,99 |
| Squeeze de Bollinger 1h/4h | PF 0,84–0,95 |
| Confluência 4h e ensemble de confluências | PF 1,08 e 1,06 |
| Cunha descendente 1d / triângulo ascendente 1d / fundo duplo 1d | PF 0,79 / 0,73 / 0,82–0,98 no 2º período |
| Modelos do TCC de Prates (2021) no BTC | Cruzamento de médias +45%/ano, mas queda de 61%; perde para segurar BTC. Estocástico +4%/ano, FFFD 0% |
| Funding negativo, clímax de venda, pullback na VWAP, filtro de IA (swing v6) | Reprovados (PF 0,50–0,92) |

### Funcionam ao contrário: por isso o técnico nunca veta com padrão de baixa

8 sinais "de baixa" foram seguidos de **alta**, de forma estável nos dois períodos. Exemplos: RSI saindo da sobrecompra (4h), CCI saindo de cima (4h), "BB Squeeze Fakeout Down" (4h, PF 0,72 quando usado como compra), "Fibo 78,6% Deep Liquidation" (4h, PF 0,77). Vetar compras por causa deles atrapalharia.

## 4. O que acontece a cada execução (cron, de hora em hora)

1. **Stops.** Em paper, confere as mínimas da última hora contra o stop de cada posição. Ao vivo, consulta as ordens de stop na Binance e recoloca as que sumiram.
2. **Travas.** O kill switch vende tudo. Cada gestor tem a própria trava de queda: tendência 45%, ondas 70%. Disparou, vende tudo e para até um reset manual.
3. **Uma vez por dia** (primeira execução depois das 00:00 UTC):
   - cada gestor calcula os pesos com **a mesma função do backtest**;
   - cada compra passa pelo comitê;
   - as ordens respeitam o mínimo de US$ 5 e a banda de 25%;
   - os stops sobem;
   - o placar recebe o resultado dos pareceres antigos.
4. **Telegram:** compras, vendas, stops, parecer do comitê e paradas.

O caçador de ondas só troca de moedas na segunda-feira. Nos outros dias ele mantém as mesmas 5 moedas, com ajustes pela banda de 25% e pelos stops.

## 5. Os especialistas do comitê

| Especialista | Lê | Confirma quando | Reduz/veta quando |
|---|---|---|---|
| Técnico | Sinais validados (seção 3), mapa de preços, score | Há sinal validado ou nada contra | Resistência forte a < 1 ATR (regra não validada) |
| Fundamentalista | CoinGecko (valor de mercado, FDV, oferta), DefiLlama (TVL) | Sem alertas | FDV ≥ 2× o valor de mercado (diluição), TVL −30% em 30 dias |
| Sentimento | Medo e Ganância, funding dos perpétuos | Sem euforia | Ganância ≥ 80, funding > 0,05%/8h |
| Risco (Vera, IA) | A proposta + os pareceres anteriores | Pre-mortem sem risco alto | Voto contra ou risco alto sem mitigação |

Todos começam **em sombra**. `python -m mqmas.runner --placar` mostra, para cada especialista, o retorno de 30 dias das moedas que ele sinalizou contra as que ele liberou. Para ativar um especialista, inclua o nome dele em `TREND_ACTIVE_SPECIALISTS` (ex.: `fundamentalista,risco`) **depois** que o placar mostrar "ajuda".

## 6. Configuração recomendada (conta de US$ 120)

```
TREND_CAPITAL_USDT=84        # 70%
TREND_TOP_N=20
TREND_VOL_TARGET=0.60
TREND_MAX_DRAWDOWN=0.45
ONDAS_CAPITAL_USDT=36        # 30%
ONDAS_MAX_DRAWDOWN=0.70
TREND_ACTIVE_SPECIALISTS=    # todos em sombra no começo
```

## 7. O que esperar, com honestidade

- **Os ganhos vêm concentrados.** A carteira teve 2021 com +399% e 2022 com −20%. Menos da metade dos meses é positiva.
- **A queda máxima de 38% vai acontecer de novo.** A do caçador de ondas sozinho foi de 67%.
- **Desde jul/2023** a carteira rendeu +36% ao ano contra +34% de segurar BTC, com queda máxima de 35% contra 53%.
- **Antes de dinheiro real:** 8 semanas em paper, conferindo se o resultado bate com o backtest. Ao vivo, abaixo de ~US$ 120, a ordem mínima de US$ 5 deixa boa parte do dinheiro parada.
