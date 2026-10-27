# Biblioteca de padrões, score de confluência e regra de execução

O que está no código e onde fica. Tudo roda em modo paper (simulado); o modo real continua bloqueado.

## 1. Banco de dados (SQLite, `mqmas/persistence/db.py`)

| Tabela | Uma linha por | Campos principais |
|---|---|---|
| `ta_patterns` | padrão (180+) | `id`, `categoria` (candle, estrutura/SMC, padrão gráfico, indicador, ondas de Elliott, worklist do time), `direcao` (alta/baixa/neutro), `familia`, `parametros` (JSON com as proporções de Fibonacci dos harmônicos, as regras de Elliott, a tolerância em ATR de topo/fundo duplo e OCO, a definição de FVG/OB) |
| `ta_pattern_stats` | padrão × timeframe | `veredito`, `ocorrencias`, `acerto` (descontada a tendência da moeda), `acerto_bruto` (o número que os sites mostram), `excesso_bps`, `pf_operacao`, `atualizado_em` |
| `ta_signals` | moeda × timeframe × candle analisado | `score`, `componentes`, `padroes` (com peso), `mapa` (suportes e resistências), `decisao`, `ordem` (entrada/stop/alvo/tamanho) — é o histórico de auditoria |

A tabela de estatísticas é atualizada sozinha a partir do estudo histórico (`ta_report.json`), no boot e depois de cada rodada de pesquisa.

APIs: `/api/patterns?tf=4h` (biblioteca), `/api/ta-signals` (histórico das varreduras), `/api/confluence/BTC?tf=4h` (score + ordem ao vivo), `/api/levels/BTC?tf=4h` (mapa de preços).

## 2. Score de confluência 0–100 (`mqmas/ta/confluence.py`)

| Componente | Peso | Como é medido |
|---|---|---|
| Localização | 25 | Quantos suportes ficam até 1 ATR abaixo do preço: fundo de 20 barras, EMA20/50/200, Bollinger inferior, VWAP, Kijun, topo da nuvem (3 ou mais = nota cheia) |
| Padrões | 35 | Soma dos padrões ativos nos últimos 3 candles, cada um com o peso que o **histórico medido** deu a ele. Padrão sem efeito vale 0; padrão de baixa comprovado **desconta** até −35 |
| Tendência | 20 | Fração dos 6 votos de alta: EMA50>EMA200, preço>EMA200, acima da nuvem, +DI>−DI, MACD>0, preço>SAR |
| Momento | 10 | Histograma do MACD subindo; RSI entre 50 e 70 ou virando para cima abaixo de 40 |
| Volume | 10 | OBV acima da média; volume acima de 1,2× a média de 20 |

Os pesos dos componentes foram fixados **antes** de qualquer teste e nunca foram otimizados nos dados.

Exemplo do pedido: o preço toca um order block, há um engolfo de alta e o RSI vem da sobrevenda.
- O order block entra como suporte (localização) e, se o reteste tiver histórico positivo, também como padrão.
- O engolfo só soma pontos se o estudo provar que ele funciona naquele timeframe; hoje ele vale zero.
- O RSI virando para cima abaixo de 40 soma no componente de momento.

## 3. Regra de execução

Igual no teste e ao vivo: o que opera é exatamente o que foi testado.

1. **Gatilho:** o score **cruza** o limiar validado de baixo para cima. É uma entrada por episódio; o robô não persegue preço.
2. **Entrada:** a mercado, na abertura do candle seguinte.
3. **Stop:** entrada − 1,5 × ATR(14).
4. **Alvo:** entrada + 2 × risco, ou seja, risco:retorno de **1:2**. Também há uma saída por tempo: 18 candles no 4h e 10 no diário.
5. **Tamanho:** 0,6% do capital ÷ distância do stop, limitado ao caixa livre e com mínimo de US$5. Uma posição por moeda e no máximo 3 na carteira.
6. **Filtros:**
   - o CRO (Vera) faz o pre-mortem e pode vetar;
   - o kill switch e o circuit breaker valem como sempre;
   - se houver uma resistência forte antes do alvo, a ordem sai com um alerta.
7. **Decisões possíveis:**
   - `EXECUTAR`;
   - `AGUARDAR`: o score está acima do limiar, mas o cruzamento já passou;
   - `NÃO OPERAR`: o score está abaixo do limiar;
   - `OBSERVAR`: o score ainda não foi validado nesse timeframe e, por isso, **não opera**.

## 4. Validação (`mqmas/research/confluence_study.py`)

O teste é fora da amostra.

1. Os pesos dos padrões vêm só do 1º período (2022-01 a 2023-06).
2. O limiar também é escolhido no 1º período, entre 40, 50, 60, 70 e 80.
3. O veredito sai do 2º período (2023-07 até hoje), que o sistema nunca viu.
4. Custos de 0,30% ida e volta estão incluídos.

Para ser **validado**, o limiar precisa de:
- 30 operações ou mais no 2º período;
- PF de pelo menos 1,2;
- expectativa **maior que a de entrar a cada 10 candles sem critério**, para separar o mérito do score da simples alta do mercado.

Testes de controle (`tests/test_ta_study.py`):
- uma vantagem real plantada nos dados é validada;
- ruído puro é rejeitado.
