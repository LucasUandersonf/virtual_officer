# Catálogo de estratégias — Prates (2021) e Stanzione (10 padrões)

Cada estratégia é uma regra isolada e executável. O código está em `mqmas/research/livro.py` e a versão para máquinas em `docs/estrategias/catalogo.json`.

## Regras comuns a todas as estratégias

- **Execução:** sinal no fechamento da vela, ordem na abertura da vela seguinte. Timeframes 1d e 3d (velas de 3 dias). Custos de 0,10% de taxa + 0,05% de slippage por lado.
- **Stop inicial:** o mais distante entre o stop técnico da estratégia e 1,5 × ATR(14), nunca além de 3 × ATR(14).
- **Breakeven:** quando a máxima chega à entrada + 1R, o stop vai para a entrada + custos.
- **Trailing:** a partir de +2R, o stop = max(stop atual, mínima das últimas 3 velas, máxima do trade − 2,5 × ATR). Ele só sobe.
- **Alvo:** a altura do padrão somada ao rompimento (padrões gráficos) ou sem alvo, deixando o trailing conduzir. As duas versões são testadas.
- **Saídas extras:** qualquer padrão de baixa do catálogo, o sinal de venda do próprio modelo (TCC) e o tempo (40 velas no 1d, 20 no 3d).
- **Venda a descoberto:** a Binance spot não permite. Por isso os padrões de baixa são **regras de saída** e não abrem vendas.

## Estratégias

### `tcc_medias` — Cruzamento de Médias (Prates (2021)) — compra

- **Gatilho:** EMA20 cruza para cima a SMA50 no fechamento
- **Saída própria:** EMA20 cruza para baixo a SMA50
- **Stop técnico:** mínima das últimas 5 velas
- **Alvo:** sem alvo (trailing)

### `tcc_macd` — Cruzamento do MACD (Prates (2021)) — compra

- **Gatilho:** MACD(12,26) cruza para cima a linha de sinal(9)
- **Saída própria:** MACD cruza para baixo o sinal
- **Stop técnico:** mínima das últimas 5 velas
- **Alvo:** sem alvo (trailing)

### `tcc_estocastico` — Rompimento do Estocástico (Prates (2021)) — compra

- **Gatilho:** %K lento (5,3,3) cruza 20 para cima
- **Saída própria:** %K cruza 80 para baixo
- **Stop técnico:** mínima das últimas 5 velas
- **Alvo:** sem alvo (trailing)

### `tcc_fffd` — Fechou Fora Fechou Dentro (Bollinger 20,2) (Prates (2021)) — compra

- **Gatilho:** fechamento anterior abaixo da banda inferior e atual de volta dentro
- **Saída própria:** fechamento anterior acima da banda superior e atual de volta dentro
- **Stop técnico:** mínima das últimas 5 velas
- **Alvo:** média de 20 / sem alvo

### `oco_invertido` — Cabeça e ombros invertido (Stanzione #2) — compra

- **Gatilho:** 3 fundos confirmados (pivô 3), cabeça < ombros − 0,6 ATR, ombros iguais (±0,9 ATR); fechamento rompe a linha de pescoço (maior topo entre os ombros)
- **Stop técnico:** mínima do ombro direito (últimas 10 velas)
- **Alvo:** pescoço + (pescoço − cabeça)

### `fundo_duplo` — Fundo duplo (Stanzione #3) — compra

- **Gatilho:** 2 fundos iguais (±0,6 ATR, ≥ 5 velas entre eles); fechamento rompe o topo entre eles
- **Stop técnico:** mínima das últimas 10 velas (meio do movimento de alta)
- **Alvo:** pescoço + altura do W

### `xicara_alca` — Xícara e alça (Stanzione #5) — compra

- **Gatilho:** bordas iguais (±3 %), ≥ 15 velas, profundidade 12–50 %, alça ≤ 1/3 da xícara; rompe a borda
- **Stop técnico:** mínima da alça
- **Alvo:** borda + profundidade da xícara

### `fundo_arredondado` — Fundo de arredondamento (Stanzione #7) — compra

- **Gatilho:** em 60 velas: fundo no terço central, ≥ 10 % abaixo da borda esquerda, metade direita com mínimas subindo; fechamento rompe a borda esquerda
- **Stop técnico:** mínima das últimas 10 velas
- **Alvo:** borda + profundidade

### `triangulo_ascendente` — Triângulo ascendente (compra) (Stanzione #8) — compra

- **Gatilho:** 2 topos iguais (±0,6 ATR) e fundos subindo; fechamento rompe a resistência
- **Stop técnico:** último fundo
- **Alvo:** resistência + altura do triângulo

### `triangulo_descendente_alta` — Triângulo descendente (compra acima da LTB) (Stanzione #9) — compra

- **Gatilho:** 2 fundos iguais e topos caindo; fechamento rompe a linha dos topos
- **Stop técnico:** logo abaixo da linha rompida / último fundo
- **Alvo:** rompimento + altura do triângulo

### `cunha_descendente` — Cunha descendente (Stanzione #10) — compra

- **Gatilho:** topos e fundos caindo, topos caindo mais rápido (convergente); fechamento rompe a linha dos topos
- **Stop técnico:** último fundo da cunha
- **Alvo:** rompimento + altura do início da cunha

### `oco` — Cabeça e ombros (Stanzione #1) — saída

- **Gatilho:** spot sem venda a descoberto: rompimento da linha de pescoço para baixo FECHA compras

### `topo_duplo` — Topo duplo (Stanzione #4) — saída

- **Gatilho:** rompimento do fundo entre os topos FECHA compras

### `topo_arredondado` — Topo de arredondamento (Stanzione #6) — saída

- **Gatilho:** espelho do fundo arredondado; rompimento da borda esquerda para baixo FECHA compras

### `triangulo_descendente` — Triângulo descendente (venda) (Stanzione #9) — saída

- **Gatilho:** rompimento do suporte horizontal FECHA compras

### `cunha_ascendente` — Cunha ascendente (Stanzione #10) — saída

- **Gatilho:** topos e fundos subindo, fundos subindo mais rápido; rompimento da linha dos fundos FECHA compras

## Confluência (etapa 2)

As combinações testadas, cada uma com e sem o filtro macro, no 1d e no 3d:
- **pares:** A dispara e B também disparou nas últimas 3 velas (55 pares);
- **k-de-n:** pelo menos 2 ou 3 estratégias diferentes nas últimas 3 velas;
- **individuais:** com e sem alvo.

O total é de 300 candidatos.

**Filtro macro:** BTC acima da SMA100, moeda acima da EMA200 com EMA50 > EMA200, ADX ≥ 20 e ATR diário entre 1,5% e 8%.

**Seleção walk-forward:** a cada janela de 6 meses, o melhor candidato nos dados anteriores é operado às cegas na janela seguinte. O resultado do estudo é dessa seleção, fora da amostra.
