# Alocador de tendência a custo quase zero

> **Atualização:** o runner agora opera dois gestores (tendência e caçador de ondas) com um comitê de especialistas. Veja **[SISTEMA.md](SISTEMA.md)** para o funcionamento completo, os resultados e a configuração recomendada (70% tendência + 30% ondas).

Este guia põe no ar o **alocador de tendência** (`python -m mqmas.runner`) como um *cron job*: o processo liga uma vez por hora, roda por cerca de um minuto e desliga. O escritório virtual (o servidor web que fica ligado 24 h) continua disponível, mas é opcional.

## 1. O que é a estratégia e de onde ela veio

É uma cópia (engenharia reversa) do sistema lucrativo mais bem documentado que encontramos para cripto: Zarattini, Pagani & Barbon (2025), *Catching Crypto Trends* ([SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)). É a mesma família do único vencedor da nossa pesquisa (o rompimento diário), sem as partes que falhavam:

| Peça | Regra | Onde está |
|---|---|---|
| Sinal | Média de 9 canais Donchian (5, 10, 20, 30, 60, 90, 150, 250 e 360 dias). Cada canal compra no fechamento acima da máxima do período e sai no fechamento abaixo do meio do canal, que só sobe | `mqmas/trend/model.py` |
| Tamanho | Sinal ÷ volatilidade de 90 dias; no máximo 35% por moeda; a cesta inteira é reduzida até a volatilidade esperada ficar em 40% ao ano. Nunca passa de 100% da banca (spot, sem alavancagem) | `mqmas/trend/model.py` |
| Universo | Todo mês, as 10 moedas mais líquidas com pelo menos 1 ano de histórico (sem stablecoins nem ações tokenizadas) | `mqmas/trend/model.py` |
| Ordens | Mínimo de US$ 5 da Binance; só ajusta uma posição se ela estiver mais de 25% fora do alvo | `mqmas/trend/rebalance.py` |
| Proteção | Stop *stop-limit* **na Binance**, 6 ATR abaixo do fechamento, que só sobe. Cobre os momentos em que o bot está desligado. Uma moeda que bate no stop só volta com um rompimento novo | `mqmas/trend/live.py` |
| Travas | Queda de 35% desde o pico: vende tudo e para até um reset manual. `TREND_KILL_SWITCH=true` vende tudo na próxima execução | `mqmas/trend/live.py` |

Nenhum parâmetro foi otimizado nos nossos dados. Os números vêm do artigo ou do mandato, e as variações aparecem no relatório só como sensibilidade.

## 2. O que o backtest mostra (dados reais da Binance, 2020-01 → 2026-10)

Taxa de 0,10% + derrapagem de 0,05% por lado, ordem mínima de US$ 5, stop na corretora, 40 moedas.

| Conta | CAGR | Queda máx. | Sharpe | Exposição média |
|---|---|---|---|---|
| US$ 50 | +10,0% | 17% | 0,86 | 12% |
| **US$ 120** | **+26,7%** | **33%** | **1,07** | 30% |
| Ideal (sem limite de ordem mínima) | +29,1% | 34% | 1,13 | 31% |
| Segurar BTC | +42,3% | 77% | 0,89 | 100% |
| Cesta igual das mesmas 10 moedas | +47,7% | 82% | 0,90 | 100% |

Ano a ano (conta ideal): 2020 +97% · 2021 +104% · **2022 −17%** (BTC −64%) · 2023 +35% · 2024 +49% · **2025 −14%** (BTC −6%) · 2026 até out. −1% (BTC −9%).

Leitura honesta:

- **O ganho está no risco, não no retorno.** O sistema rende menos que segurar BTC num mercado de alta, mas a queda máxima cai pela metade. O Sharpe sobe de 0,89 para 1,13.
- **Desde jul/2023 o retorno ficou abaixo do BTC** (+18% contra +34% ao ano), com o mesmo Sharpe (0,87 contra 0,86) e com queda máxima de 25% contra 53%.
- **Abaixo de ~US$ 120 não funciona bem:** a ordem mínima de US$ 5 deixa a banca quase toda parada.
- **Viés de sobrevivência:** as moedas são as mais negociadas de hoje, e as que morreram desde 2019 ficaram de fora. Isso infla todos os resultados *long-only*, inclusive as referências.
- O relatório completo fica em `trend_report.json` (veja a seção 5).

## 3. Custos

| Item | Antes | Agora |
|---|---|---|
| Hospedagem | Railway Hobby, servidor 24 h: ~US$ 5/mês | Railway **Free** + *cron* de hora em hora: ~US$ 0,10–0,30/mês, dentro do crédito de US$ 1 do plano gratuito |
| Pesquisa | Rodava no boot do servidor (CPU paga) | GitHub Actions (gratuito): `.github/workflows/research.yml` |
| IA (LLMs) | Planos gratuitos | O alocador não usa LLM (é determinístico). Os agentes continuam no escritório, se você quiser mantê-lo |
| Alertas | — | Telegram + healthchecks.io (os dois gratuitos) |
| Taxas da Binance | 0,10% por lado | ~2–5% da banca por ano no backtest. Pagar taxas com BNB reduz para 0,075% (ajuste `FEE_RATE=0.00075`) |

## 4. Passo a passo no Railway

> O Railway precisa de um servidor fora dos EUA: a Binance recusa IPs americanos. O serviço atual já roda em Amsterdã.

1. **Faça o merge deste branch na `main`.**
2. Na mesma tela do serviço `mq-mas` (ou num serviço novo, a partir do mesmo repositório):
   - **Settings → Config-as-code → Railway config file:** `railway.runner.toml`
   - Confira se o **volume** continua montado em `/data` (o estado fica em `/data/mqmas.db`).
   - Isso troca o servidor 24 h pelo *cron* `7 * * * *` (todo minuto 7 de cada hora, em UTC).
3. **Variables:** `TREND_CAPITAL_USDT=120` (ou o valor da sua banca) e, opcionalmente, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` e `HEARTBEAT_URL`. Remova `RESEARCH_ON_BOOT` e `REVIEW_ON_BOOT`, que só valem para o servidor web.
4. Com o serviço rodando só como *cron*, mude o workspace para o **plano Free** (Workspace → Plans). Plano Free: 0,5 GB de RAM e 0,5 GB de volume, o suficiente para o runner.
5. Acompanhe os logs da primeira execução: o resumo sai em JSON e, com Telegram configurado, você recebe cada compra, venda e stop.

### Alternativa sem Railway

Qualquer máquina **fora dos EUA** que fique ligada serve (um PC em casa, uma VM *Always Free* da Oracle Cloud na região São Paulo etc.):

```bash
pip install -r requirements.txt
cp .env.example .env    # preencha
crontab -e
7 * * * * cd /caminho/virtual_officer && .venv/bin/python -m mqmas.runner >> runner.log 2>&1
```

## 5. Do paper para o real

1. Rode em **paper** (padrão) por pelo menos 8 semanas. `python -m mqmas.runner --status` mostra o estado sem enviar ordens.
2. Compare o resultado do paper com o `trend_report.json` (aba **Actions → research → artefato `research-trend`**).
3. Para operar de verdade, as **duas chaves** precisam estar ligadas: `TRADING_MODE=live` + `LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISKS` **e** `TREND_LIVE_ENABLED=true`. Use uma chave de API só com Spot, **sem saque**.
4. O alocador nunca usa mais que `TREND_CAPITAL_USDT` e o lucro dele, e confere os saldos reais da corretora a cada rebalanceamento.
5. Se ele parar por queda máxima, entenda o motivo antes de liberar. No Railway: troque o *Start Command* por `python -m mqmas.runner --reset-halt`, rode uma vez (**Deploy**) e volte o comando para `python -m mqmas.runner`. Desligar o `TREND_KILL_SWITCH` libera o alocador sozinho na próxima execução.

> Nada disso garante lucro. O backtest mostra o que teria acontecido com estas regras, não o que vai acontecer.
