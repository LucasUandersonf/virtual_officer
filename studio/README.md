# Product Studio: escritório virtual autônomo

Escritório de cinco agentes de IA que pesquisam dores de solopreneurs no mercado global, votam ideias em reunião, constroem produtos (workflows n8n e microsserviços FastAPI), escrevem o marketing em inglês, publicam no Stripe, no Gumroad e no Whop, e entregam o download ao comprador. Você acompanha tudo como CEO num escritório isométrico ao vivo, com receita, custos (incluindo a Railway) e lucro em dólar.

![Studio Floor](docs/office.png)

> **O que já está pronto e o que fica com os agentes.** Este diretório é a empresa montada: o escritório, os cinco agentes, as integrações, o banco, o painel e o deploy. **Nenhum produto vem pronto.** Quem cria os produtos é o Theo, o engenheiro, a partir das ideias que a Iris aprova em reunião, assim que o escritório liga.

---

## 1. A equipe

| Agente | Sala | O que faz | Como vota nas reuniões |
|---|---|---|---|
| **Iris**, Head of Product | Research Lab | Minera Reddit, Hacker News e Product Hunt; classifica cada post numa dor conhecida; pontua a demanda; propõe um MVP com preço em USD | Aprova se o score de evidência for ≥ 2 |
| **Theo**, Senior Dev & Builder | Dev Room | Gera o workflow n8n (JSON importável) ou o serviço FastAPI, passa pelo QA e empacota o `.zip` com README em inglês, `.env.example` e scripts | Aprova se o build cabe em ≤ 8 h |
| **Maya**, Growth & Copy | Growth Studio | Landing page, texto do checkout, 4 e-mails de onboarding e nutrição, posts para X, LinkedIn e Reddit | Aprova com ≥ 2 sinais ou score ≥ 4 |
| **Leo**, Account Executive & Ops | Sales Desk | Publica nas plataformas, recebe webhooks de pagamento, gera o link de download e manda o e-mail ao comprador | Ajusta o preço para a faixa de US$ 19 a 149 |
| **Nina**, CFO & Metas | Finance | Calcula se a ideia se paga (receita esperada contra a fatia do custo do escritório), fecha o dia, define o foco da semana | **Veto**: rejeita o que não paga sua parte |

**Regra das reuniões:** os votos saem de números calculados em código, e a IA só escreve as falas. Uma ideia é aprovada com 4 de 5 votos **e** o voto da Nina. Cada reunião custa uma única chamada de IA.

**Aprendizado:** a pontuação de cada tema soma a evidência dos últimos 30 dias, ganha +3 por venda já feita naquele tema e perde pontos quando um produto publicado há mais de 21 dias não vende. A reunião semanal escolhe os três temas de foco da semana.

## 2. O ciclo (a cada `PIPELINE_INTERVAL_MIN`)

```
fechamento do dia (custos Railway acumulados por dia)
  → pesquisa (a cada RESEARCH_INTERVAL_MIN) → ideias
  → reunião semanal de metas (a cada 7 dias)
  → reunião de revisão de ideia (aprova / rejeita)
  → Theo constrói + QA → Maya escreve o kit → produto "pronto"
  → AUTOPUBLISH=true: Leo publica · false: aguarda 1 clique seu ("Aprovar e publicar")
  → venda (webhook) → registro idempotente → link de download por e-mail → alerta no Telegram/Discord
```

**Travas de qualidade antes de vender:**
- **n8n:** o JSON é montado a partir de modelos de nó com `typeVersion` reais. A IA só escolhe os passos. O QA confere gatilho, nomes únicos, conexões e chaves balanceadas no código, e documenta todas as variáveis `$env`.
- **Python:** o código é analisado sem ser executado. Exige `process(payload)` e bloqueia `subprocess`, `eval`, `exec`, `os.system` e afins. Se o código da IA reprovar, entra o código do playbook.

## 3. Custo zero de IA

As IAs são todas de plano gratuito, em cascata: **Groq → Gemini → OpenRouter (modelos `:free`) → playbook interno**. O playbook (`app/agents/playbook.py`) traz 12 dores recorrentes com produto, preço, passos de workflow e código prontos. **Sem nenhuma chave, o escritório funciona do mesmo jeito** (com textos mais genéricos). `LLM_DAILY_CALL_BUDGET` limita as chamadas por dia para não estourar os planos gratuitos.

As fontes de pesquisa também são gratuitas. O X/Twitter (API paga) e o Google Trends (não tem API oficial) ficaram de fora de propósito.

## 4. Painel executivo (API)

| Endpoint | Retorno |
|---|---|
| `GET /api/dashboard/summary` | Receita, taxas, custos e lucro (total, mês e hoje), metas, custos fixos, plataformas, status das IAs, estado de cada agente |
| `GET /api/dashboard/sales/daily?days=30` | Vendas, receita, custos e lucro por dia |
| `GET /api/products` · `/api/products/{id}` | Produtos ativos, anúncios, vendas e kit de marketing |
| `GET /api/ideas` · `/api/signals` | Ideias com evidências (links para os posts originais) |
| `GET /api/meetings` · `/api/activity` | Atas com votos; atividade por agente |
| `GET /api/sectors` | Um painel por setor |
| `POST /api/products/{id}/approve` · `POST /api/run/{tick\|research}` | Ações do CEO (header `X-Admin-Token`) |
| `POST /webhooks/stripe` · `/webhooks/gumroad?key=` · `/webhooks/whop` | Vendas (assinatura verificada) |
| `GET /p/{slug}` · `GET /d/{token}` | Landing page do produto · download do comprador (expira) |
| `WS /ws` | Eventos ao vivo do escritório |

## 5. Plataformas de venda: o que é automático

| Plataforma | Publicação | Vendas |
|---|---|---|
| **Stripe** (recomendado) | 100% automática: Product → Price → Payment Link | Webhook `checkout.session.completed` (e `charge.refunded`) |
| **Whop** | Automática via API v1 (`/products` + `/plans` one_time). Confira os campos na doc do Whop: não consegui acessá-la deste ambiente | Webhook `payment.succeeded` (HMAC) |
| **Gumroad** | Tenta `POST /v2/products`. Esse endpoint não aparece em todas as versões da doc, então, se falhar, o anúncio vira "manual" e o Leo deixa título, descrição, preço e `.zip` prontos para você subir | Ping de venda em `/webhooks/gumroad?key=…` |

A taxa do Stripe é estimada (2,9% + US$ 0,30, configurável). O painel marca esse valor como estimado.

## 6. Rodar localmente

```bash
cd studio
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # tudo opcional
uvicorn app.main:app --reload    # http://localhost:8000
pytest -q && ruff check app tests
```

Stack completa (Postgres + Redis + n8n): `docker compose up --build` (escritório em :8000, n8n em :5678).

## 7. Deploy na Railway (econômico)

1. **New Project → Deploy from GitHub repo** → este repositório. No serviço, em **Settings → Root Directory**, coloque `studio`. O `studio/railway.json` cuida do resto (Dockerfile, healthcheck `/health`, 1 réplica).
2. **+ New → Database → PostgreSQL.** No serviço do escritório, crie a variável `DATABASE_URL=${{Postgres.DATABASE_URL}}`.
3. **Volume:** monte um volume em `/data`, onde ficam os `.zip` dos produtos.
4. **Variables:** `ADMIN_TOKEN`, `PUBLIC_BASE_URL` (o domínio gerado pela Railway) e as chaves gratuitas de IA que quiser (Groq/Gemini/OpenRouter). Para vender: `STRIPE_SECRET_KEY` + `STRIPE_WEBHOOK_SECRET`, com o webhook apontando para `https://<domínio>/webhooks/stripe`. Para entregar: `RESEND_API_KEY`. Para os alertas: Telegram ou Discord.
5. **Settings → Networking → Generate Domain.** Abra o domínio: é o escritório.

**Custo mensal estimado:** web ~US$ 5 + Postgres ~US$ 1 a 3. Redis e n8n são **opcionais**: o escritório usa uma fila em memória e entrega os workflows como arquivo, então não precisa de n8n próprio. Se você adicionar esses serviços, informe os valores em `RAILWAY_REDIS_USD` e `RAILWAY_N8N_USD` para o P&L contar. Os valores `RAILWAY_*_USD` são estimativas que entram no custo diário. Confira o consumo real em **Railway → Usage** e ajuste.

**Escalar com Redis (opcional):** com `REDIS_URL` definido, rode um segundo serviço com o comando `python -m app.worker` e coloque `AGENTS_IN_WEB=false` no serviço web. Os agentes passam a rodar no worker, e o web só serve o escritório e retransmite os eventos.

## 8. Estrutura

```
studio/
├── Dockerfile · railway.json · docker-compose.yml · .env.example
├── app/
│   ├── main.py            # FastAPI: escritório, API, webhooks, WebSocket, loop dos agentes
│   ├── worker.py          # worker separado (Redis) ou --once (cron)
│   ├── config.py · db.py
│   ├── models/            # SQLAlchemy: sinais, ideias, produtos, kits, anúncios, vendas, custos, reuniões
│   ├── agents/            # personas, playbook, Iris, Theo, Maya, Leo, Nina, office (reuniões + pipeline)
│   ├── integrations/      # pesquisa (Reddit/HN/PH), pagamentos (Stripe/Gumroad/Whop), notificações (Telegram/Discord/Resend)
│   ├── llm/router.py      # cascata de IAs gratuitas com validação JSON e fallback
│   ├── services/          # finanças (P&L), n8n (builder + QA), empacotamento, fila/agendador, eventos
│   ├── api/               # painel executivo + páginas públicas
│   └── web/static/        # escritório isométrico (motor CSS compartilhado com o MQ-MAS)
└── tests/
```

## 9. Limites honestos

- **Autonomia de publicação:** `AUTOPUBLISH=false` por padrão. Revise os primeiros produtos antes de ligar: vender algo gerado por IA sem olhar arrisca reembolsos e reclamações nas plataformas.
- **O modelo de demanda da Nina é conservador e arbitrário** (0,5 venda por mês + 0,25 por ponto de score) até existirem vendas reais. A partir daí, quem manda são os dados de venda por tema.
- **Reddit a partir da nuvem** costuma bloquear sem OAuth. Crie um app "script" gratuito e preencha `REDDIT_CLIENT_ID` e `REDDIT_CLIENT_SECRET`.
- **Esquema do banco:** `create_all` cria as tabelas novas, mas não altera colunas existentes. Mudanças de esquema depois precisam de migração (Alembic).
- **Os workflows n8n** usam `$env`. No n8n self-hosted isso exige `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` (o README de cada produto explica).
