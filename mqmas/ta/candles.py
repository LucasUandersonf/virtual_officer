"""Candlestick patterns. Each detector returns a boolean Series marking the bar that *completes*
the pattern. Reversal patterns require the move they reverse (prior 5-bar trend), as in the books;
without that context a hammer is just a candle."""

from __future__ import annotations

import pandas as pd

TREND_BARS = 5


def _parts(df: pd.DataFrame) -> tuple[pd.Series, ...]:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - lo).replace(0, float("nan"))
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - lo
    avg_body = body.rolling(20).mean()
    return o, h, lo, c, body, rng, upper, lower, avg_body


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, lo, c, body, rng, upper, lower, avg_body = _parts(df)
    down_before = c.shift(1) < c.shift(1 + TREND_BARS)
    up_before = c.shift(1) > c.shift(1 + TREND_BARS)
    green, red = c > o, c < o
    small = body < 0.35 * rng
    long_body = body > 1.2 * avg_body
    p = pd.DataFrame(index=df.index)
    # bullish reversals
    p["martelo"] = down_before & small & (lower >= 2 * body) & (upper <= 0.25 * rng)
    p["martelo_invertido"] = down_before & small & (upper >= 2 * body) & (lower <= 0.25 * rng)
    p["engolfo_alta"] = (down_before & red.shift(1, fill_value=False) & green & (c >= o.shift(1))
                         & (o <= c.shift(1)) & (body > body.shift(1)))
    p["harami_alta"] = (down_before & red.shift(1, fill_value=False) & long_body.shift(1, fill_value=False) & green
                        & (o > c.shift(1)) & (c < o.shift(1)))
    p["estrela_manha"] = (down_before.shift(1, fill_value=False) & red.shift(2, fill_value=False)
                          & long_body.shift(2, fill_value=False) & (body.shift(1) < 0.5 * avg_body.shift(1))
                          & green & (c > (o.shift(2) + c.shift(2)) / 2))
    p["linha_perfuracao"] = (down_before & red.shift(1, fill_value=False) & long_body.shift(1, fill_value=False)
                             & green & (o < c.shift(1)) & (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1)))
    # bearish reversals
    p["estrela_cadente"] = up_before & small & (upper >= 2 * body) & (lower <= 0.25 * rng)
    p["homem_enforcado"] = up_before & small & (lower >= 2 * body) & (upper <= 0.25 * rng)
    p["engolfo_baixa"] = (up_before & green.shift(1, fill_value=False) & red & (c <= o.shift(1))
                          & (o >= c.shift(1)) & (body > body.shift(1)))
    p["harami_baixa"] = (up_before & green.shift(1, fill_value=False) & long_body.shift(1, fill_value=False) & red
                         & (o < c.shift(1)) & (c > o.shift(1)))
    p["estrela_noite"] = (up_before.shift(1, fill_value=False) & green.shift(2, fill_value=False)
                          & long_body.shift(2, fill_value=False) & (body.shift(1) < 0.5 * avg_body.shift(1))
                          & red & (c < (o.shift(2) + c.shift(2)) / 2))
    p["nuvem_negra"] = (up_before & green.shift(1, fill_value=False) & long_body.shift(1, fill_value=False) & red
                        & (o > c.shift(1)) & (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1)))
    # indecision / continuation
    p["doji"] = body <= 0.1 * rng
    p["piao"] = (body < 0.3 * rng) & (upper > body) & (lower > body) & ~p["doji"]
    first_up = green.shift(4, fill_value=False) & long_body.shift(4, fill_value=False)
    inside3 = pd.concat([(h.shift(k) <= h.shift(4)) & (lo.shift(k) >= lo.shift(4)) for k in (1, 2, 3)], axis=1).all(axis=1)
    p["tres_metodos_alta"] = first_up & inside3 & green & long_body & (c > c.shift(4))
    first_dn = red.shift(4, fill_value=False) & long_body.shift(4, fill_value=False)
    p["tres_metodos_baixa"] = first_dn & inside3 & red & long_body & (c < c.shift(4))
    return p.fillna(False).astype(bool)


BULLISH = ("martelo", "martelo_invertido", "engolfo_alta", "harami_alta", "estrela_manha", "linha_perfuracao",
           "tres_metodos_alta")
BEARISH = ("estrela_cadente", "homem_enforcado", "engolfo_baixa", "harami_baixa", "estrela_noite", "nuvem_negra",
           "tres_metodos_baixa")
NEUTRAL = ("doji", "piao")
