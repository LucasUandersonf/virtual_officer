"""The team's 100-pattern worklist (Bollinger · Fibonacci · Support/Resistance · harmonics · Wyckoff · SMC).

Every item becomes an exact, causal rule stamped on the bar that completes it. Shared geometry:

* **Bollinger** 20/2 on closes; squeeze = width in the lowest 20 % of the last 120 bars.
* **Support / resistance** = the last *confirmed* swing low / high (5-bar pivots).
* **Fibonacci**: the last confirmed up-leg (swing low → later swing high) gives retracements
  ``H − r·(H − L)`` for longs; the last down-leg gives ``L + r·(H − L)`` for the bearish side.
  A level is "touched" when the bar trades within 0.25 ATR of it and closes on the right side.
* **Harmonics (XABCD)** are checked on the alternating pivot sequence when the D pivot is confirmed.

Bias follows what the pattern *means*, not where it sat in the list: #35/#36 (extension targets) and
#49 are exit/sell signals, #50 (seller exhaustion at the 161.8 % extension) is a buy signal. On a
spot account, bearish items are used as exits or "do not buy", never as shorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.ta.structure import Pivot, pivots

TOL_ATR = 0.25


def _ffill_levels(n: int, piv: list[Pivot], kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Price, bar index of the extreme, and previous price of the last confirmed pivot of ``kind``."""
    price = np.full(n, np.nan)
    idx = np.full(n, -1, dtype=int)
    prev = np.full(n, np.nan)
    last_p, last_i, prev_p = np.nan, -1, np.nan
    events = sorted((p.conf, p.idx, p.price) for p in piv if p.kind == kind)
    j = 0
    for i in range(n):
        while j < len(events) and events[j][0] <= i:
            prev_p, last_i, last_p = last_p, events[j][1], events[j][2]
            j += 1
        price[i], idx[i], prev[i] = last_p, last_i, prev_p
    return price, idx, prev


def _harmonics(df: pd.DataFrame, piv: list[Pivot], bb_lo: np.ndarray, bb_up: np.ndarray) -> dict[str, np.ndarray]:
    n = len(df)
    out = {k: np.zeros(n, dtype=bool) for k in ("gartley_bull", "gartley_bear", "bat_bull", "butterfly_bull",
                                                "crab_bull", "abcd_bull", "abcd_bear")}
    seq: list[Pivot] = []
    for p in sorted(piv, key=lambda q: q.idx):  # alternate H/L keeping the extreme of a run
        if seq and seq[-1].kind == p.kind:
            if (p.kind == "H" and p.price > seq[-1].price) or (p.kind == "L" and p.price < seq[-1].price):
                seq[-1] = p
            continue
        seq.append(p)

    def r(a: float, b: float) -> float:
        return abs(a) / abs(b) if b else np.nan

    for k in range(4, len(seq)):
        x, a, b, c, d = seq[k - 4:k + 1]
        conf = d.conf
        if conf >= n:
            continue
        xa, ab, bc, cd, ad = a.price - x.price, b.price - a.price, c.price - b.price, d.price - c.price, d.price - a.price
        ab_xa, bc_ab, ad_xa = r(ab, xa), r(bc, ab), r(ad, xa)
        bull = d.kind == "L"
        if 0.58 <= ab_xa <= 0.66 and 0.382 <= bc_ab <= 0.886 and 0.74 <= ad_xa <= 0.82:
            key = "gartley_bull" if bull else "gartley_bear"
            band_ok = (d.price <= bb_lo[d.idx] * 1.002) if bull else (d.price >= bb_up[d.idx] * 0.998)
            out[key][conf] = bool(band_ok)
        if bull and 0.382 <= ab_xa <= 0.5 and 0.86 <= ad_xa <= 0.91:
            out["bat_bull"][conf] = True
        if bull and 0.74 <= ab_xa <= 0.82 and 1.2 <= ad_xa <= 1.35 and d.price < bb_lo[d.idx]:
            out["butterfly_bull"][conf] = True
        if bull and 0.382 <= ab_xa <= 0.618 and 1.55 <= ad_xa <= 1.7:
            out["crab_bull"][conf] = True
        # AB=CD on the last four pivots (a, b, c, d)
        if 0.618 <= bc_ab <= 0.786 and 0.9 <= r(cd, ab) <= 1.1:
            out["abcd_bull" if bull else "abcd_bear"][conf] = True
    return out


def worklist(df: pd.DataFrame, f: pd.DataFrame, cp: pd.DataFrame, struct: pd.DataFrame) -> pd.DataFrame:
    """``f`` = all_indicators(df), ``cp`` = candle_patterns(df), ``struct`` = structure_events(df)[0]."""
    o, h, lo, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    n = len(df)
    a = f["atr"]
    tol = TOL_ATR * a
    up, dn, mid = f["bb_up"], f["bb_lo"], f["bb_mid"]
    width = f["bb_width"]
    squeeze = width.shift(1) <= width.rolling(120, min_periods=60).quantile(0.2).shift(1)
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - lo
    green, red = c > o, c < o
    long_body = body > 1.2 * body.rolling(20).mean()
    pin_bull = (lower >= 0.66 * rng) & (body <= 0.3 * rng)
    pin_bear = (upper >= 0.66 * rng) & (body <= 0.3 * rng)
    inside = (h <= h.shift(1)) & (lo >= lo.shift(1))
    out_in_bull = (lo < dn) & (c > dn)
    out_in_bear = (h > up) & (c < up)
    closed_out_in_bull = (c.shift(1) < dn.shift(1)) & (h.shift(1) < dn.shift(1)) & (c > dn)
    closed_out_in_bear = (c.shift(1) > up.shift(1)) & (lo.shift(1) > up.shift(1)) & (c < up)
    near_lower = lo <= dn + tol
    near_upper = h >= up - tol

    piv = pivots(df, 5)
    sl, sl_i, sl_prev = _ffill_levels(n, piv, "L")
    sh, sh_i, sh_prev = _ffill_levels(n, piv, "H")
    sup, res = pd.Series(sl, index=df.index), pd.Series(sh, index=df.index)
    sup_prev, res_prev = pd.Series(sl_prev, index=df.index), pd.Series(sh_prev, index=df.index)
    false_break_sup = (lo < sup) & (c > sup)
    false_break_res = (h > res) & (c < res)
    # Fibonacci on the last confirmed up-leg (low before high) and down-leg (high before low)
    up_leg = pd.Series(sh_i > sl_i, index=df.index)
    rng_leg = res - sup

    def fib_bull(r: float) -> pd.Series:
        lvl = res - r * rng_leg
        return up_leg & (lo <= lvl + tol) & (c >= lvl - tol) & (c > lvl - tol)

    def fib_bear(r: float) -> pd.Series:
        lvl = sup + r * rng_leg
        return ~up_leg & (h >= lvl - tol) & (c <= lvl + tol)

    gp_lo, gp_hi = res - 0.65 * rng_leg, res - 0.618 * rng_leg
    in_pocket = up_leg & (lo <= gp_hi + tol) & (c >= gp_lo - tol)
    gp_bear = ~up_leg & (h >= sup + 0.618 * rng_leg - tol) & (c <= sup + 0.65 * rng_leg + tol)
    ext_up_1618 = up_leg & (h >= sup + 1.618 * rng_leg) & (h.shift(1) < sup + 1.618 * rng_leg)
    ext_up_1000 = up_leg & (h >= res + 1.0 * rng_leg) & (h.shift(1) < res + 1.0 * rng_leg)
    ext_dn_1618 = ~up_leg & (lo <= res - 1.618 * rng_leg) & (c > lo + 0.5 * (h - lo))
    trend_up = (c > f["ema50"]) & (f["ema50"] > f["ema200"])
    trend_dn = (c < f["ema50"]) & (f["ema50"] < f["ema200"])
    dbl_bottom, dbl_top = struct["fundo_duplo"], struct["topo_duplo"]
    harm = _harmonics(df, piv, dn.to_numpy(), up.to_numpy())
    H = {k: pd.Series(val, index=df.index) for k, val in harm.items()}

    dragonfly = (body <= 0.1 * rng) & (upper <= 0.1 * rng) & (lower >= 0.6 * rng)
    gravestone = (body <= 0.1 * rng) & (lower <= 0.1 * rng) & (upper >= 0.6 * rng)
    tweezer_bot = ((lo - lo.shift(1)).abs() <= 0.1 * a) & red.shift(1, fill_value=False) & green
    tweezer_top = ((h - h.shift(1)).abs() <= 0.1 * a) & green.shift(1, fill_value=False) & red
    belt_bull = green & long_body & (lower <= 0.05 * rng)
    belt_bear = red & long_body & (upper <= 0.05 * rng)
    crows = red & red.shift(1, fill_value=False) & red.shift(2, fill_value=False) & long_body & \
        long_body.shift(1, fill_value=False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    stars_south = red & red.shift(1, fill_value=False) & red.shift(2, fill_value=False) & \
        (body < body.shift(1)) & (body.shift(1) < body.shift(2)) & (lo >= lo.shift(1) - 0.1 * a)
    tower_bot = long_body.shift(3, fill_value=False) & red.shift(3, fill_value=False) & \
        (body.shift(1) < 0.5 * body.shift(3)) & (body.shift(2) < 0.5 * body.shift(3)) & green & long_body
    tower_top = long_body.shift(3, fill_value=False) & green.shift(3, fill_value=False) & \
        (body.shift(1) < 0.5 * body.shift(3)) & (body.shift(2) < 0.5 * body.shift(3)) & red & long_body
    walk_up = (c >= up * 0.995).rolling(4).sum() >= 3
    walk_dn = (c <= dn * 1.005).rolling(4).sum() >= 3
    hug_up = (c > up - 0.15 * (up - dn)).rolling(8).sum() >= 5
    hug_dn = (c < dn + 0.15 * (up - dn)).rolling(8).sum() >= 5
    box_h, box_l = h.shift(1).rolling(10).max(), lo.shift(1).rolling(10).min()
    darvas = ((box_h - box_l) <= 2.5 * a) & (box_l <= mid) & (mid <= box_h) & (c > box_h)
    range_lo, range_hi = lo.shift(1).rolling(30).min(), h.shift(1).rolling(30).max()
    flag_up, flag_dn = struct["bandeira_alta"], struct["bandeira_baixa"]
    mid_touch_up = (lo.rolling(5).min() <= mid + 0.5 * a)
    mid_touch_dn = (h.rolling(5).max() >= mid - 0.5 * a)
    pole_up = c.shift(5) - c.shift(15)
    pause_depth = (c.shift(5) - lo.rolling(5).min()) / pole_up.where(pole_up > 0)

    w: dict[str, pd.Series] = {
        "01_BB_Bullish_Trap_Classic": false_break_sup & out_in_bull,
        "02_BB_Hammer_Out_In": cp["martelo"] & out_in_bull,
        "03_BB_Engulfing_Lower": cp["engolfo_alta"] & near_lower,
        "04_BB_Inside_Bar_Lower": cp["harami_alta"] & near_lower.shift(1, fill_value=False),
        "05_BB_Pinbar_2B_Bottom": pin_bull & false_break_sup & (c > dn),
        "06_BB_Doji_Dragonfly_Lower": dragonfly & out_in_bull,
        "07_BB_Closed_Out_In_Bull": closed_out_in_bull,
        "08_BB_Squeeze_Fakeout_Down": squeeze & (lo < dn) & (c > dn),
        "09_BB_Inverted_Hammer_Floor": cp["martelo_invertido"] & near_lower,
        "10_BB_Piercing_Line_Lower": cp["linha_perfuracao"] & near_lower,
        "11_BB_Tweezer_Bottom_Out": tweezer_bot & (lo < dn) & (lo.shift(1) < dn.shift(1)),
        "12_BB_Tower_Bottom_Bollinger": tower_bot & (lo.rolling(4).min() < dn) & (c > mid - 0.5 * a),
        "13_BB_Three_Stars_South": stars_south & near_lower,
        "14_BB_Morning_Star_Lower": cp["estrela_manha"] & (lo.rolling(3).min() <= dn + tol),
        "15_BB_Belt_Hold_Bullish": belt_bull & (o < dn) & (c > dn),
        "16_BB_Bearish_Trap_Classic": false_break_res & out_in_bear,
        "17_BB_Shooting_Star_Out_In": cp["estrela_cadente"] & out_in_bear,
        "18_BB_Engulfing_Upper": cp["engolfo_baixa"] & near_upper,
        "19_BB_Inside_Bar_Upper": cp["harami_baixa"] & near_upper.shift(1, fill_value=False),
        "20_BB_Pinbar_2B_Top": pin_bear & false_break_res & (c < up),
        "21_BB_Doji_Gravestone_Upper": gravestone & out_in_bear,
        "22_BB_Closed_Out_In_Bear": closed_out_in_bear,
        "23_BB_Squeeze_Fakeout_Up": squeeze & (h > up) & (c < up),
        "24_BB_Hanging_Man_Roof": cp["homem_enforcado"] & near_upper,
        "25_BB_Dark_Cloud_Upper": cp["nuvem_negra"] & near_upper,
        "26_BB_Tweezer_Top_Out": tweezer_top & (h > up) & (h.shift(1) > up.shift(1)),
        "27_BB_Tower_Top_Bollinger": tower_top & (h.rolling(4).max() > up) & (c < mid + 0.5 * a),
        "28_BB_Three_Black_Crows_Upper": crows & (h.rolling(4).max() >= up),
        "29_BB_Evening_Star_Upper": cp["estrela_noite"] & (h.rolling(3).max() >= up - tol),
        "30_BB_Belt_Hold_Bearish": belt_bear & (o > up) & (c < up),
        "31_FIBO_618_Support_Macro": fib_bull(0.618) & ((sup - (res - 0.618 * rng_leg)).abs() <= a),
        "32_FIBO_500_Support_Macro": fib_bull(0.5) & ((sup - (res - 0.5 * rng_leg)).abs() <= 1.5 * a),
        "33_FIBO_382_Shallow_Retrace": fib_bull(0.382) & trend_up,
        "34_FIBO_786_Deep_Liquidation": fib_bull(0.786) & (lower >= 0.5 * rng),
        "35_FIBO_1618_Extension_Target": ext_up_1618,
        "36_FIBO_1000_Measured_Move": ext_up_1000,
        "37_FIBO_Golden_Pocket_Zone": in_pocket & green,
        "38_FIBO_Double_Bottom_618": dbl_bottom & (sup - (res - 0.618 * rng_leg)).abs().le(a),
        "39_FIBO_W_Pattern_Incline_500": dbl_bottom & (sup > sup_prev),
        "40_FIBO_OCOI_Head_786": struct["oco_invertido"] & up_leg,
        "41_FIBO_OCOI_Shoulder_618": struct["oco_invertido"] & in_pocket.rolling(10).max().astype(bool),
        "42_FIBO_ABC_Correction_Flat": fib_bull(0.5) & ((sup - sup_prev).abs() <= a),
        "43_FIBO_ABC_Correction_ZigZag": in_pocket & (sup < sup_prev) & green,
        "44_FIBO_2B_Setup_Fibo_Zone": false_break_sup & (fib_bull(0.5) | fib_bull(0.618)),
        "45_FIBO_Throwback_Polarity": struct["ob_alta_reteste"] & (fib_bull(0.382) | fib_bull(0.5) | fib_bull(0.618)),
        "46_FIBO_618_Resistance_Macro": fib_bear(0.618) & ((res - (sup + 0.618 * rng_leg)).abs() <= a),
        "47_FIBO_500_Resistance_Macro": fib_bear(0.5) & ((res - (sup + 0.5 * rng_leg)).abs() <= 1.5 * a),
        "48_FIBO_382_Shallow_Rebound": fib_bear(0.382) & trend_dn,
        "49_FIBO_786_Deep_Stop_Hunt": fib_bear(0.786) & (upper >= 0.5 * rng),
        "50_FIBO_1618_Bearish_Extension": ext_dn_1618,
        "51_FIBO_Double_Top_618": dbl_top & ~up_leg,
        "52_FIBO_M_Pattern_Incline_500": dbl_top & (res < res_prev),
        "53_FIBO_OCO_Head_786": struct["oco"] & ~up_leg,
        "54_FIBO_OCO_Shoulder_618": struct["oco"] & gp_bear.rolling(10).max().astype(bool),
        "55_FIBO_Pullback_Polarity": struct["ob_baixa_reteste"] & (fib_bear(0.382) | fib_bear(0.5) | fib_bear(0.618)),
        "56_COMBO_Bullish_Perfect_Storm": false_break_sup & out_in_bull & (fib_bull(0.618) | in_pocket),
        "57_COMBO_Hammer_Fibo_618_BB": (cp["martelo"] | pin_bull) & out_in_bull & (fib_bull(0.618) | in_pocket),
        "58_COMBO_Engulfing_Fibo_500_BB": cp["engolfo_alta"] & near_lower & fib_bull(0.5),
        "59_COMBO_Inside_Fibo_Pocket": (cp["harami_alta"] | inside) & green & in_pocket & near_lower,
        "60_COMBO_Divergence_W_Fibo_BB": dbl_bottom & (sup >= dn) & up_leg,
        "61_COMBO_Pinbar_Liquidity_Fibo": pin_bull & false_break_sup & out_in_bull & fib_bull(0.786),
        "62_COMBO_Closed_Out_In_Fibo": closed_out_in_bull & (fib_bull(0.5) | fib_bull(0.618) | fib_bull(0.786)),
        "63_COMBO_Squeeze_Break_Fibo_Up": squeeze & (c > up) & fib_bull(0.382).rolling(10).max().astype(bool),
        "64_COMBO_V_Bottom_Extension": ext_dn_1618 & (lo < dn) & green,
        "65_COMBO_Belt_Hold_Gold_Zone": belt_bull & in_pocket,
        "66_COMBO_Bearish_Perfect_Storm": false_break_res & out_in_bear & (fib_bear(0.618) | gp_bear),
        "67_COMBO_Shooting_Fibo_618_BB": (cp["estrela_cadente"] | pin_bear) & out_in_bear & (fib_bear(0.618) | gp_bear),
        "68_COMBO_Engulfing_Fibo_500_BB": cp["engolfo_baixa"] & near_upper & fib_bear(0.5),
        "69_COMBO_Inside_Bear_Fibo_Pocket": (cp["harami_baixa"] | inside) & red & gp_bear & near_upper,
        "70_COMBO_Divergence_M_Fibo_BB": dbl_top & (res <= up) & ~up_leg,
        "71_COMBO_Pinbar_Liquidity_Roof": pin_bear & false_break_res & out_in_bear & fib_bear(0.786),
        "72_COMBO_Closed_Out_In_Bear_Fibo": closed_out_in_bear & (fib_bear(0.5) | fib_bear(0.618) | fib_bear(0.786)),
        "73_COMBO_Squeeze_Break_Fibo_Down": squeeze & (c < dn) & fib_bear(0.618).rolling(10).max().astype(bool),
        "74_COMBO_V_Top_Extension": ext_up_1618 & (h > up) & red,
        "75_COMBO_Belt_Hold_Bear_Pocket": belt_bear & gp_bear,
        "76_TREND_BB_Central_Pullback_Bull": trend_up & (lo <= mid) & (c > mid) & green,
        "77_TREND_BB_Central_Pullback_Bear": trend_dn & (h >= mid) & (c < mid) & red,
        "78_TREND_Polarity_Fibo_500_MA": trend_up & fib_bull(0.5) & ((res - 0.5 * rng_leg - mid).abs() <= a),
        "79_TREND_Bandeira_Alta_MA20": flag_up & mid_touch_up,
        "80_TREND_Bandeira_Baixa_MA20": flag_dn & mid_touch_dn,
        "81_TREND_Walk_The_Band_Bull": walk_up & ~walk_up.shift(1, fill_value=False),
        "82_TREND_Walk_The_Band_Bear": walk_dn & ~walk_dn.shift(1, fill_value=False),
        "83_TREND_Power_Breakout_Upper": hug_up.shift(1, fill_value=False) & (c > h.shift(1).rolling(8).max()),
        "84_TREND_Power_Breakout_Lower": hug_dn.shift(1, fill_value=False) & (c < lo.shift(1).rolling(8).min()),
        "85_TREND_Mean_Reversion_Ignition": (c.shift(1) < mid.shift(1) - 2.5 * a.shift(1)) & green & (c > h.shift(1)),
        "86_TREND_Pivot_Alta_BB_Center": trend_up & (mid > mid.shift(5)) & (lo <= mid + tol) & (c > mid) & green
        & (lower >= body),
        "87_TREND_Pivot_Baixa_BB_Center": trend_dn & (mid < mid.shift(5)) & (h >= mid - tol) & (c < mid) & red,
        "88_TREND_Flamula_Alta_Fibo_382": flag_up & pause_depth.between(0.3, 0.45),
        "89_TREND_Flamula_Baixa_Fibo_382": flag_dn & fib_bear(0.382).rolling(8).max().astype(bool),
        "90_TREND_Darvas_Box_BB_Mid": darvas,
        "91_HARM_Gartley_Bullish_BB": H["gartley_bull"],
        "92_HARM_Gartley_Bearish_BB": H["gartley_bear"],
        "93_HARM_Bat_Pattern_Fibo_886": H["bat_bull"],
        "94_HARM_Butterfly_Extension_127": H["butterfly_bull"],
        "95_HARM_Crab_Extreme_1618": H["crab_bull"],
        "96_HARM_ABCD_Classic_Bull": H["abcd_bull"],
        "97_HARM_ABCD_Classic_Bear": H["abcd_bear"],
        "98_WYCK_Spring_Lower_Band": (lo < range_lo) & (c > range_lo) & (lo < dn),
        "99_WYCK_Upthrust_Upper_Band": (h > range_hi) & (c < range_hi) & (h > up),
        "100_SMC_Order_Block_Mitigation_Fibo": struct["ob_alta_reteste"] & in_pocket,
    }
    out = pd.DataFrame({k: s.fillna(False).astype(bool) for k, s in w.items()}, index=df.index)
    return out


BEARISH_IDS = set(range(16, 31)) | {35, 36} | set(range(46, 50)) | set(range(51, 56)) | set(range(66, 76)) | \
    {77, 80, 82, 84, 87, 89, 92, 97, 99}


def worklist_bias(name: str) -> str:
    num = int(name.split("_")[0])
    return "baixa" if num in BEARISH_IDS else "alta"
