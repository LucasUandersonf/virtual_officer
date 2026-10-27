"""Technical-analysis engine: indicators, candlestick patterns, chart patterns and market structure.

Every detector is vectorised over an OHLCV frame and only uses information available at the close
of each bar (pivots are reported at the bar where they become *confirmed*, never earlier), so the
same code drives the live analysis and the historical study of which patterns actually work.
"""
