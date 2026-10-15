# Deribit Market Dashboard

A Python research toolkit for **BTC and ETH options, futures, perpetuals and spot**, with a Streamlit dashboard. It collects public Deribit market data, preserves market snapshots, fits volatility surfaces, and evaluates portfolio stresses and option-hedging experiments.

The workflow connects three questions: what is the market quoting, what does a model imply, and what happens to a position when prices, volatility or funding change? Analyses can be saved with their original data and settings.

The application uses public market data and does not place orders. An API key is not required. Snapshots and saved experiments use a local SQLite database.

## What the project does

| Area | What you can explore |
|---|---|
| Market explorer | Discover BTC/ETH instruments, inspect bid/ask quotes, basis and funding, and view public ticker updates or local two-second quote aggregation. |
| Saved snapshots | Replay captured prices and contract metadata using their original timestamps. Export and import snapshots as JSON. |
| Quant laboratory | Compute independent implied volatility, infer forwards using put–call parity, audit quote quality, fit constrained total-variance smiles, and compare model prices and cash/coin Greeks. |
| Pricing models | Select Black–Scholes, Bachelier or binomial pricing with optional American exercise through the library's model layer. |
| Portfolio strategies | Build BTC/ETH pairs, option spreads and hypothetical barrier overlays; vary spot, volatility, correlation, basis, funding and collateral assumptions. Inspect P&L, drawdown and simulated risk metrics. |
| Barrier comparison | Compare Black–Scholes, local-volatility and Heston barrier valuations, touch probabilities, vanilla replication and perpetual hedges. |
| Walk-forward evaluation | Calibrate on an earlier recorded option chain and evaluate against later quotes, reporting unavailable observations and hypothetical execution outcomes. |
| Relative-Value Trading Evaluation | Test independent BTC vanilla-option edges with captured-side executions, an initial perpetual delta hedge, costs, chronological validation/test periods and simple benchmarks. |
| Research experiments | Save inputs, settings, results, diagnostics and software provenance; compare runs and reproduce them with the current code. |

Pricing, fitting and risk calculations live in the Python library. Streamlit provides controls and renders the results. The core package does not require Streamlit; the dashboard is an optional installation extra.

The market explorer displays captured exchange IV and Greeks. **Quant laboratory computes its own IV and model risk** and uses exchange values for comparison.
