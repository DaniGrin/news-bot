# news-bot — cross-asset, market-neutral, news-driven paper trader

Paper money only. No API keys, no exchange account, no credit card. Runs free on GitHub Actions,
once a day after the US close (22:00 UTC, Mon–Fri).

## What it does

Every trading day it reads free news headlines for **15 assets across five asset classes**,
scores each one's news tone with an offline finance lexicon, ranks the whole universe, then goes
**long the 3 best-news assets and short the 3 worst-news assets**.

| class | assets |
|---|---|
| crypto | BTC, ETH, SOL, DOGE |
| equity indices | SPY, QQQ, IWM |
| sectors / regions | SMH, FXI, EEM |
| commodities | GLD, SLV, USO |
| bonds / FX | TLT, UUP |

It is **not dependent on any single stock or coin**: the signal is each asset's *relative* news
standing against the others, and the long/short construction means a market-wide crash or
melt-up largely cancels out.

## The three accounts

Headline sentiment can't be backtested for free — nobody gives away years of historical news
tone. So instead of guessing, the bot runs a controlled experiment. Three identical **$10,000**
accounts, same universe, same sizing, same exits, differing **only** in what ranks the assets:

| account | ranked by | role |
|---|---|---|
| `news` | news tone | the thing being tested |
| `mom` | 60-day price momentum | the control — what we already know works |
| `blend` | news + momentum, 50/50 | does news add anything on top? |

After a few months the comparison is an answer, not an opinion:

- `news` ≈ `mom` → the news added nothing.
- `news` > `mom` → headline tone carries real information.
- `blend` > both → news carries information price does not.

## Signal

Google News RSS (free, keyless), last 2 days of headlines per asset. Forecast/opinion
headlines ("prediction", "price target", "analyst", "could", "should you…") are discarded —
they're filler. The rest are word-counted against a positive/negative finance lexicon:

```
tone = (positive_words − negative_words) / (positive_words + negative_words)   ∈ [−1, +1]
```

Then z-scored across the universe, so only relative standing matters. Crude, but every score
is traceable to the exact words that produced it — no black box.

## Risk and exits

- `R = 3 × ATR(14)` — wide, daily-bar stops. Tight stops die to fees; see the note below.
- 0.75% of equity risked per leg, 6 legs → ~4.5% gross risk.
- **Step-trail exit:** stop starts at 1R. At **+2R** it moves to breakeven; every further
  **+1R** lifts the stop another **1R**. This is the one exit mechanism that has held up in
  forward testing — and it only works on daily bars, where it has time to activate.
- Rebalance every 10 trading days; positions that drop out of the ranking are closed.
- Costs modeled at 0.05% round trip.

> **Why daily and not intraday.** Required gross edge just to break even is
> `2 × cost × (entry / R)`. An intraday stop makes `entry/R` huge (~270), so the fee drag
> swamps any edge. Daily bars with a 3×ATR stop keep `entry/R` small and the drag negligible.

## It also builds a dataset

Every run appends one row per asset to **`news_history.csv`**: tone, the positive/negative word
counts, headline counts, close, ATR, 60-day momentum, and both z-scores. That's historical
cross-asset sentiment data that isn't available for free — after a few months it becomes
backtestable. This file is arguably worth more than the three accounts.

## Honest caveat

Before building this, the *measurable* news channel was tested properly: English Wikipedia
pageviews as an attention proxy (keyless, daily, back to 2015) against 15 assets, 2021–2026.
Pooled, high attention looked strongly predictive (+0.95% over 20 days, t = +4.7). **It was
entirely market beta.** Cross-sectionally demeaned, every bucket collapsed to noise
(|t| < 0.2), the attention cross-section was flat to negative, and plain price momentum with
no news at all beat it. So: there is no evidence yet that a news signal works. Headline *tone*
is a different channel from news *volume* and remains untested — which is exactly why this bot
ships with a control arm instead of a promise.

## Files

| file | what |
|---|---|
| `news_paper.py` | the whole bot |
| `state_news.json`, `state_mom.json`, `state_blend.json` | the three accounts |
| `news_history.csv` | the growing sentiment dataset |
| `news_bot.log` | every score, entry and exit |

## Running it yourself

```bash
pip install yfinance pandas numpy
python news_paper.py --dry      # score + rank only, trade nothing
python news_paper.py --once     # one daily tick
python news_paper.py --reset    # wipe all three accounts back to $10,000
```
