"""NEWS BOT - cross-asset, market-neutral, news-driven paper trader.

WHAT IT IS
  Every trading day it reads free news headlines for 15 assets spanning five asset classes
  (crypto / equity indices / commodities / bonds / FX), scores each one's news tone with a
  finance lexicon, ranks the whole universe, then goes LONG the K best-news assets and SHORT
  the K worst-news assets. It never depends on any single stock or coin: the signal is the
  RELATIVE news standing of each asset against the others, and the long/short construction
  means a market-wide crash or melt-up cancels out.

WHY THREE ACCOUNTS
  Headline sentiment cannot be backtested for free - nobody gives away years of historical
  news tone. So instead of pretending, this bot runs a controlled experiment: three identical
  $10,000 accounts, same universe, same sizing, same exits, differing ONLY in what ranks the
  assets.
      news   - ranked by news tone  (the thing we want to test)
      mom    - ranked by 60-day price momentum (the control: what we already know works)
      blend  - ranked by news + momentum combined
  In six months the comparison is a real answer, not an opinion. If news ~ mom, the news
  added nothing. If blend beats both, news carries information price does not.

IT ALSO BUILDS A DATASET
  Every run appends each asset's news score to news_history.csv. That file is the historical
  sentiment data that does not exist for free - after a few months it becomes backtestable.

EXITS - the step-trail, the one mechanism we validated on daily bars:
  R = 3 x ATR(14). Stop starts at 1R. At +2R the stop moves to breakeven; every further +1R
  lifts the stop another 1R. Positions also close when they drop out of the ranking on the
  10-day rebalance.

  python news_paper.py --once                 # one daily tick (what GitHub Actions runs)
  python news_paper.py --dry                  # fetch + score + rank, trade nothing
"""
import os, sys, re, json, math, time, html, argparse, traceback
import urllib.request, urllib.parse
import numpy as np, pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
COST = 0.0005          # 0.05% round trip - daily bars, wide stops, so the drag is small
ARMS = ("news", "mom", "blend")

# ---------------------------------------------------------------- the universe
# ticker -> news query. Five asset classes so no single market drives the book.
UNIVERSE = {
    "BTC-USD":  "bitcoin",
    "ETH-USD":  "ethereum",
    "SOL-USD":  "solana crypto",
    "DOGE-USD": "dogecoin",
    "SPY":      "S&P 500 stock market",
    "QQQ":      "nasdaq technology stocks",
    "IWM":      "russell 2000 small cap stocks",
    "SMH":      "semiconductor chip stocks",
    "FXI":      "china stocks economy",
    "EEM":      "emerging markets stocks",
    "GLD":      "gold price",
    "SLV":      "silver price",
    "USO":      "oil price crude",
    "TLT":      "treasury bonds yields",
    "UUP":      "us dollar index",
}

# ------------------------------------------------------- the finance lexicon
# Deliberately small, transparent and offline - no API, no model, no key. Word counts on
# headlines. Crude, but every score it produces can be traced back to the words that caused it.
POS = set("""surge surges surged soar soars soared rally rallies rallied jump jumps jumped
climb climbs climbed gain gains gained rise rises rising rose advance advances advanced
rebound rebounds rebounded beat beats beating exceeded record high highs peak outperform
outperforms upgrade upgraded upgrades bullish bull boom booming strong stronger strength
optimism optimistic upbeat confidence growth grows expand expanding expansion profit profits
profitable breakthrough approval approved approves inflow inflows demand accelerate
accelerating acceleration recovery recover recovers recovering resilient resilience momentum
boost boosted boosts win wins winning favorable ease eases easing dovish stimulus deal deals
agreement truce ceasefire peace resolved resolution relief rebuilt adoption adopted
institutional breakout soaring surging topped tops milestone""".split())

NEG = set("""plunge plunges plunged tumble tumbles tumbled crash crashes crashed slump slumps
slumped sink sinks sank slide slides slid drop drops dropped fall falls falling fell decline
declines declined declining loss losses lose loses losing plummet plummets plummeted selloff
sell-off rout bearish bear weak weaker weakness fear fears fearful panic worry worries worried
concern concerns concerning downgrade downgraded downgrades miss misses missed warning warns
warned warning risk risks risky recession slowdown slowing contraction contract default
defaults bankruptcy bankrupt insolvent probe probes lawsuit lawsuits sue sued fraud scandal
hack hacked hacker breach exploit exploited stolen steals theft outflow outflows ban banned
bans crackdown tariff tariffs sanction sanctions sanctioned war warfare conflict strike
strikes attack attacks invasion escalation escalate escalates hawkish inflation inflationary
layoff layoffs collapse collapses collapsed crisis turmoil uncertainty volatile volatility
correction capitulation liquidation liquidated dump dumped halt halted freeze frozen
investigation delay delayed reject rejected rejects""".split())

# Headlines phrased as a question or a forecast carry no information - they are opinion filler.
JUNK = re.compile(r"(prediction|forecast|will .* be|should you|here's why|what to know|"
                  r"analyst|price target|outlook for|could|might|may hit)", re.I)


# ---------------------------------------------------------------- plumbing
def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(HERE, "news_bot.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sfile(arm):
    return os.path.join(HERE, "state_%s.json" % arm)


def fresh(cap):
    return {"equity": cap, "pos": {}, "last_day": "", "day_count": 0,
            "trades": 0, "wins": 0, "realized": 0.0}


def load_state(arm, cap):
    if os.path.exists(sfile(arm)):
        s = json.load(open(sfile(arm)))
        s.setdefault("realized", 0.0)
        return s
    return fresh(cap)


def save_state(arm, s):
    json.dump(s, open(sfile(arm), "w"), indent=2)


# ---------------------------------------------------------------- news
def headlines(query, timeout=25):
    """Google News RSS - free, keyless, works from GitHub Actions. Last 2 days of headlines."""
    u = ("https://news.google.com/rss/search?q=%s+when:2d&hl=en-US&gl=US&ceid=US:en"
         % urllib.parse.quote(query))
    raw = urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                 timeout=timeout).read().decode("utf8", "ignore")
    t = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw, re.S)
    if not t:
        t = re.findall(r"<title>(.*?)</title>", raw, re.S)
    out = []
    for x in t:
        x = html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
        # drop the feed's own titles and the query echo
        if not x or x == "Google News" or x.lower().startswith(query.lower()[:12]):
            continue
        out.append(x)
    return out[:60]


def tone(titles):
    """Signed tone in [-1,+1] plus the raw counts, so every score is auditable."""
    p = n = used = 0
    for t in titles:
        if JUNK.search(t):
            continue
        used += 1
        # strip the trailing " - Publisher" so outlet names never score
        body = t.rsplit(" - ", 1)[0] if " - " in t[-40:] else t
        for w in re.findall(r"[a-zA-Z\-']+", body.lower()):
            if w in POS:
                p += 1
            elif w in NEG:
                n += 1
    s = 0.0 if (p + n) == 0 else (p - n) / float(p + n)
    return s, p, n, used, len(titles)


def gather_news(tickers):
    rows = {}
    for t in tickers:
        try:
            h = headlines(UNIVERSE[t])
            s, p, n, used, tot = tone(h)
            rows[t] = {"score": s, "pos": p, "neg": n, "used": used, "total": tot}
            log("  news %-9s  score %+5.2f   pos %3d  neg %3d   headlines %2d/%2d"
                % (t, s, p, n, used, tot))
        except Exception as e:
            rows[t] = {"score": 0.0, "pos": 0, "neg": 0, "used": 0, "total": 0}
            log("  news %-9s  FAILED (%s) -> neutral" % (t, str(e)[:60]))
        time.sleep(0.8)          # be polite to the feed
    return rows


# ---------------------------------------------------------------- prices
def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def gather_prices(tickers):
    d = yf.download(" ".join(tickers), period="1y", interval="1d",
                    progress=False, auto_adjust=True, group_by="ticker", threads=False)
    out = {}
    for t in tickers:
        try:
            sub = d[t] if isinstance(d.columns, pd.MultiIndex) else d
            sub = sub.rename(columns=str.lower)[["high", "low", "close"]].dropna()
            if len(sub) < 70:
                log("  price %-9s only %d bars - skipped" % (t, len(sub)))
                continue
            out[t] = {"close": float(sub.close.iloc[-1]),
                      "atr": float(atr(sub.high, sub.low, sub.close).iloc[-1]),
                      "high": float(sub.high.iloc[-1]),
                      "low": float(sub.low.iloc[-1]),
                      "mom": float(np.log(sub.close.iloc[-1] / sub.close.iloc[-61])),
                      "date": str(sub.index[-1].date())}
        except Exception as e:
            log("  price %-9s FAILED (%s)" % (t, str(e)[:60]))
    return out


# ---------------------------------------------------------------- trading
def zscore(d):
    v = np.array(list(d.values()), float)
    if v.std() < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (x - v.mean()) / v.std() for k, x in d.items()}


def rank_for(arm, news_z, mom_z):
    if arm == "news":
        return news_z
    if arm == "mom":
        return mom_z
    return {k: 0.5 * news_z.get(k, 0.0) + 0.5 * mom_z.get(k, 0.0) for k in mom_z}


def close_pos(s, tkr, px, reason, arm):
    p = s["pos"].pop(tkr)
    pnl = p["side"] * (px - p["entry"]) * p["qty"] - 2 * COST * p["entry"] * p["qty"]
    s["equity"] += pnl
    s["realized"] += pnl
    s["trades"] += 1
    if pnl > 0:
        s["wins"] += 1
    log("  [%-5s] EXIT  %-9s %-5s @ %10.4f  %-9s PnL %+9.2f  equity %9.2f  (WR %d/%d)"
        % (arm, tkr, "LONG" if p["side"] == 1 else "SHORT", px, reason, pnl,
           s["equity"], s["wins"], s["trades"]))


def open_pos(s, tkr, side, px, R, risk, arm):
    qty = (s["equity"] * risk) / R
    s["pos"][tkr] = {"side": side, "entry": px, "R": R, "qty": qty, "peak": 0.0,
                     "stop": px - side * R, "opened": time.strftime("%Y-%m-%d")}
    log("  [%-5s] ENTER %-9s %-5s @ %10.4f  R %8.4f  qty %10.4f  stop %10.4f"
        % (arm, tkr, "LONG" if side == 1 else "SHORT", px, R, qty, px - side * R))


def manage_stops(s, prices, arm):
    """Step-trail: 1R stop -> breakeven at +2R -> every further +1R lifts the stop 1R."""
    for tkr in list(s["pos"].keys()):
        if tkr not in prices:
            continue
        p, q = s["pos"][tkr], prices[tkr]
        if p["side"] == 1:
            if q["low"] <= p["stop"]:
                close_pos(s, tkr, p["stop"], "stop", arm)
                continue
            p["peak"] = max(p["peak"], (q["high"] - p["entry"]) / p["R"])
        else:
            if q["high"] >= p["stop"]:
                close_pos(s, tkr, p["stop"], "stop", arm)
                continue
            p["peak"] = max(p["peak"], (p["entry"] - q["low"]) / p["R"])
        lvl = math.floor(p["peak"] - 2.0) if p["peak"] >= 2.0 else -1.0
        p["stop"] = p["entry"] + p["side"] * lvl * p["R"]


def rebalance(s, ranks, prices, a, arm):
    r = {k: v for k, v in ranks.items() if k in prices}
    if len(r) < 2 * a.k:
        log("  [%-5s] only %d rankable assets - no rebalance" % (arm, len(r)))
        return
    order = sorted(r, key=lambda k: r[k], reverse=True)
    longs, shorts = set(order[:a.k]), set(order[-a.k:])
    log("  [%-5s] LONG  %s" % (arm, ", ".join("%s(%+.2f)" % (t, r[t]) for t in order[:a.k])))
    log("  [%-5s] SHORT %s" % (arm, ", ".join("%s(%+.2f)" % (t, r[t]) for t in order[-a.k:])))
    for tkr in list(s["pos"].keys()):
        side = s["pos"][tkr]["side"]
        if (side == 1 and tkr not in longs) or (side == -1 and tkr not in shorts):
            if tkr in prices:
                close_pos(s, tkr, prices[tkr]["close"], "rotated", arm)
    for tkr in longs | shorts:
        if tkr in s["pos"]:
            continue
        R = a.atr_mult * prices[tkr]["atr"]
        if R <= 0:
            continue
        open_pos(s, tkr, 1 if tkr in longs else -1, prices[tkr]["close"], R, a.risk, arm)


def mark_to_market(s, prices):
    unreal = 0.0
    for tkr, p in s["pos"].items():
        if tkr in prices:
            unreal += p["side"] * (prices[tkr]["close"] - p["entry"]) * p["qty"]
    return s["equity"] + unreal


def append_history(day, news, prices, ranks_by_arm):
    """The dataset nobody sells for free: our own daily cross-asset sentiment record."""
    f = os.path.join(HERE, "news_history.csv")
    new = not os.path.exists(f)
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write("date,ticker,news_score,pos_words,neg_words,headlines_used,"
                     "headlines_total,close,atr14,mom60,news_z,mom_z\n")
        for t in sorted(news):
            if t not in prices:
                continue
            n, q = news[t], prices[t]
            fh.write("%s,%s,%.4f,%d,%d,%d,%d,%.6f,%.6f,%.5f,%.4f,%.4f\n"
                     % (day, t, n["score"], n["pos"], n["neg"], n["used"], n["total"],
                        q["close"], q["atr"], q["mom"],
                        ranks_by_arm["news"].get(t, 0.0), ranks_by_arm["mom"].get(t, 0.0)))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3, help="legs per side")
    p.add_argument("--risk", type=float, default=0.0075, help="equity risked per leg")
    p.add_argument("--atr_mult", type=float, default=3.0, help="R = this x ATR(14)")
    p.add_argument("--rebal", type=int, default=10, help="trading days between rebalances")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry", action="store_true", help="score and rank only, trade nothing")
    p.add_argument("--reset", action="store_true", help="wipe all three accounts")
    a = p.parse_args()

    tickers = list(UNIVERSE.keys())
    if a.reset:
        for arm in ARMS:
            save_state(arm, fresh(a.capital))
        log("=== all three accounts reset to $%.0f ===" % a.capital)
        return

    log("=" * 100)
    log("=== NEWS BOT tick | %d assets | K=%d/side | risk %.2f%%/leg | R=%.1fxATR | rebal %dd ==="
        % (len(tickers), a.k, a.risk * 100, a.atr_mult, a.rebal))

    prices = gather_prices(tickers)
    if not prices:
        log("no prices - aborting tick, nothing changed")
        return
    day = max(q["date"] for q in prices.values())
    log("  latest bar date: %s   (%d assets priced)" % (day, len(prices)))

    news = gather_news([t for t in tickers if t in prices])
    news_z = zscore({t: news[t]["score"] for t in news})
    mom_z = zscore({t: prices[t]["mom"] for t in prices})
    ranks = {arm: rank_for(arm, news_z, mom_z) for arm in ARMS}

    if a.dry:
        log("--- DRY RUN: rankings only, no trading ---")
        for arm in ARMS:
            o = sorted(ranks[arm], key=lambda k: ranks[arm][k], reverse=True)
            log("  [%-5s] LONG  %s   ||   SHORT %s"
                % (arm, ", ".join(o[:a.k]), ", ".join(o[-a.k:])))
        return

    for arm in ARMS:
        s = load_state(arm, a.capital)
        if s["last_day"] == day:
            log("  [%-5s] %s already processed - skipping" % (arm, day))
            continue
        manage_stops(s, prices, arm)
        s["day_count"] += 1
        due = (not s["pos"]) or (s["day_count"] % a.rebal == 1)
        if due:
            rebalance(s, ranks[arm], prices, a, arm)
        else:
            log("  [%-5s] holding %d positions, next rebalance in %d days"
                % (arm, len(s["pos"]), a.rebal - (s["day_count"] % a.rebal) + 1))
        s["last_day"] = day
        save_state(arm, s)
        mtm = mark_to_market(s, prices)
        log("  [%-5s] equity $%.2f realized (%+.2f%%)  |  $%.2f mark-to-market (%+.2f%%)  |  %d open"
            % (arm, s["equity"], (s["equity"] / a.capital - 1) * 100,
               mtm, (mtm / a.capital - 1) * 100, len(s["pos"])))

    append_history(day, news, prices, ranks)
    log("=== tick done ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        sys.exit(1)
