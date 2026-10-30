"""Does news/public ATTENTION predict forward returns? Keyless, fully historical test.

Attention proxy = English Wikipedia daily pageviews for each asset's topic page
(Wikimedia REST API: free, no key, daily data back to 2015).
Prices = yfinance daily.

Test 1 (time-series): bucket each asset's attention z-score, measure forward returns.
Test 2 (cross-section): each day rank the universe by attention z, go long top-K /
short bottom-K, hold H days. This is the asset-agnostic version.
"""
import os, json, time, urllib.request, sys
import numpy as np, pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "attention-research/1.0 (danisd422@gmail.com)"}

# ticker -> Wikipedia article title (the public-attention page for that asset/theme)
UNIVERSE = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "DOGE-USD": "Dogecoin",
    "SOL-USD":  "Solana_(blockchain_platform)",
    "SPY":      "S&P_500",
    "QQQ":      "Nasdaq-100",
    "GLD":      "Gold",
    "SLV":      "Silver",
    "USO":      "Price_of_oil",
    "TLT":      "United_States_Treasury_security",
    "SMH":      "Semiconductor_industry",
    "FXI":      "Economy_of_China",
    "EEM":      "Emerging_market",
    "XLE":      "Petroleum_industry",
    "UUP":      "United_States_dollar",
}

START, END = "20210101", "20261029"


def wiki_views(article):
    f = os.path.join(CACHE, "wiki_%s.csv" % article.replace("/", "_"))
    if os.path.exists(f):
        return pd.read_csv(f, index_col=0, parse_dates=True)["views"]
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           "en.wikipedia/all-access/user/%s/daily/%s/%s" % (article, START, END))
    j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
    idx = pd.to_datetime([it["timestamp"][:8] for it in j["items"]])
    s = pd.Series([it["views"] for it in j["items"]], index=idx, name="views")
    s.to_frame().to_csv(f)
    time.sleep(0.4)
    return s


def prices(tkr):
    f = os.path.join(CACHE, "px_%s.csv" % tkr)
    if os.path.exists(f):
        return pd.read_csv(f, index_col=0, parse_dates=True)["close"]
    d = yf.download(tkr, start="2020-10-01", end="2026-10-30", interval="1d",
                    progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    s = d["Close"].rename("close").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.to_frame().to_csv(f)
    return s


def main():
    views, px = {}, {}
    for t, art in UNIVERSE.items():
        try:
            views[t] = wiki_views(art)
            px[t] = prices(t)
            print("  loaded %-9s %-34s views %d  px %d" % (t, art, len(views[t]), len(px[t])))
        except Exception as e:
            print("  SKIP %s: %s" % (t, e))
    V = pd.DataFrame(views).sort_index()
    P = pd.DataFrame(px).sort_index()

    # attention z-score vs its own trailing 60d baseline (log views = heavy tails)
    LV = np.log(V.clip(lower=1))
    Z = (LV - LV.rolling(60).mean()) / LV.rolling(60).std()
    Z = Z.reindex(P.index).ffill(limit=3)          # align to trading days
    R = np.log(P).diff()                            # daily log returns

    print("\n" + "=" * 74)
    print("TEST 1 - TIME SERIES: forward return by attention-z bucket (all assets pooled)")
    print("=" * 74)
    for H in (1, 5, 20):
        fwd = np.log(P).diff(H).shift(-H)           # forward H-day log return
        z = Z.shift(1)                              # yesterday's attention -> today's entry
        zf = z.values.ravel(); rf = fwd.values.ravel()
        m = ~(np.isnan(zf) | np.isnan(rf))
        zf, rf = zf[m], rf[m]
        print("\n  H=%2dd  (n=%d)" % (H, len(zf)))
        edges = [-9, -1, -0.25, 0.25, 1, 2, 9]
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (zf >= a) & (zf < b)
            if sel.sum() < 50:
                continue
            mu = rf[sel].mean() * 100
            se = rf[sel].std() / np.sqrt(sel.sum()) * 100
            print("    z in [%5.2f,%5.2f)  n=%5d  fwd %+6.3f%%  t=%+5.2f" %
                  (a, b, sel.sum(), mu, mu / se if se else 0))

    print("\n" + "=" * 74)
    print("TEST 2 - CROSS SECTION: long top-K / short bottom-K attention, hold H days")
    print("=" * 74)
    for K in (3, 4):
        for H in (1, 5, 10, 20):
            rets = []
            z = Z.shift(1)
            days = P.index[::H]
            for d in days:
                if d not in z.index:
                    continue
                row = z.loc[d].dropna()
                if len(row) < 2 * K:
                    continue
                i = P.index.get_loc(d)
                if i + H >= len(P):
                    break
                fwd = np.log(P.iloc[i + H] / P.iloc[i])
                hi = row.nlargest(K).index
                lo = row.nsmallest(K).index
                a = fwd[hi].mean()
                b = fwd[lo].mean()
                if np.isnan(a) or np.isnan(b):
                    continue
                rets.append(a - b)
            if len(rets) < 20:
                continue
            r = np.array(rets)
            per = r.mean() * 100
            t = r.mean() / (r.std() / np.sqrt(len(r)))
            ann = per * (252 / H)
            print("  K=%d H=%2dd  n=%3d periods  per-period %+6.3f%%  ann %+7.1f%%  t=%+5.2f" %
                  (K, H, len(r), per, ann, t))

    print("\n" + "=" * 74)
    print("TEST 3 - ATTENTION AS A FILTER: trend return when attention rising vs falling")
    print("=" * 74)
    ema = P.ewm(span=50, adjust=False).mean()
    trend = np.sign(P - ema).shift(1)                # +1 above EMA50, -1 below
    zdir = Z.diff(5).shift(1)                        # is attention building?
    fwd5 = np.log(P).diff(5).shift(-5)
    base = (trend * fwd5).values.ravel()
    zz = zdir.values.ravel()
    m = ~(np.isnan(base) | np.isnan(zz))
    base, zz = base[m], zz[m]
    for lbl, sel in (("ALL", np.ones(len(zz), bool)),
                     ("attention RISING", zz > 0),
                     ("attention FALLING", zz <= 0),
                     ("attention SURGING (d5z>+1)", zz > 1)):
        if sel.sum() < 50:
            continue
        mu = base[sel].mean() * 100
        se = base[sel].std() / np.sqrt(sel.sum()) * 100
        print("  %-28s n=%5d  trend-following 5d ret %+6.3f%%  t=%+5.2f" %
              (lbl, sel.sum(), mu, mu / se if se else 0))


if __name__ == "__main__":
    main()
