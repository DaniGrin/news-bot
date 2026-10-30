"""Round 2: strip the beta out. Test 1's positive numbers may just be the 2021-26 bull market.
Everything here uses CROSS-SECTIONALLY DEMEANED forward returns (each day, subtract the
universe average) so only relative performance counts - no market drift can leak in.

Also tests the classic academic claim: attention does not predict direction on its own,
it AMPLIFIES whatever the price is already doing (attention x momentum interaction).
"""
import os, numpy as np, pandas as pd
from research_attention import UNIVERSE, wiki_views, prices

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    v, p = {}, {}
    for t, art in UNIVERSE.items():
        try:
            v[t] = wiki_views(art); p[t] = prices(t)
        except Exception as e:
            print("skip", t, e)
    return pd.DataFrame(v).sort_index(), pd.DataFrame(p).sort_index()


def block_t(x, H):
    """t-stat with a crude Newey-West style haircut for H-day overlapping windows."""
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan, np.nan
    n_eff = max(len(x) / H, 5)
    return x.mean(), x.mean() / (x.std() / np.sqrt(n_eff))


def main():
    V, P = load()
    LV = np.log(V.clip(lower=1))
    Z = ((LV - LV.rolling(60).mean()) / LV.rolling(60).std()).reindex(P.index).ffill(limit=3)
    dZ = Z.diff(5)                                    # attention SURPRISE (5d change in z)

    print("=" * 78)
    print("TEST 1b - same buckets, but DEMEANED forward returns (beta removed)")
    print("=" * 78)
    for H in (5, 20):
        fwd = np.log(P).diff(H).shift(-H)
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)       # <- cross-sectional demean
        z = Z.shift(1)
        zf, rf = z.values.ravel(), fwd.values.ravel()
        m = ~(np.isnan(zf) | np.isnan(rf))
        zf, rf = zf[m], rf[m]
        print("\n  H=%2dd  (n=%d, effective n~%d)" % (H, len(zf), len(zf) / H / len(P.columns)))
        for a, b in zip([-9, -1, -0.25, 0.25, 1, 2], [-1, -0.25, 0.25, 1, 2, 9]):
            sel = (zf >= a) & (zf < b)
            if sel.sum() < 50:
                continue
            mu, t = block_t(rf[sel], H * len(P.columns))
            print("    z in [%5.2f,%5.2f)  n=%5d  demeaned fwd %+6.3f%%  t=%+5.2f"
                  % (a, b, sel.sum(), mu * 100, t))

    print("\n" + "=" * 78)
    print("TEST 4 - ATTENTION x MOMENTUM: does attention amplify the existing move?")
    print("   rows = 20d price momentum sign, cols = attention level. Values = demeaned 20d fwd ret")
    print("=" * 78)
    H = 20
    fwd = np.log(P).diff(H).shift(-H)
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    mom = np.log(P).diff(20).shift(1)
    mom = mom.sub(mom.mean(axis=1), axis=0)           # relative momentum
    z = Z.shift(1)
    mf, zf, rf = mom.values.ravel(), z.values.ravel(), fwd.values.ravel()
    m = ~(np.isnan(mf) | np.isnan(zf) | np.isnan(rf))
    mf, zf, rf = mf[m], zf[m], rf[m]
    print("\n  %-22s %18s %18s %18s" % ("", "attn LOW (z<0)", "attn MID (0..1)", "attn HIGH (z>1)"))
    for mlbl, msel in (("mom NEGATIVE", mf < 0), ("mom POSITIVE", mf > 0)):
        cells = []
        for zsel in (zf < 0, (zf >= 0) & (zf < 1), zf >= 1):
            sel = msel & zsel
            if sel.sum() < 50:
                cells.append("        n/a       ")
                continue
            mu, t = block_t(rf[sel], H * len(P.columns))
            cells.append("%+6.2f%% (t=%+4.1f) n%4d" % (mu * 100, t, sel.sum() // 100))
        print("  %-22s %s" % (mlbl, " ".join(cells)))

    print("\n" + "=" * 78)
    print("TEST 5 - CROSS-SECTION on attention SURPRISE (dz over 5d) instead of level")
    print("=" * 78)
    for K in (3, 4, 5):
        for H in (5, 10, 20):
            s = dZ.shift(1)
            rets, dates = [], P.index[::H]
            for d in dates:
                if d not in s.index:
                    continue
                row = s.loc[d].dropna()
                if len(row) < 2 * K:
                    continue
                i = P.index.get_loc(d)
                if i + H >= len(P):
                    break
                f = np.log(P.iloc[i + H] / P.iloc[i])
                a, b = f[row.nlargest(K).index].mean(), f[row.nsmallest(K).index].mean()
                if not (np.isnan(a) or np.isnan(b)):
                    rets.append(a - b)
            if len(rets) < 20:
                continue
            r = np.array(rets)
            t = r.mean() / (r.std() / np.sqrt(len(r)))
            print("  K=%d H=%2dd  n=%3d  per-period %+6.3f%%  ann %+7.1f%%  t=%+5.2f"
                  % (K, H, len(r), r.mean() * 100, r.mean() * 100 * 252 / H, t))

    print("\n" + "=" * 78)
    print("TEST 6 - pure momentum cross-section (NO news) as the benchmark to beat")
    print("=" * 78)
    for K in (3, 4, 5):
        for H in (5, 10, 20):
            s = np.log(P).diff(60).shift(1)
            rets, dates = [], P.index[::H]
            for d in dates:
                row = s.loc[d].dropna()
                if len(row) < 2 * K:
                    continue
                i = P.index.get_loc(d)
                if i + H >= len(P):
                    break
                f = np.log(P.iloc[i + H] / P.iloc[i])
                a, b = f[row.nlargest(K).index].mean(), f[row.nsmallest(K).index].mean()
                if not (np.isnan(a) or np.isnan(b)):
                    rets.append(a - b)
            r = np.array(rets)
            if len(r) < 20:
                continue
            t = r.mean() / (r.std() / np.sqrt(len(r)))
            print("  K=%d H=%2dd  n=%3d  per-period %+6.3f%%  ann %+7.1f%%  t=%+5.2f"
                  % (K, H, len(r), r.mean() * 100, r.mean() * 100 * 252 / H, t))


if __name__ == "__main__":
    main()
