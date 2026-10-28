"""
OIBC 2026  W 테스터 + 파이프라인 (v4)

■ 목적함수 (온라인 설명회 슬라이드 기준)
    W = Σ_t SMP_t · k_t · |Ŷ(g,t) − A(g,t)|    (G1, G2 각각 '집합 합계'로 계산 후 합산, 집합 간 상쇄 없음)
    k = 3 (ts 라벨 시각 11~15시 = 실제 10~15시), 그 외 1
    정규화 점수 = 100 × (1 − W팀 / W기준선)

■ 테스터: tester(fit_years, eval_years)
    fit_years 데이터로 모델 학습 + 집합자원 구성 최적화 → eval_years 에서 W 를 채점
    (연도만 바꿔서 반복 실행. 예: tester(2022, 2023), tester([2021, 2022], 2024))
    출력 W 3종:  W_base  기준선형(L2·무가중·참조구성)
                 W_ref   팀 모델 + 참조구성
                 W_opt   팀 모델 + 최적화 구성    → 점수 = 100×(1−W/W_base)

■ 제출: make_submission()  →  out/composition.csv, out/forecast.csv  (+ 자동 형식 점검)
        check_submission()  →  무효 조건 6가지 직접 점검 (검증 도구가 없으므로 필수)

■ Colab / Jupyter 사용 예 (이 파일을 셀에 붙여 실행한 뒤)
    CFG["data_dir"] = "/content/data"
    inspect()                          # 파일/컬럼/연도별 행 수 확인 (먼저!)
    selftest()                         # W 계산 검증 (슬라이드 예시)
    r = tester(2022, 2023)             # 연도 2개 비교
    year_grid([(2021, 2022), (2022, 2023), (2023, 2024)])
    make_submission(); check_submission()

■ 데이터 폴더 구조 (슬라이드 기준)
    data/sites.csv, data/smp.csv
    data/train/{generation, forecast_wa, forecast_wb, observed_wa, observed_wb}.parquet, data/train/smp.csv
    data/test/{forecast_wa, forecast_wb}.parquet
    (폴더 없이 한 곳에 몰아 올려도 train 파일은 자동 탐색)
"""
import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from importlib import metadata

# ----------------------------------------------------------------------
# [자동 설치] 필요한 패키지가 없거나 버전이 다르면 실행 시 자동으로 pip install
#   (모듈 이름, pip 패키지 이름, 고정 버전 또는 None)
# ----------------------------------------------------------------------
REQUIRED = [
    ("numpy", "numpy", None),
    ("pandas", "pandas", None),
    ("pyarrow", "pyarrow", None),
    ("lightgbm", "lightgbm", "4.6.0"),
    ("sklearn", "scikit-learn", None),      # backend="sklearn" 대체 경로용
]


def _ensure_packages():
    to_install = []
    for mod, pkg, ver in REQUIRED:
        if importlib.util.find_spec(mod) is None:
            to_install.append(f"{pkg}=={ver}" if ver else pkg)
        elif ver:
            try:
                if metadata.version(pkg) != ver:
                    to_install.append(f"{pkg}=={ver}")
            except metadata.PackageNotFoundError:
                pass
    if not to_install:
        return
    print(f"[auto-install] 설치 중: {' '.join(to_install)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    except subprocess.CalledProcessError as e:
        sys.exit(f"[auto-install] 설치 실패 ({e}). 네트워크/권한을 확인하거나 수동으로 설치하세요:\n"
                 f"  {sys.executable} -m pip install {' '.join(to_install)}")
    importlib.invalidate_caches()
    print("[auto-install] 완료")


_ensure_packages()
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# [CONFIG]
# ----------------------------------------------------------------------
CFG = dict(
    data_dir="data",
    out_dir="out",
    files=dict(
        sites="sites.csv",
        smp_eval="smp.csv",                      # 채점 구간(2025-06~2026-05) 시각 목록 + SMP + k
        smp_train="train/smp.csv",               # 학습 구간 SMP
        gen="train/generation.parquet",
        fc_a="train/forecast_wa.parquet",
        fc_b="train/forecast_wb.parquet",
        obs_a="train/observed_wa.parquet",       # inspect 전용 (피처 미사용)
        obs_b="train/observed_wb.parquet",
        test_fc_a="test/forecast_wa.parquet",    # 평가 구간 예보 (제출용)
        test_fc_b="test/forecast_wb.parquet",
    ),
    # 표준 컬럼명 ← 실제 파일의 후보 이름들 (대소문자 무시, 앞에서부터 일치하는 것 사용)
    aliases=dict(
        site=["site_id", "plant_id", "site", "발전소"],
        ts=["ts", "datetime", "timestamp", "time", "date_time", "target_time", "valid_time", "dt"],
        gen=["generation", "generation_kwh", "gen_kwh", "gen", "kwh", "power", "energy", "발전량"],
        cap=["capacity_kw", "capacity", "cap_kw", "cap", "설비용량"],
        lat=["lat", "latitude"], lon=["lon", "lng", "longitude"],
        cur=["current_group", "ref_group"],
        wa=["wa_station"], wb=["wb_station"],
        smp=["smp", "smp_price", "price"],
        k=["k", "k_weight", "weight_k", "weight"],
        rnd=["round", "fc_round", "run", "cycle"],
        issued=["issued_at", "issue_time", "base_time", "init_time"],
        st=["station", "station_id", "stn", "stn_id", "point", "point_id", "obs_station", "지점"],
    ),
    min_cap_kw=20_000.0,        # 집합별 최소 설비용량 (슬라이드: 미달이면 제출 무효)
    start_month=1,              # 연도 구분 시작 월: 1=달력연도, 6=6월~다음해 5월 (평가 구간과 동일 구조)
    k_peak_hours=(11, 12, 13, 14, 15), k_peak=3.0,   # ts 라벨 기준
    # 모델
    models=dict(
        base=dict(objective="regression", weight=False),       # 기준선형
        team=dict(objective="regression_l1", weight=True),     # 팀 모델: L1 + (SMP·k) 가중
    ),
    n_estimators=400,
    backend="lightgbm",         # lightgbm 이 오류나면 "sklearn" (느리지만 안전)
    cv_folds=3,                 # 주(week) 블록 교차검증 → OOF 오차로 구성 최적화
    opt_iters=200_000,
    seeds=(0,),                 # 최종 예측 시드 (제출 시 (0,1,2) 권장)
    submit_train_from="2023-06-01",   # 제출용 학습 시작일 (기준선과 동일: 최근 2년). None=전체(메모리 주의)
)
C = dict(site="site_id", ts="ts", gen="generation", cap="capacity_kw", lat="lat", lon="lon",
         cur="current_group", wa="wa_station", wb="wb_station", smp="smp", k="k",
         rnd="round", issued="issued_at", st="station")
HINTS = dict(gen=("gen", "kwh"), smp=("smp",), cap=("cap",))


# ----------------------------------------------------------------------
# 입출력 / 정규화
# ----------------------------------------------------------------------
def _find(name, flat_ok=True):
    cands = [os.path.join(CFG["data_dir"], name)]
    if flat_ok:
        cands.append(os.path.join(CFG["data_dir"], os.path.basename(name)))
    for q in cands:
        root, _ = os.path.splitext(q)
        for alt in (q, root + ".parquet", root + ".csv"):
            if os.path.exists(alt):
                return alt
    return None


def read_table(name, flat_ok=True, required=True):
    p = _find(name, flat_ok)
    if p is None:
        if required:
            raise FileNotFoundError(f"'{name}' 을(를) '{CFG['data_dir']}' 아래에서 찾지 못함 → CFG['data_dir'] 확인")
        return None
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def _to_kst(s):
    """시각 → KST naive datetime (우측닫힘 라벨 그대로 유지)"""
    try:
        t = pd.to_datetime(s)
    except (ValueError, TypeError):
        t = pd.to_datetime(s, utc=True)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return t.astype("datetime64[ns]")      # 파일마다 ms/us 단위가 달라 ns 로 통일 (매칭 오류 방지)


def _key(s):
    """관측지점 키를 문자열로 통일 (정수/문자 혼재 방지)"""
    try:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype("Int64").astype(str)
    except (TypeError, ValueError):
        pass
    return s.astype(str).str.strip()


def normalize(df, hints=True):
    df = df.copy()
    if C["ts"] not in [c for c in df.columns] and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": C["ts"]})
    low = {c.lower(): c for c in df.columns}
    for std, cands in CFG["aliases"].items():
        tgt = C[std]
        if tgt in df.columns:
            continue
        hit = next((low[c.lower()] for c in cands if c.lower() in low), None)
        if hit is None and hints and std in HINTS:
            m = [c for c in df.columns if any(h in c.lower() for h in HINTS[std]) and c not in C.values()]
            hit = m[0] if len(m) == 1 else None
        if hit is not None:
            df = df.rename(columns={hit: tgt})
    if C["ts"] in df.columns:
        df[C["ts"]] = _to_kst(df[C["ts"]])
    return df


def inspect():
    """모든 파일의 스키마·인식 결과·연도별 행 수를 출력"""
    for key, name in CFG["files"].items():
        flat = not key.startswith("test") and key != "smp_eval"
        p = _find(name, flat_ok=flat if key not in ("smp_train",) else False)
        if p is None:
            print(f"[{key}] {name} 없음\n")
            continue
        raw = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        df = normalize(raw, hints=key in ("gen", "sites", "smp_eval", "smp_train"))
        print(f"[{key}] {p}  ({len(raw):,}행)")
        print(df.dtypes.to_string())
        print("  인식된 표준 컬럼:", {k: v for k, v in C.items() if v in df.columns})
        if C["ts"] in df.columns:
            print("  연도별 행 수:", df[C["ts"]].dt.year.value_counts().sort_index().to_dict())
        print(df.head(2).to_string(), "\n")
    g = read_table(CFG["files"]["gen"], required=False)
    if g is not None:
        g = normalize(g)
        if C["site"] in g.columns and C["ts"] in g.columns:
            print("[generation] 연도별 발전소 수:",
                  g.groupby(g[C["ts"]].dt.year)[C["site"]].nunique().to_dict())


# ----------------------------------------------------------------------
# 데이터 로딩 (1회 캐시)
# ----------------------------------------------------------------------
_D = {}


def load_smp():
    parts = []
    for key, flat in (("smp_train", False), ("smp_eval", False)):
        d = read_table(CFG["files"][key], flat_ok=flat, required=False)
        if d is not None:
            parts.append(normalize(d))
    if not parts:
        raise FileNotFoundError("smp.csv / train/smp.csv 를 찾지 못함")
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(C["ts"], keep="last").sort_values(C["ts"]).set_index(C["ts"])
    rule = np.where(d.index.hour.isin(CFG["k_peak_hours"]), CFG["k_peak"], 1.0)   # ts 라벨 기준
    if C["k"] in d.columns:
        k = d[C["k"]].astype(float)
        bad = (k.notna() & (k != rule)).mean()
        if bad > 0:
            print(f"[경고] smp 파일의 k 가 규칙(11~15시=3)과 {bad:.1%} 불일치 → 파일 값을 우선 사용")
        d["k"] = k.fillna(pd.Series(rule, index=d.index))
    else:
        d["k"] = rule
    d["smp"] = d[C["smp"]].astype(float)
    full = pd.date_range(d.index.min(), d.index.max(), freq="h")
    d = d[["smp", "k"]].reindex(full)
    d["smp"] = d["smp"].ffill().bfill()
    d["k"] = d["k"].fillna(pd.Series(np.where(full.hour.isin(CFG["k_peak_hours"]), CFG["k_peak"], 1.0), index=full))
    d["w"] = d["smp"] * d["k"]
    return d


def load_fc(name, flat_ok=True):
    """기상 예보 1개 제공사 → (long 테이블[st, ts, _r, vars...], 변수 목록)"""
    fc = normalize(read_table(name, flat_ok), hints=False)
    if C["st"] not in fc.columns:
        cand = [c for c in fc.columns if c not in (C["ts"], C["rnd"], C["issued"])
                and not pd.api.types.is_numeric_dtype(fc[c]) and not pd.api.types.is_datetime64_any_dtype(fc[c])]
        if len(cand) != 1:
            raise KeyError(f"{name}: 관측지점 컬럼을 특정할 수 없음 → CFG['aliases']['st'] 에 추가 ({list(fc.columns)})")
        fc = fc.rename(columns={cand[0]: C["st"]})
    fc[C["st"]] = _key(fc[C["st"]])
    if C["rnd"] in fc.columns:
        fc["_r"] = _key(fc[C["rnd"]])
    elif C["issued"] in fc.columns:
        fc["_r"] = pd.to_datetime(fc[C["issued"]]).dt.hour.astype(str)
    else:
        fc["_r"] = "0"
    skip = {C["st"], C["ts"], "_r", C["rnd"], C["issued"], C["site"], C["lat"], C["lon"], C["cap"], C["gen"]}
    vars_ = [c for c in fc.columns if c not in skip and pd.api.types.is_numeric_dtype(fc[c])
             and not pd.api.types.is_bool_dtype(fc[c])]
    fc[vars_] = fc[vars_].astype("float32")
    return fc[[C["st"], C["ts"], "_r"] + vars_], vars_


def load_data(force=False):
    if _D and not force:
        return _D
    sites = normalize(read_table(CFG["files"]["sites"]))
    for need in ("site", "cap", "lat", "lon", "wa", "wb"):
        if C[need] not in sites.columns:
            raise KeyError(f"sites 에 '{C[need]}' 없음: {list(sites.columns)}")
    sites = sites.sort_values(C["site"]).reset_index(drop=True)
    for c in (C["wa"], C["wb"]):
        sites[c] = _key(sites[c])
    gen = normalize(read_table(CFG["files"]["gen"]))
    for need in ("site", "ts", "gen"):
        if C[need] not in gen.columns:
            raise KeyError(f"generation 에 '{C[need]}' 없음 → CFG['aliases'] 수정: {list(gen.columns)}")
    gen = gen[[C["site"], C["ts"], C["gen"]]].dropna()
    gen = gen[gen[C["site"]].isin(sites[C["site"]])].reset_index(drop=True)

    ref = None
    if C["cur"] in sites.columns:
        ref = sites[C["cur"]].astype(str).str.upper().map({"A": 0, "B": 1, "G1": 0, "G2": 1}).values
        if np.isnan(ref.astype(float)).any():
            ref = None
        else:
            ref = ref.astype(np.int8)
    _D.update(sites=sites, gen=gen, wtab=load_smp(), ref=ref,
              fc={"A": load_fc(CFG["files"]["fc_a"]), "B": load_fc(CFG["files"]["fc_b"])})
    print(f"로딩 완료: 발전소 {len(sites)} | 발전 행 {len(gen):,} | 기간 {gen[C['ts']].min()} ~ {gen[C['ts']].max()}")
    return _D


def season_of(ts, start_month):
    """우측닫힘 라벨 기준 연도 구분: 구간 시작시각(ts-1h)이 속한 연도(start_month 기준)"""
    s = ts - pd.Timedelta(hours=1)
    return s.dt.year - (s.dt.month < start_month).astype(int)


# ----------------------------------------------------------------------
# 피처
# ----------------------------------------------------------------------
def solar_features(ts, lat, lon):
    mid = ts - pd.Timedelta(minutes=30)            # 우측닫힘: 구간 중앙, KST=UTC+9
    doy = mid.dt.dayofyear.values
    hr = (mid.dt.hour + mid.dt.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hr * 60 + eq + 4 * np.asarray(lon) - 60 * 9
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(np.asarray(lat))
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    i0 = 1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    return pd.DataFrame({
        "cosz": cosz.astype("float32"), "etr": (i0 * np.clip(cosz, 0, None)).astype("float32"),
        "ha": ha.astype("float32"), "doy_sin": np.sin(2 * np.pi * doy / 365).astype("float32"),
        "doy_cos": np.cos(2 * np.pi * doy / 365).astype("float32"),
        "hour": mid.dt.hour.values.astype("int8"),
    }, index=ts.index)


def fc_features(fc, vars_, prov, hours, stations):
    """제공사 1개: 라운드 평균(avg) + 라운드 차이(d, 예보 변동성). 제공사끼리는 섞지 않음(단위계 상이)"""
    d = fc[fc[C["st"]].isin(stations) & fc[C["ts"]].isin(hours)]
    g = d.groupby([C["st"], C["ts"], "_r"])[vars_].mean()
    found = set(g.index.get_level_values(0))
    miss = [s for s in stations if s not in found]
    if miss:
        print(f"  [경고] 예보 {prov}: 지점 {len(miss)}개 없음 → 중앙값 대체 (예: {miss[:3]})")
    rounds = sorted(g.index.get_level_values("_r").unique())
    idx = pd.MultiIndex.from_product([stations, hours], names=[C["st"], C["ts"]])
    a = g.xs(rounds[0], level="_r").reindex(idx)
    b = g.xs(rounds[-1], level="_r").reindex(idx) if len(rounds) > 1 else a
    a, b = a.fillna(b), b.fillna(a)                 # 결손: 다른 라운드로 대체 (행 제외 금지)
    out = pd.concat([((a + b) / 2).add_suffix(f"__{prov}_avg"), (b - a).add_suffix(f"__{prov}_d")], axis=1)
    out = out.groupby(level=0).transform(lambda s: s.interpolate(limit=6, limit_direction="both"))
    return out.fillna(out.median()).astype("float32")


def make_frame(base, fcs):
    """base: [site, ts(, generation)] → 낮 시간(cosz>0) 행에 피처 부착. '_row' = base 내 위치"""
    D = load_data()
    sites = D["sites"]
    b = base.reset_index(drop=True).merge(
        sites[[C["site"], C["lat"], C["lon"], C["cap"], C["wa"], C["wb"]]], on=C["site"], how="left")
    b["cap_"] = b[C["cap"]].astype("float32")
    b = pd.concat([b, solar_features(b[C["ts"]], b[C["lat"]], b[C["lon"]])], axis=1)
    b["_row"] = np.arange(len(b))
    day = b[b["cosz"] > 0.0].copy()
    hours = pd.DatetimeIndex(np.sort(day[C["ts"]].unique()))
    feats = []
    for prov, key in (("A", C["wa"]), ("B", C["wb"])):
        fc, vars_ = fcs[prov]
        w = fc_features(fc, vars_, prov, hours, sorted(day[key].unique()))
        w.index.names = [key, C["ts"]]
        day = day.merge(w.reset_index(), on=[key, C["ts"]], how="left")
        feats += list(w.columns)
    day[feats] = day[feats].fillna(day[feats].median())
    wt = D["wtab"]["w"]
    day["_w"] = day[C["ts"]].map(wt).fillna(wt.median()).astype("float32")
    base_feats = ["cosz", "etr", "ha", "doy_sin", "doy_cos", "hour", C["lat"], C["lon"], "cap_"]
    return day.reset_index(drop=True), base_feats + feats


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
def make_model(seed, objective):
    try:
        if CFG["backend"] != "lightgbm":
            raise ImportError
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective=objective, n_estimators=CFG["n_estimators"], learning_rate=0.06,
                                 num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                 min_child_samples=100, random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        loss = "absolute_error" if objective in ("regression_l1", "l1") else "squared_error"
        return HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.08,
                                             max_leaf_nodes=63, random_state=seed, early_stopping=False)


def with_site_cf(df, tr):
    """발전소별 평균 이용률 (학습 데이터 기준 → 누수 방지)"""
    m = tr.groupby(C["site"])["cf"].mean()
    out = df.copy()
    out["site_cf"] = out[C["site"]].map(m).fillna(tr["cf"].mean()).astype("float32")
    return out


def fit_predict(tr, te, feats, spec, seed=0):
    """학습 → te 예측 (kWh)"""
    tr = with_site_cf(tr, tr)
    te = with_site_cf(te, tr)
    f = feats + ["site_cf"]
    # pandas 최신판 ↔ LightGBM 호환 문제(access violation) 방지: 순수 numpy 연속 배열로 변환
    X = np.ascontiguousarray(tr[f].to_numpy(dtype="float32", na_value=np.nan))
    y = np.ascontiguousarray(tr["cf"].to_numpy(dtype="float32", na_value=np.nan))
    sw = np.ascontiguousarray((tr["_w"] / tr["_w"].mean()).to_numpy(dtype="float32")) if spec["weight"] else None
    Xt = np.ascontiguousarray(te[f].to_numpy(dtype="float32", na_value=np.nan))
    m = make_model(seed, spec["objective"])
    try:
        m.fit(X, y, sample_weight=sw)
    except OSError as e:                      # 일부 Windows 환경의 LightGBM 크래시 → sklearn 으로 자동 전환
        print(f"  [경고] LightGBM 오류({e}) → scikit-learn 모델로 자동 전환 (더 느림)")
        CFG["backend"] = "sklearn"
        m = make_model(seed, spec["objective"])
        m.fit(X, y, sample_weight=sw)
    return np.clip(m.predict(Xt), 0, 1.3).astype("float32") * te["cap_"].to_numpy(dtype="float32")


def prep_train(day):
    day = day.copy()
    day["cf"] = (day[C["gen"]] / day["cap_"]).clip(0, 1.3).astype("float32")
    return day[day["cosz"] > 0.02].reset_index(drop=True)


def oof_errors(tr, feats, spec, K, seed=0):
    """주(week) 블록 CV 의 OOF 오차 = 예측 − 실적 (kWh)"""
    assert K >= 2
    week = ((tr[C["ts"]] - tr[C["ts"]].min()).dt.days // 7) % K
    oof = np.zeros(len(tr), dtype="float32")
    for k in range(K):
        m = (week == k).values
        oof[m] = fit_predict(tr[~m], tr[m], feats, spec, seed)
        mae = np.abs(oof[m] - tr.loc[m, C["gen"]].values).mean()
        print(f"    OOF fold {k}: MAE {mae:,.2f} kWh/행")
    return oof - tr[C["gen"]].values


# ----------------------------------------------------------------------
# W 계산 (핵심)
# ----------------------------------------------------------------------
def compute_W(base, pred, assign, wtab):
    """
    base  : DataFrame [site_id, ts, generation(=A, kWh)]   pred: base 와 같은 길이의 예측(kWh)
    assign: Series  site_id → 0(G1)/1(G2)                  wtab: index=ts, 컬럼 smp, k
    W = Σ SMP·k·|Σ_{i∈g} Ŷ_i − Σ_{i∈g} A_i|   (집합·시각별 합계 후 절댓값)
    """
    d = pd.DataFrame({"g": base[C["site"]].map(assign).values, C["ts"]: base[C["ts"]].values,
                      "A": base[C["gen"]].values.astype("float64"), "P": np.asarray(pred, dtype="float64")})
    t = d.groupby(["g", C["ts"]], sort=False)[["A", "P"]].sum().reset_index()
    t = t.merge(wtab[["smp", "k"]], left_on=C["ts"], right_index=True, how="left")
    t["err"] = (t["P"] - t["A"]).abs()
    t["pen"] = t["smp"] * t["k"] * t["err"]
    W = float(t["pen"].sum())
    out = dict(W=W,
               W_G1=float(t.loc[t["g"] == 0, "pen"].sum()), W_G2=float(t.loc[t["g"] == 1, "pen"].sum()),
               peak_share=float(t.loc[t["k"] > 1, "pen"].sum() / W) if W > 0 else np.nan,
               abs_err=float(t["err"].sum()), penalty=0.10 * W, n_nan_smp=int(t["smp"].isna().sum()))
    return out, t


def selftest():
    """슬라이드 예시로 W 계산 검증: k=3, SMP 120, 실적 12,000 / 예측 11,500 → 페널티 18,000원"""
    w = pd.DataFrame({"smp": [120.0, 120.0], "k": [3.0, 1.0], "w": [360.0, 120.0]},
                     index=pd.to_datetime(["2025-07-01 13:00", "2025-07-01 08:00"]))
    mk = lambda sites, ts, a, p: (pd.DataFrame({C["site"]: sites, C["ts"]: pd.to_datetime(ts), C["gen"]: a}), p)
    # (1) 단일 시각 예시
    b, p = mk(["s1"], ["2025-07-01 13:00"], [12000.0], [11500.0])
    r, _ = compute_W(b, p, pd.Series({"s1": 0}), w)
    assert abs(r["W"] - 180_000) < 1e-6 and abs(r["penalty"] - 18_000) < 1e-6, r
    b, p = mk(["s1"], ["2025-07-01 08:00"], [12000.0], [11500.0])
    r, _ = compute_W(b, p, pd.Series({"s1": 0}), w)
    assert abs(r["penalty"] - 6_000) < 1e-6, r
    # (2) 같은 집합 → 반대 오차 상쇄 / 다른 집합 → 상쇄 없음
    b, p = mk(["s1", "s2"], ["2025-07-01 13:00"] * 2, [100.0, 100.0], [110.0, 90.0])
    same, _ = compute_W(b, p, pd.Series({"s1": 0, "s2": 0}), w)
    diff, _ = compute_W(b, p, pd.Series({"s1": 0, "s2": 1}), w)
    assert same["W"] == 0 and abs(diff["W"] - 360 * 20) < 1e-6, (same, diff)
    print("selftest OK  (예시 13시 k=3: 페널티 18,000원 / 08시 k=1: 6,000원 / 집합 간 상쇄 없음 확인)")


# ----------------------------------------------------------------------
# 집합자원 최적화 (SA: swap / move, 설비용량 20,000 kW 제약 유지)
# ----------------------------------------------------------------------
def optimize_groups(Et, caps, w, init, iters, seed, min_cap):
    """Et: (N, T) 오차(예측-실적). cost = Σ_t w_t Σ_g |Σ_{i∈g} Et|"""
    rng = np.random.default_rng(seed)
    N = Et.shape[0]
    keep = np.abs(Et).sum(0) > 0                       # 오차 없는 시각(야간 등) 제외 → 속도
    Et, w = np.ascontiguousarray(Et[:, keep]), w[keep]
    assign = init.copy()
    S = [Et[assign == g].sum(0) for g in (0, 1)]
    cg = [float(caps[assign == g].sum()) for g in (0, 1)]
    cost = float(w @ np.abs(S[0]) + w @ np.abs(S[1]))
    best, best_assign = cost, assign.copy()
    t0 = cost * 0.002 + 1e-9
    for it in range(iters):
        temp = max(t0 * (1 - it / iters), 1e-12)
        if rng.random() < 0.7:                                       # swap
            i = rng.choice(np.flatnonzero(assign == 0))
            j = rng.choice(np.flatnonzero(assign == 1))
            c0 = cg[0] - caps[i] + caps[j]
            c1 = cg[1] + caps[i] - caps[j]
            if c0 < min_cap or c1 < min_cap:
                continue
            S0 = S[0] - Et[i] + Et[j]
            S1 = S[1] + Et[i] - Et[j]
            new = float(w @ np.abs(S0) + w @ np.abs(S1))
            if new < cost or rng.random() < np.exp(-(new - cost) / temp):
                assign[i], assign[j] = 1, 0
                S, cg, cost = [S0, S1], [c0, c1], new
        else:                                                        # move
            i = int(rng.integers(N))
            a, b = int(assign[i]), 1 - int(assign[i])
            ca, cb = cg[a] - caps[i], cg[b] + caps[i]
            if ca < min_cap:
                continue
            Sa, Sb = S[a] - Et[i], S[b] + Et[i]
            new = float(w @ np.abs(Sa) + w @ np.abs(Sb))
            if new < cost or rng.random() < np.exp(-(new - cost) / temp):
                assign[i] = b
                S[a], S[b], cg[a], cg[b], cost = Sa, Sb, ca, cb, new
        if cost < best:
            best, best_assign = cost, assign.copy()
    return best_assign, best


def err_matrix(tr, err, sites, wtab):
    hours = pd.DatetimeIndex(tr[C["ts"]].unique()).sort_values()
    ti = hours.get_indexer(tr[C["ts"]])
    si = pd.Series(np.arange(len(sites)), index=sites[C["site"]].values)[tr[C["site"]]].values
    E = np.zeros((len(sites), len(hours)), dtype="float32")
    E[si, ti] = err
    w = wtab["w"].reindex(hours).fillna(wtab["w"].median()).values.astype("float64")
    return E, w


def group_caps(assign, sites):
    c = sites[C["cap"]].values.astype("float64")
    return float(c[assign == 0].sum()), float(c[assign == 1].sum())


def find_composition(Et, sites, w, ref, seeds, iters):
    caps = sites[C["cap"]].values.astype("float64")
    init = ref if ref is not None else (np.arange(len(sites)) % 2).astype(np.int8)
    if min(group_caps(init, sites)) < CFG["min_cap_kw"]:
        print(f"  [경고] 초기 구성이 {CFG['min_cap_kw']:,.0f} kW 제약 미달 → 용량 기준 재분배")
        order = np.argsort(-caps)
        init = np.zeros(len(caps), dtype=np.int8)
        tot = [0.0, 0.0]
        for i in order:                                  # 큰 것부터 작은 쪽에 배정 (그리디)
            g = int(tot[1] < tot[0])
            init[i] = g
            tot[g] += caps[i]
    best = (None, np.inf)
    for s in seeds:
        a, c = optimize_groups(Et, caps, w, init, iters, s, CFG["min_cap_kw"])
        print(f"    seed {s}: OOF cost {c:,.0f}")
        if c < best[1]:
            best = (a, c)
    return best[0], best[1], init


# ----------------------------------------------------------------------
# 테스터
# ----------------------------------------------------------------------
def _as_list(x):
    return [int(v) for v in (x if isinstance(x, (list, tuple, set)) else [x])]


def tester(fit_years, eval_years, start_month=None, seeds=None, verbose=True):
    """
    fit_years 로 학습·구성 최적화 → eval_years 에서 W 채점.
      W_base : 기준선형 (L2·무가중, 참조 구성)       ← 점수 0 기준
      W_ref  : 팀 모델 (L1·SMP·k 가중), 참조 구성     → 예측 개선 효과
      W_opt  : 팀 모델 + 최적화 구성                  → 예측 + 구성 개선 효과
    """
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다 (서로 다른 연도를 지정하세요)")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    for nm, b, ys in (("fit", base_fit, fit_y), ("eval", base_ev, ev_y)):
        if len(b) == 0:
            raise ValueError(f"{nm} 연도 {ys} 에 발전 데이터 없음. 보유 연도: {sorted(sea.unique())} (start_month={sm})")
    n_ev_h = base_ev[C["ts"]].nunique()
    print(f"■ fit={fit_y} → eval={ev_y} (start_month={sm}) | 학습 {len(base_fit):,}행, 평가 {len(base_ev):,}행 / {n_ev_h:,}시각")
    n_fs, n_es = base_fit[C["site"]].nunique(), base_ev[C["site"]].nunique()
    print(f"  데이터가 있는 발전소: fit {n_fs} / eval {n_es} (전체 {len(sites)})")
    if n_fs < len(sites) or n_es < len(sites):
        print("  (참고) 발전소가 연도별로 늘어난 데이터입니다. 실제 평가는 253개소 전체이므로 eval 은 2024년 이후(253개소)를 권장")
    if n_ev_h < 8000:
        print(f"  (참고) 평가 시각이 {n_ev_h:,}개 — 부분 연도라 W 절댓값은 8,760시간 기준보다 작습니다. 점수(%)로 비교하세요")

    # --- 학습·평가 프레임
    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"])

    # --- OOF 오차 → 구성 최적화 (fit 연도 안에서만)
    print("  OOF 오차 계산 (팀 모델)...")
    err = oof_errors(tr, feats, CFG["models"]["team"], CFG["cv_folds"])
    Et, w = err_matrix(tr, err, sites, wtab)
    print("  구성 최적화...")
    opt, opt_cost, init = find_composition(Et, sites, w, ref, seeds, CFG["opt_iters"])
    ref_cost = float(w @ np.abs(Et[ref == 0].sum(0)) + w @ np.abs(Et[ref == 1].sum(0))) if ref is not None else np.nan

    # --- 최종 학습 → 평가 연도 예측 (낮 시간만 예측, 밤은 0)
    print("  최종 학습/예측 (base, team)...")
    preds = {}
    for name in ("base", "team"):
        p = np.mean([fit_predict(tr, ev, feats, CFG["models"][name], seed=s) for s in seeds], axis=0)
        full = np.zeros(len(base_ev))
        full[ev["_row"].values] = p
        preds[name] = full

    sid = sites[C["site"]].values
    a_ref = pd.Series(ref if ref is not None else init, index=sid)
    a_opt = pd.Series(opt, index=sid)
    r_base, _ = compute_W(base_ev, preds["base"], a_ref, wtab)
    r_ref, _ = compute_W(base_ev, preds["team"], a_ref, wtab)
    r_opt, detail = compute_W(base_ev, preds["team"], a_opt, wtab)
    sc = lambda r: 100 * (1 - r["W"] / r_base["W"])
    res = dict(fit=fit_y, eval=ev_y, W_base=r_base["W"], W_ref=r_ref["W"], W_opt=r_opt["W"],
               score_model=sc(r_ref), score_total=sc(r_opt), score_comp=100 * (1 - r_opt["W"] / r_ref["W"]),
               peak_share=r_opt["peak_share"], caps=group_caps(opt, sites), n_move=int((opt != a_ref.values).sum()),
               oof_gain=(1 - opt_cost / ref_cost) * 100 if ref_cost == ref_cost else np.nan,
               assign=a_opt, detail=detail)
    if verbose:
        print(f"\n  W_base (기준선형·참조구성) : {r_base['W']:>16,.0f}")
        print(f"  W_ref  (팀 모델·참조구성)  : {r_ref['W']:>16,.0f}   점수 {res['score_model']:+.2f}")
        print(f"  W_opt  (팀 모델·최적구성)  : {r_opt['W']:>16,.0f}   점수 {res['score_total']:+.2f}")
        print(f"  └ 구성 최적화만의 효과(W_ref→W_opt): {res['score_comp']:+.2f}%  | fit 연도 OOF 개선 {res['oof_gain']:+.2f}%  "
              f"(둘의 차이가 크면 과적합 의심)")
        print(f"  피크(k=3) 페널티 비중 {res['peak_share']:.1%} | 집합 용량 G1 {res['caps'][0]:,.1f} / G2 {res['caps'][1]:,.1f} kW"
              f" | 참조 대비 이동 발전소 {res['n_move']}개")
    return res


def year_grid(pairs, **kw):
    """여러 (fit, eval) 쌍을 연속 실행해 요약표 반환. 예: year_grid([(2021,2022),(2022,2023)])"""
    rows = []
    for f, e in pairs:
        r = tester(f, e, verbose=False, **kw)
        rows.append(dict(fit=r["fit"], eval=r["eval"], W_base=r["W_base"], W_ref=r["W_ref"], W_opt=r["W_opt"],
                         score_model=r["score_model"], score_total=r["score_total"], score_comp=r["score_comp"],
                         oof_gain=r["oof_gain"], peak_share=r["peak_share"]))
    df = pd.DataFrame(rows)
    print("\n" + df.round(2).to_string(index=False))
    return df


# ----------------------------------------------------------------------
# 제출 생성 + 형식 점검
# ----------------------------------------------------------------------
def make_submission(seeds=(0, 1, 2)):
    os.makedirs(CFG["out_dir"], exist_ok=True)
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    g = D["gen"]
    if CFG["submit_train_from"]:
        g = g[g[C["ts"]] >= pd.Timestamp(CFG["submit_train_from"])]
    print(f"학습 행 {len(g):,} ({g[C['ts']].min()} ~ {g[C['ts']].max()})")
    tr_all, feats = make_frame(g.reset_index(drop=True), D["fc"])
    tr = prep_train(tr_all)
    err = oof_errors(tr, feats, CFG["models"]["team"], CFG["cv_folds"])
    Et, w = err_matrix(tr, err, sites, wtab)
    opt, cost, init = find_composition(Et, sites, w, ref, seeds, CFG["opt_iters"])

    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    eval_hours = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    fcT = {"A": load_fc(CFG["files"]["test_fc_a"], flat_ok=False), "B": load_fc(CFG["files"]["test_fc_b"], flat_ok=False)}
    grid = pd.MultiIndex.from_product([sites[C["site"]].values, eval_hours],
                                      names=[C["site"], C["ts"]]).to_frame(index=False)
    ev, _ = make_frame(grid, fcT)
    p = np.mean([fit_predict(tr, ev, feats, CFG["models"]["team"], seed=s) for s in seeds], axis=0)
    full = np.zeros(len(grid))
    full[ev["_row"].values] = p
    grid["P"] = full
    grid["group"] = grid[C["site"]].map(pd.Series(opt, index=sites[C["site"]].values))
    agg = grid.groupby(["group", C["ts"]])["P"].sum().reset_index()
    agg["group"] = agg["group"].map({0: "G1", 1: "G2"})
    agg[C["ts"]] = agg[C["ts"]].dt.tz_localize("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    agg = agg.rename(columns={C["ts"]: "ts", "P": "forecast_kwh"})[["group", "ts", "forecast_kwh"]]
    agg["forecast_kwh"] = agg["forecast_kwh"].clip(lower=0.0)
    comp = pd.DataFrame({"site_id": sites[C["site"]].values, "group": np.where(opt == 0, "G1", "G2")})
    comp.to_csv(os.path.join(CFG["out_dir"], "composition.csv"), index=False)
    agg.to_csv(os.path.join(CFG["out_dir"], "forecast.csv"), index=False)
    print(f"저장 완료 → {CFG['out_dir']}/composition.csv, forecast.csv")
    return check_submission()


def check_submission(comp_path=None, fc_path=None):
    """제출 전 무효 조건 6가지 점검 (형식 오류도 제출 횟수에 포함되므로 반드시 실행)"""
    comp_path = comp_path or os.path.join(CFG["out_dir"], "composition.csv")
    fc_path = fc_path or os.path.join(CFG["out_dir"], "forecast.csv")
    rd = lambda p: pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    comp, fc = rd(comp_path), rd(fc_path)
    D = load_data()
    sites = D["sites"]
    errs = []
    if set(comp["site_id"]) != set(sites[C["site"]]) or len(comp) != len(sites) or comp["site_id"].duplicated().any():
        errs.append(f"① site_id 가 {len(sites)}개 정확히 한 번씩이 아님 (행 {len(comp)})")
    if not set(comp["group"]).issubset({"G1", "G2"}) or comp["group"].nunique() != 2:
        errs.append("② group 값이 G1/G2 이외이거나 한쪽이 비어 있음")
    cap = comp.merge(sites[[C["site"], C["cap"]]], left_on="site_id", right_on=C["site"]).groupby("group")[C["cap"]].sum()
    if (cap < CFG["min_cap_kw"]).any():
        errs.append(f"③ 집합 용량 미달: {cap.round(1).to_dict()}")
    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    exp_h = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    tsr = fc["ts"].astype(str)
    if not tsr.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$").all():
        errs.append("⑥ ts 에 타임존 표기 없음")
    try:
        k = fc.assign(_t=_to_kst(fc["ts"]))
        got = set(zip(k["group"], k["_t"]))
        want = {(g, t) for g in ("G1", "G2") for t in exp_h}
        if len(fc) != len(want) or got != want:
            errs.append(f"④ (group, ts) 가 {len(want):,}개와 일치하지 않음 (행 {len(fc):,}, 누락 {len(want - got):,}, 초과 {len(got - want):,})")
    except Exception as e:                                           # noqa
        errs.append(f"④ ts 해석 실패: {e}")
    v = pd.to_numeric(fc["forecast_kwh"], errors="coerce").values
    if (~np.isfinite(v)).any() or (v < 0).any():
        errs.append("⑤ 음수/NaN/inf 포함")
    if errs:
        print("✗ 제출 무효 위험:\n  " + "\n  ".join(errs))
    else:
        print(f"✓ 형식 점검 통과 | composition {len(comp)+1}행(헤더 포함) · forecast {len(fc)+1:,}행(헤더 포함) | 집합 용량 {cap.round(1).to_dict()}")
    return not errs


# ----------------------------------------------------------------------
def _in_notebook():
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


if __name__ == "__main__":
    if _in_notebook():
        print("노트북 환경: inspect() / selftest() / tester(2022, 2023) / year_grid([...]) / make_submission() 를 직접 호출하세요.")
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["inspect", "selftest", "test", "submit", "check"])
        ap.add_argument("--fit", type=int, nargs="+")
        ap.add_argument("--eval", type=int, nargs="+")
        ap.add_argument("--start-month", type=int, default=None)
        ap.add_argument("--data-dir", default=None)
        ap.add_argument("--backend", default=None, choices=["lightgbm", "sklearn"])
        a = ap.parse_args()
        if a.data_dir:
            CFG["data_dir"] = a.data_dir
        if a.backend:
            CFG["backend"] = a.backend
        {"inspect": inspect, "selftest": selftest, "submit": make_submission, "check": check_submission,
         "test": lambda: tester(a.fit, a.eval, a.start_month)}[a.cmd]()
