#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================
# 편의점 수요예측 & 발주 추천 — Pro Suite (패치 버전, 멀티 CSV + 월별 그래프)
#  - ① 여러 CSV 업로드/선택 → 자동 결합(옵션: source 열 추가)
#  - ② 컬럼 매핑: "컬럼명"이 아니라 "예시 값" 기반 선택
#  - ③ 예측·발주: 재고 컬럼 자동 인식 → 예측 기간/발주량 자동 계산
#       · 리드타임 / 서비스레벨 / 안전재고 / MOQ / 팩단위 입력 제거
#  - ④ 분석(그래프):
#       · 우산: 월별 강수량 ↔ 우산 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
#       · 군고구마: 월별 기온 ↔ 군고구마 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
#       · 전체: 우산·군고구마 제외 전체 상품 일별 판매량 선형 그래프
#  - 사이드바: 실행 파일 표시 + 캐시 초기화
# ============================================================

import os, io, pickle, time, subprocess, sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from utils_io import read_csv_flexible, save_utf8sig, ensure_dirs, auto_map_columns
from preprocess import make_matrix
from train_core import train_and_score, save_artifacts

# Altair 대용량 렌더링 안전장치 (행 수 제한 해제)
alt.data_transformers.disable_max_rows()

# ------------------------------------------------------------
# 페이지/사이드바
# ------------------------------------------------------------
st.set_page_config(page_title="편의점 수요예측 & 발주 추천 — Pro Suite (패치)", layout="wide")

# __file__ 이 없는 Colab 같은 환경 방어용
try:
    script_name = Path(__file__).resolve().name
except NameError:
    script_name = "app_streamlit_pro.py"

st.sidebar.write("🧭 실행 파일:", script_name)
if st.sidebar.button("캐시 초기화 후 다시 실행"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.experimental_rerun()

# ------------------------------------------------------------
# 기본 환경/경로 설정
# ------------------------------------------------------------
PROJ = os.getcwd()                         # 현재 작업 디렉토리(앱 루트)
DATA_DIR   = os.path.join(PROJ, "data")    # CSV 데이터 폴더
ARTI_DIR   = os.path.join(PROJ, "artifacts")  # 학습 중간산출물(로그/성능 등) 보관
MODELS_DIR = os.path.join(PROJ, "models")     # 학습된 모델 pkl 보관
ensure_dirs(DATA_DIR, ARTI_DIR, MODELS_DIR)   # 폴더 없으면 생성

# ------------------------------------------------------------
# 유틸: data 폴더의 CSV 파일 리스트 캐시
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_data_files():
    try:
        return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    except FileNotFoundError:
        return []

# ------------------------------------------------------------
# 퍼블릭 URL: cloudflared 시작 함수
# ------------------------------------------------------------
def start_cloudflared(port=8501):
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        st.session_state["_cfd_proc"] = proc  # 종료용 핸들 저장
        with st.expander("cloudflared logs"):
            for _ in range(120):  # 최초 120라인 정도만 읽어 표시
                line = proc.stdout.readline()
                if not line:
                    break
                st.text(line.strip())
                if "trycloudflare.com" in line:
                    st.success(line.strip())  # 퍼블릭 URL 포함 로그
                    break
    except FileNotFoundError:
        st.error("cloudflared 바이너리가 없습니다. `pip install cloudflared` 또는 바이너리 설치 후 다시 시도하세요.")

# ------------------------------------------------------------
# 퍼블릭 URL: ngrok 시작 함수
# ------------------------------------------------------------
def start_ngrok(port=8501, token: str | None = None):
    try:
        from pyngrok import ngrok, conf
    except Exception:
        st.error("pyngrok가 설치되어 있지 않습니다. `pip install pyngrok` 후 다시 시도하세요.")
        return

    # 기존 ngrok 세션 정리(재실행 시 충돌 방지)
    try:
        ngrok.kill()
        time.sleep(1.0)
    except Exception:
        pass

    token = (token or os.environ.get("NGROK_AUTHTOKEN", "")).strip()
    if token:
        conf.get_default().auth_token = token
    else:
        st.warning("NGROK_AUTHTOKEN이 비어 있습니다. 인증 없이 열면 제한/에러(4018) 가능.")

    for attempt in range(2):
        try:
            tunnel = ngrok.connect(addr=f"http://localhost:{port}", proto="http")
            url = tunnel.public_url
            st.session_state["_ngrok_tunnel"] = tunnel
            st.success(f"🌐 Public URL: {url}")
            st.caption("런타임/프로세스를 종료하면 터널도 닫힙니다.")
            break
        except Exception as e:
            if attempt == 0:
                time.sleep(1.5)
            else:
                msg = str(e)
                if "4018" in msg:
                    st.error("ngrok 인증 실패(4018). 토큰을 다시 확인하세요.")
                elif "already online" in msg or "334" in msg:
                    st.error("동일 엔드포인트가 이미 열려 있습니다. 세션 재시작 또는 기존 터널 종료 후 재시도.")
                else:
                    st.error(f"ngrok 연결 실패: {e}")

# ------------------------------------------------------------
# 앱 타이틀/탭 구성
# ------------------------------------------------------------
st.title("편의점 수요예측 & 발주 추천 — Pro Suite")
tabs = st.tabs(["① 데이터", "② 학습/모델", "③ 예측·발주", "④ 분석(그래프)", "⑤ 진단/로그"])

# ============================================================
# ① 데이터: CSV 업로드/선택 + 자동 컬럼 매핑 저장 (멀티 CSV 지원)
# ============================================================
with tabs[0]:
    st.subheader("CSV 업로드 또는 선택")
    cols_top = st.columns([2,1])
    with cols_top[0]:
        add_source = st.checkbox("파일명(source) 열 추가", value=True, help="여러 CSV를 합칠 때 원본 파일명을 남깁니다.")
    with cols_top[1]:
        st.caption("※ 업로드/선택 후 아래에서 컬럼 매핑 저장")

    cols = st.columns(2)

    # --- 다중 파일 업로드 ---
    with cols[0]:
        up_multi = st.file_uploader("CSV 파일 업로드(여러 개 가능)", type=["csv"], accept_multiple_files=True, key="multi_up")
        if up_multi:
            dfs = []
            for f in up_multi:
                raw = f.read()
                df_i = read_csv_flexible(io.BytesIO(raw))
                if add_source:
                    df_i["source"] = f.name
                dfs.append(df_i)
                # data/에 저장
                save_path = os.path.join(DATA_DIR, f.name)
                try:
                    with open(save_path, "wb") as fp:
                        fp.write(raw)
                except Exception as e:
                    st.warning(f"파일 저장 경고({f.name}): {e}")
            try:
                list_data_files.clear()  # 캐시 무효화
            except Exception:
                pass
            df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
            st.session_state["df"] = df
            st.success(f"업로드/결합 완료: {df.shape} (파일 {len(dfs)}개)")
            st.dataframe(df.head(20), use_container_width=True)

    # --- data 폴더에서 다중 선택 ---
    with cols[1]:
        files = list_data_files()
        picks = st.multiselect("data 폴더에서 선택(여러 개)", files)
        if st.button("선택 파일 불러오기", disabled=(len(picks)==0)):
            dfs = []
            for name in picks:
                path = os.path.join(DATA_DIR, name)
                df_i = read_csv_flexible(path)
                if add_source:
                    df_i["source"] = name
                dfs.append(df_i)
            df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
            st.session_state["df"] = df
            st.success(f"불러오기/결합 완료: {df.shape} (파일 {len(dfs)}개)")
            st.dataframe(df.head(20), use_container_width=True)

    # --- 자동 컬럼 매핑 + 보정 ---
    if "df" in st.session_state:
        st.divider()
        st.caption("자동 컬럼 매핑 — 선택 없이 자동 적용됩니다.")

        df = st.session_state["df"]

        # auto_map_columns 결과 사용
        auto = auto_map_columns(df)
        mapping = {
            "date": auto.get("date"),
            "target": auto.get("target"),
            "region": auto.get("region"),
            "brand": auto.get("brand"),
            "item": auto.get("item"),
        }
        st.session_state["mapping"] = mapping

        # ★ data 폴더용 보정:
        #   seoul_gyeonggi_with_demand.csv / usan.csv / gungoguma.csv 는
        #   auto_map_columns가 타깃을 '강수량'으로 잡는 케이스가 있어서,
        #   '일일판매량' 컬럼이 있으면 그걸 target으로 강제 교체
        if mapping.get("target") == "강수량" and "일일판매량" in df.columns:
            mapping["target"] = "일일판매량"

        # 확인용으로만 읽기 전용 테이블 표시
        mapping_view = pd.DataFrame(
            {
                "역할": ["날짜(date)", "수요/판매량(target)", "지역/점포(region)", "브랜드(선택)", "상품/품목(선택)"],
                "컬럼": [
                    mapping.get("date"),
                    mapping.get("target"),
                    mapping.get("region"),
                    mapping.get("brand"),
                    mapping.get("item"),
                ],
            }
        )

        st.write("현재 자동 매핑 결과:")
        st.dataframe(mapping_view, use_container_width=True)

# ============================================================
# ② 학습/모델
# ============================================================
with tabs[1]:
    st.subheader("모델 학습")

    use_optuna = st.checkbox("Optuna 하이퍼파라미터 튜닝 사용", value=False)
    trials = st.slider("Optuna 시도 횟수", 5, 60, 15, 5)

    if "df" not in st.session_state or "mapping" not in st.session_state:
        st.info("먼저 ① 탭에서 데이터와 컬럼 매핑을 지정하세요.")
    else:
        v = st.slider("검증 비율(valid_ratio)", 0.05, 0.4, 0.2, 0.05)

        if st.button("학습 시작"):
            # ➜ 여기서 예외가 나도 앱이 죽지 않도록 방어
            try:
                df, X, y, feat_names = make_matrix(
                    st.session_state["df"],
                    st.session_state["mapping"],
                )
            except Exception as e:
                st.error(f"학습용 데이터 구성 중 오류가 발생했습니다: {e}")
            else:
                try:
                    best_model, lb = train_and_score(
                        X,
                        y,
                        valid_ratio=v,
                        use_optuna=use_optuna,
                        optuna_trials=trials,
                    )
                    save_artifacts(
                        [ARTI_DIR, MODELS_DIR],
                        best_model,
                        feat_names,
                        st.session_state["mapping"],
                        lb,
                    )
                except Exception as e:
                    st.error(f"모델 학습/저장 중 오류가 발생했습니다: {e}")
                else:
                    st.session_state["leaderboard"] = lb
                    st.session_state["feat_names"] = feat_names
                    st.success("학습 완료")

        if "leaderboard" in st.session_state:
            st.dataframe(st.session_state["leaderboard"], use_container_width=True)

# ============================================================
# ③ 예측·발주: 반복(AR) 예측 + 재고 기반 자동 발주 계산
# ============================================================
with tabs[2]:
    st.subheader("예측(반복 AR) & 발주량 추천")
    st.caption("학습된 모델로 미래 피처를 생성하고, 재고를 고려해 자동으로 발주 기간과 수량을 계산합니다.")

    if "df" not in st.session_state or "mapping" not in st.session_state:
        st.info("먼저 ① 탭에서 데이터와 컬럼 매핑을 지정하고 ②에서 학습을 완료하세요.")
    else:
        horizon_days = 14  # 고정 기간

        # 정확도(보정 계수)
        accuracy = st.slider(
            "정확도(예측 보정 계수)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )

        # ==============================
        # 세그먼트 선택
        # ==============================
        seg_cols = [
            c for c in [
                st.session_state["mapping"].get("region"),
                st.session_state["mapping"].get("brand"),
                st.session_state["mapping"].get("item"),
            ] if c
        ]
        seg_vals = {}
        if seg_cols:
            col_objs = st.columns(len(seg_cols))
            for i, ccol in enumerate(seg_cols):
                opts = ["<전체>"] + sorted(
                    list(map(str, st.session_state["df"][ccol].dropna().astype(str).unique()))
                )
                seg_vals[ccol] = col_objs[i].selectbox(f"{ccol} 선택", opts, index=0)

        # ==============================
        # 반복 예측 함수
        # ==============================
        def iterative_forecast(df, mapping, model, feat_names, horizon, seg_vals):
            df = df.copy()
            dtc = mapping["date"]
            tgt = mapping["target"]

            if dtc not in df.columns or tgt not in df.columns:
                st.error(f"예측에 필요한 컬럼이 없습니다. (date='{dtc}', target='{tgt}')")
                return pd.DataFrame(columns=[dtc, "예측수량"])

            df[dtc] = pd.to_datetime(df[dtc], errors="coerce")
            df = df.dropna(subset=[dtc]).sort_values(dtc)

            for k, v in seg_vals.items():
                if v and v != "<전체>" and k in df.columns:
                    df = df[df[k].astype(str) == str(v)]

            if df.empty:
                st.error("선택한 세그먼트에 해당하는 데이터가 없습니다.")
                return pd.DataFrame(columns=[dtc, "예측수량"])

            if len(df) < 30:
                st.warning("해당 세그먼트 데이터가 적어 예측 품질이 낮을 수 있습니다.")

            last_date = df[dtc].max()

            hist = list(
                pd.to_numeric(df[tgt], errors="coerce")
                .fillna(0)
                .astype(float)
                .values
            )

            def build_row_features(current_date, hist_vals):
                if pd.isna(current_date):
                    current_date = df[dtc].max()

                year = current_date.year
                month = current_date.month
                day = current_date.day
                dow = current_date.weekday()
                is_weekend = 1 if dow >= 5 else 0

                try:
                    week = int(pd.Timestamp(current_date).isocalendar().week)
                except Exception:
                    week = 0

                def get_lag(k):
                    if len(hist_vals) >= k:
                        return float(hist_vals[-k])
                    return float(np.mean(hist_vals[-min(len(hist_vals), 7):])) if hist_vals else 0.0

                lag1 = get_lag(1)
                lag7 = get_lag(7)
                lag14 = get_lag(14)

                def rmean(w):
                    arr = np.array(hist_vals[-w:]) if len(hist_vals) >= 1 else np.array([0.0])
                    if len(arr) < max(2, w // 2):
                        arr = np.array(hist_vals[-max(2, w // 2):]) if len(hist_vals) else np.array([0.0])
                    return float(np.mean(arr))

                def rstd(w):
                    arr = np.array(hist_vals[-w:]) if len(hist_vals) >= 2 else np.array([0.0, 0.0])
                    return float(np.std(arr))

                feats = {
                    "year": year,
                    "month": month,
                    "day": day,
                    "dow": dow,
                    "week": week,
                    "is_weekend": is_weekend,
                    "lag1": lag1,
                    "lag7": lag7,
                    "lag14": lag14,
                    "rmean7": rmean(7),
                    "rmean14": rmean(14),
                    "rstd7": rstd(7),
                    "rstd14": rstd(14),
                }

                for fn in feat_names:
                    if fn not in feats:
                        feats[fn] = 0.0

                x = [feats.get(fn, 0.0) for fn in feat_names]
                return np.array(x, dtype=float)

            preds, dates = [], []
            cur = last_date
            for _ in range(int(horizon)):
                cur = cur + timedelta(days=1)
                x = build_row_features(cur, hist)
                val = float(model.predict([x])[0])
                preds.append(val)
                dates.append(cur)
                hist.append(val)

            return pd.DataFrame({dtc: dates, "예측수량": preds})

        # ==============================
        # 재고 자동 인식
        # ==============================
        def guess_inventory_onhand(df_seg: pd.DataFrame, mapping):
            candidates = [
                "재고", "재고수", "재고수량",
                "현재재고", "onhand", "on_hand",
                "stock", "inventory",
            ]
            inv_col = None
            for col in df_seg.columns:
                low = col.lower()
                if any(key in low for key in candidates):
                    inv_col = col
                    break
            if not inv_col:
                return None, None

            series = pd.to_numeric(df_seg[inv_col], errors="coerce").dropna()
            if series.empty:
                return None, None

            return inv_col, float(series.iloc[-1])

        # ==============================
        # 가격 자동 인식
        # ==============================
        def guess_price_column(df_seg):
            keys = ["price", "가격", "단가", "판매가", "amount", "금액"]
            for col in df_seg.columns:
                low = col.lower()
                if any(k in low for k in keys):
                    return col
            return None

        # ==============================
        # 모델 로드
        # ==============================
        pkl_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    payload = pickle.load(f)
                model = payload["model"]
                feat_names = payload["feature_names"]
                mapping = payload["mapping"]
            except Exception as e:
                st.error(f"저장된 모델 로딩 중 오류: {e}")
            else:
                dtc = mapping["date"]

                # ======================================
                # 1) 예측 수행
                # ======================================
                fc_df = iterative_forecast(
                    st.session_state["df"],
                    mapping,
                    model,
                    feat_names,
                    horizon_days,
                    seg_vals,
                )
                if fc_df.empty:
                    st.stop()

                # ======================================
                # 2) 가격 자동 인식 + 금액예측
                # ======================================
                df_seg_price = st.session_state["df"].copy()
                for k, v in seg_vals.items():
                    if v and v != "<전체>" and k in df_seg_price.columns:
                        df_seg_price = df_seg_price[df_seg_price[k].astype(str) == str(v)]
                df_seg_price = df_seg_price.sort_values(dtc)

                price_col = guess_price_column(df_seg_price)

                if price_col:
                    price_val = float(
                        pd.to_numeric(df_seg_price[price_col], errors="coerce").dropna().iloc[-1]
                    )
                    st.info(f"CSV '{price_col}' 컬럼에서 가격 {price_val:,.0f}원 자동 인식.")
                else:
                    price_val = st.number_input(
                        "가격(원) – CSV에서 가격 컬럼을 찾지 못해 직접 입력",
                        min_value=0,
                        max_value=100000000,
                        value=0,
                    )

                # **수량 총합**
                total_qty_demand = float(fc_df["예측수량"].sum())

                # **금액 총합**
                fc_df["금액예측"] = (fc_df["예측수량"] * price_val * float(accuracy)).clip(lower=0.0)
                total_amt_demand = float(fc_df["금액예측"].sum())

                # ======================================
                # 3) 재고 자동 인식
                # ======================================
                df_seg = st.session_state["df"].copy()
                df_seg[dtc] = pd.to_datetime(df_seg[dtc], errors="coerce")
                for k, v in seg_vals.items():
                    if v and v != "<전체>" and k in df_seg.columns:
                        df_seg = df_seg[df_seg[k].astype(str) == str(v)]
                df_seg = df_seg.sort_values(dtc)

                inv_col, onhand_auto = guess_inventory_onhand(df_seg, mapping)
                if onhand_auto is None:
                    onhand = st.number_input(
                        "현재 재고(직접 입력)",
                        min_value=0,
                        max_value=100000,
                        value=0,
                    )
                else:
                    onhand = onhand_auto
                    st.info(f"재고 '{inv_col}' 자동 인식 → {onhand:,.0f}개")

                # ======================================
                # 4) 발주량/소진일 계산 (수량 기준)
                # ======================================
                avg_daily_qty = total_qty_demand / horizon_days if horizon_days > 0 else 0.0
                days_to_out = (onhand / avg_daily_qty) if avg_daily_qty > 0 else float("inf")
                rec_qty = max(0.0, total_qty_demand - onhand)

                c1, c2, c3 = st.columns(3)
                c1.metric("예측 기간(일)", f"{horizon_days}")
                c2.metric("재고 소진 예상일수", "∞" if np.isinf(days_to_out) else f"{days_to_out:,.1f}")
                c3.metric("2주 총 예상 매출", f"{total_amt_demand:,.0f}원")

                # ======================================
                # 5) 표 출력
                # ======================================
                st.dataframe(fc_df.set_index(dtc), use_container_width=True)
                st.caption("※ 예측수량 × 가격 × 정확도 보정 = 금액예측")

        else:
            st.warning("best_model.pkl 이 없습니다. ② 탭에서 학습을 먼저 수행하세요.")

# ============================================================
# ④ 분석(그래프):
#   - 우산: 한 달 강수량 vs 우산 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
#   - 군고구마: 한 달 기온 vs 군고구마 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
#   - 전체: 우산·군고구마 제외 일별 판매량 선형 그래프
# ============================================================
with tabs[3]:
    st.subheader("분석(그래프) — 한 달 단위 상관 분석")

    if "df" not in st.session_state or "mapping" not in st.session_state or not st.session_state["mapping"].get("date"):
        st.info("먼저 ① 탭에서 데이터와 컬럼 매핑(특히 '날짜'와 '타깃')을 지정하세요.")
    else:
        mapping = st.session_state["mapping"]
        date_col = mapping["date"]
        target_col = mapping.get("target")

        def guess(colnames, cands):
            low = [str(c).lower() for c in colnames]
            for key in cands:
                key_low = str(key).lower()
                for i, l in enumerate(low):
                    if key_low in l:
                        return colnames[i]
            return None

        # 공통: 연-월 선택용 옵션 만드는 함수
        def build_year_month_options(df, date_col):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            if df.empty:
                return df, []
            df["year_month"] = df[date_col].dt.to_period("M")
            ym_unique = sorted(df["year_month"].unique())
            ym_labels = [str(p) for p in ym_unique]  # '2024-10' 같은 형식
            return df, list(zip(ym_labels, ym_unique))

        tab_u, tab_g, tab_all = st.tabs([
            "☔ 우산: 한 달 강수량 vs 판매량",
            "🍠 군고구마: 한 달 기온 vs 판매량",
            "📈 전체: 우산·군고구마 제외 일별 판매량(선형)"
        ])

        # ------------------------------
        # 1) 우산: 선택한 한 달의 강수량 ↔ 우산 판매량
        # ------------------------------
        with tab_u:
            st.caption("우산 판매량과 강수량의 관계를 '한 달' 단위로 봅니다.")

            up_u = st.file_uploader("우산/날씨 데이터 CSV (선택)", type=["csv"], key="umbrella_month_up")
            if up_u is not None:
                df_u_raw = read_csv_flexible(io.BytesIO(up_u.read()))
            else:
                df_u_raw = st.session_state["df"].copy()

            if date_col not in df_u_raw.columns:
                st.warning(f"날짜 컬럼 '{date_col}' 을(를) 데이터에서 찾지 못했습니다.")
            else:
                # item에서 우산만 필터 (있으면)
                item_col = mapping.get("item")
                if item_col and item_col in df_u_raw.columns:
                    mask = df_u_raw[item_col].astype(str).str.contains("우산|umbrella", case=False, na=False)
                    if mask.any():
                        df_u_raw = df_u_raw[mask]

                cols_all = list(df_u_raw.columns)

                # 판매량 컬럼: 매핑 target 우선, 없으면 추정
                sales_col = target_col if target_col in cols_all else guess(
                    cols_all,
                    ["umbrella", "우산", "일일판매량", "판매량", "sales", "qty", "quantity", "target"],
                )

                # 강수량 컬럼 추정
                rain_col = guess(
                    cols_all,
                    ["rain", "precip", "precipitation", "강수", "강수량", "일강수량", "강우", "강우량"],
                )

                if not sales_col or not rain_col:
                    st.warning(
                        "우산 판매량 또는 강수량 컬럼을 자동으로 찾지 못했습니다.\n"
                        "판매량: '우산/umbrella/판매량/sales', 강수량: '강수량/rain' 등의 이름을 사용해 주세요."
                    )
                else:
                    # 날짜/숫자 형식 정리 + 연-월 옵션 생성
                    df_u_raw[sales_col] = pd.to_numeric(df_u_raw[sales_col], errors="coerce")
                    df_u_raw[rain_col] = pd.to_numeric(df_u_raw[rain_col], errors="coerce")

                    df_u_raw, ym_options = build_year_month_options(df_u_raw, date_col)

                    if not ym_options:
                        st.info("유효한 날짜 데이터가 없습니다.")
                    else:
                        # 연-월 선택 (YYYY-MM 형식만 보여줌)
                        labels = [lab for lab, _ in ym_options]
                        default_idx = len(labels) - 1  # 기본값: 가장 최근 월
                        sel_label = st.selectbox("분석할 연월(YYYY-MM)", labels, index=default_idx, key="ym_umbrella")
                        sel_period = dict(ym_options)[sel_label]

                        # 선택한 한 달만 필터
                        df_month = df_u_raw[df_u_raw["year_month"] == sel_period].copy()
                        if df_month.empty:
                            st.info(f"{sel_label} 에 해당하는 데이터가 없습니다.")
                        else:
                            # 일 단위 집계
                            df_month["date_only"] = df_month[date_col].dt.date
                            daily = (
                                df_month.groupby("date_only", as_index=False)
                                .agg({sales_col: "sum", rain_col: "mean"})
                                .dropna(subset=[sales_col, rain_col])
                            )
                            daily = daily.rename(
                                columns={"date_only": "date", sales_col: "sales", rain_col: "rain"}
                            )

                            if daily.empty:
                                st.info("해당 연월에서 일별로 집계할 수 있는 데이터가 없습니다.")
                            else:
                                st.markdown(f"**{sel_label} 한 달 기준 · 강수량에 따른 우산 판매량**")

                                base = alt.Chart(daily).encode(
                                    x=alt.X("rain:Q", title="일 강수량"),
                                    y=alt.Y("sales:Q", title="일 우산 판매량"),
                                )

                                # 붉은색 산점도 + 선형 회귀선
                                points = base.mark_circle(size=70, color="#d62728").encode(
                                    tooltip=[
                                        alt.Tooltip("date:T", title="날짜"),
                                        alt.Tooltip("rain:Q", title="강수량"),
                                        alt.Tooltip("sales:Q", title="우산 판매량"),
                                    ]
                                )
                                reg_line = base.transform_regression("rain", "sales").mark_line(color="#b22222")

                                st.altair_chart((points + reg_line).interactive(), use_container_width=True)

                                # ★ 추가: 일별 우산 판매량 선형 그래프
                                st.markdown("**일별 우산 판매량 추세(선형 그래프)**")
                                line_umbrella = (
                                    alt.Chart(daily)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="날짜"),
                                        y=alt.Y("sales:Q", title="일 우산 판매량"),
                                        tooltip=[
                                            alt.Tooltip("date:T", title="날짜"),
                                            alt.Tooltip("sales:Q", title="우산 판매량"),
                                            alt.Tooltip("rain:Q", title="강수량"),
                                        ],
                                    )
                                )
                                st.altair_chart(line_umbrella.interactive(), use_container_width=True)

                                # 참고용 테이블
                                st.dataframe(daily, use_container_width=True)

        # ------------------------------
        # 2) 군고구마: 선택한 한 달의 기온 ↔ 군고구마 판매량
        # ------------------------------
        with tab_g:
            st.caption("군고구마 판매량과 기온(추위)의 관계를 '한 달' 단위로 봅니다.")

            up_g = st.file_uploader("군고구마/날씨 데이터 CSV (선택)", type=["csv"], key="goguma_month_up")
            if up_g is not None:
                df_g_raw = read_csv_flexible(io.BytesIO(up_g.read()))
            else:
                df_g_raw = st.session_state["df"].copy()

            if date_col not in df_g_raw.columns:
                st.warning(f"날짜 컬럼 '{date_col}' 을(를) 데이터에서 찾지 못했습니다.")
            else:
                # item에서 군고구마만 필터 (있으면)
                item_col_g = mapping.get("item")
                if item_col_g and item_col_g in df_g_raw.columns:
                    mask_g = df_g_raw[item_col_g].astype(str).str.contains(
                        "고구마|군고구마|sweet|goguma", case=False, na=False
                    )
                    if mask_g.any():
                        df_g_raw = df_g_raw[mask_g]

                cols_all_g = list(df_g_raw.columns)

                goguma_col = target_col if target_col in cols_all_g else guess(
                    cols_all_g,
                    ["고구마", "군고구마", "sweetpotato", "goguma", "판매량", "sales", "qty", "quantity", "target"],
                )
                temp_col = guess(
                    cols_all_g,
                    ["온도", "tmin", "temp_min", "min_temp", "최저", "최저기온", "일최저기온", "temperature", "temp"],
                )

                if not goguma_col or not temp_col:
                    st.warning(
                        "군고구마 판매량 또는 기온 컬럼을 자동으로 찾지 못했습니다.\n"
                        "판매량: '군고구마/고구마/sales/target', 기온: 'tmin/최저기온/temperature' 등의 이름을 사용해 주세요."
                    )
                else:
                    df_g_raw[goguma_col] = pd.to_numeric(df_g_raw[goguma_col], errors="coerce")
                    df_g_raw[temp_col] = pd.to_numeric(df_g_raw[temp_col], errors="coerce")

                    df_g_raw, ym_options_g = build_year_month_options(df_g_raw, date_col)

                    if not ym_options_g:
                        st.info("유효한 날짜 데이터가 없습니다.")
                    else:
                        labels_g = [lab for lab, _ in ym_options_g]
                        default_idx_g = len(labels_g) - 1
                        sel_label_g = st.selectbox("분석할 연월(YYYY-MM)", labels_g, index=default_idx_g, key="ym_goguma")
                        sel_period_g = dict(ym_options_g)[sel_label_g]

                        df_month_g = df_g_raw[df_g_raw["year_month"] == sel_period_g].copy()
                        if df_month_g.empty:
                            st.info(f"{sel_label_g} 에 해당하는 데이터가 없습니다.")
                        else:
                            df_month_g["date_only"] = df_month_g[date_col].dt.date
                            daily_g = (
                                df_month_g.groupby("date_only", as_index=False)
                                .agg({goguma_col: "sum", temp_col: "mean"})
                                .dropna(subset=[goguma_col, temp_col])
                            )
                            daily_g = daily_g.rename(
                                columns={"date_only": "date", goguma_col: "sales", temp_col: "temp"}
                            )

                            if daily_g.empty:
                                st.info("해당 연월에서 일별로 집계할 수 있는 데이터가 없습니다.")
                            else:
                                st.markdown(f"**{sel_label_g} 한 달 기준 · 기온에 따른 군고구마 판매량**")

                                base_g = alt.Chart(daily_g).encode(
                                    x=alt.X("temp:Q", title="일 평균 기온"),
                                    y=alt.Y("sales:Q", title="일 군고구마 판매량"),
                                )

                                points_g = base_g.mark_circle(size=70, color="#ff7f0e").encode(
                                    tooltip=[
                                        alt.Tooltip("date:T", title="날짜"),
                                        alt.Tooltip("temp:Q", title="기온"),
                                        alt.Tooltip("sales:Q", title="군고구마 판매량"),
                                    ]
                                )
                                reg_g = base_g.transform_regression("temp", "sales").mark_line(color="#d35400")

                                st.altair_chart((points_g + reg_g).interactive(), use_container_width=True)

                                # ★ 추가: 일별 군고구마 판매량 선형 그래프
                                st.markdown("**일별 군고구마 판매량 추세(선형 그래프)**")
                                line_goguma = (
                                    alt.Chart(daily_g)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="날짜"),
                                        y=alt.Y("sales:Q", title="일 군고구마 판매량"),
                                        tooltip=[
                                            alt.Tooltip("date:T", title="날짜"),
                                            alt.Tooltip("temp:Q", title="기온"),
                                            alt.Tooltip("sales:Q", title="군고구마 판매량"),
                                        ],
                                    )
                                )
                                st.altair_chart(line_goguma.interactive(), use_container_width=True)

                                st.dataframe(daily_g, use_container_width=True)

        # ------------------------------
        # 3) 전체: 우산·군고구마 제외 전체 상품 일별 판매량 선형 그래프
        # ------------------------------
        with tab_all:
            st.caption("우산·군고구마를 제외한 모든 상품의 일별 판매량 추세를 한 번에 봅니다.")

            df_all = st.session_state["df"].copy()

            if date_col not in df_all.columns or not target_col or target_col not in df_all.columns:
                st.warning(f"날짜('{date_col}') 또는 타깃('{target_col}') 컬럼을 찾을 수 없습니다.")
            else:
                # item 컬럼이 있으면 우산/군고구마 관련 상품 제외
                item_col_all = mapping.get("item")
                if item_col_all and item_col_all in df_all.columns:
                    ex_mask = df_all[item_col_all].astype(str).str.contains(
                        "우산|umbrella|고구마|군고구마|sweet|goguma", case=False, na=False
                    )
                    df_all = df_all[~ex_mask]

                df_all[target_col] = pd.to_numeric(df_all[target_col], errors="coerce")

                df_all, ym_options_all = build_year_month_options(df_all, date_col)

                if not ym_options_all:
                    st.info("유효한 날짜 데이터가 없습니다.")
                else:
                    labels_all = [lab for lab, _ in ym_options_all]
                    default_idx_all = len(labels_all) - 1
                    sel_label_all = st.selectbox(
                        "분석할 연월(YYYY-MM)",
                        labels_all,
                        index=default_idx_all,
                        key="ym_all",
                    )
                    sel_period_all = dict(ym_options_all)[sel_label_all]

                    df_month_all = df_all[df_all["year_month"] == sel_period_all].copy()
                    if df_month_all.empty:
                        st.info(f"{sel_label_all} 에 해당하는 데이터가 없습니다.")
                    else:
                        df_month_all["date_only"] = df_month_all[date_col].dt.date
                        daily_all = (
                            df_month_all.groupby("date_only", as_index=False)
                            .agg({target_col: "sum"})
                            .dropna(subset=[target_col])
                        )
                        daily_all = daily_all.rename(
                            columns={"date_only": "date", target_col: "sales"}
                        )

                        if daily_all.empty:
                            st.info("해당 연월에서 일별로 집계할 수 있는 데이터가 없습니다.")
                        else:
                            st.markdown(f"**{sel_label_all} 한 달 기준 · 우산·군고구마 제외 전체 상품 일별 판매량(선형)**")

                            line_all = (
                                alt.Chart(daily_all)
                                .mark_line()
                                .encode(
                                    x=alt.X("date:T", title="날짜"),
                                    y=alt.Y("sales:Q", title="일 판매량(전체 상품 합계)"),
                                    tooltip=[
                                        alt.Tooltip("date:T", title="날짜"),
                                        alt.Tooltip("sales:Q", title="일 판매량 합계"),
                                    ],
                                )
                            )
                            st.altair_chart(line_all.interactive(), use_container_width=True)
                            st.dataframe(daily_all, use_container_width=True)

# ============================================================
# ⑤ 진단/로그: 경로/파일 확인 + 퍼블릭 URL 열기/닫기
# ============================================================
with tabs[4]:
    st.subheader("경로/파일 상태")

    cols = st.columns(2)
    with cols[0]:
        st.write("**data**", DATA_DIR)
        st.write(os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else [])
        st.write("**artifacts**", ARTI_DIR)
        st.write(os.listdir(ARTI_DIR) if os.path.exists(ARTI_DIR) else [])
    with cols[1]:
        st.write("**models**", MODELS_DIR)
        st.write(os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else [])

    st.caption("필요 시 퍼블릭 URL을 열어 외부에서 접속할 수 있습니다.")
    mode = st.radio("퍼블릭 URL 터널러", ["ngrok", "cloudflared"], horizontal=True, index=0)

    ngk = None
    if mode == "ngrok":
        ngk = st.text_input(
            "NGROK_AUTHTOKEN",
            value=os.environ.get("NGROK_AUTHTOKEN", ""),
            type="password",
            help="환경변수에 넣어두면 다음부터 자동 인식합니다.",
        )

    c_open, c_close = st.columns(2)
    if c_open.button("퍼블릭 URL 열기", use_container_width=True):
        if mode == "ngrok":
            if ngk:
                os.environ["NGROK_AUTHTOKEN"] = ngk
            start_ngrok()
        else:
            start_cloudflared()

    if c_close.button("퍼블릭 URL 닫기", use_container_width=True):
        if mode == "ngrok":
            try:
                from pyngrok import ngrok
                ngrok.kill()
                st.info("ngrok 터널을 종료했습니다.")
            except Exception as e:
                st.warning(f"ngrok 종료 중 경고: {e}")
        else:
            proc = st.session_state.get("_cfd_proc")
            if proc:
                proc.terminate()
                st.info("cloudflared 터널을 종료했습니다.")
            else:
                st.info("cloudflared 활성 프로세스가 없습니다.")
