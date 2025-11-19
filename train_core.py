#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
train_core.py — 학습 핵심 로직(주석 아주 자세히)
------------------------------------------------------------
이 파일은 다음 일을 해요:
1) 평가 지표 함수 정의(RMSE/MAE/MAPE)
2) 사용할 모델 후보들을 모아주는 함수(get_candidates)
3) 시계열 분할(학습/검증 나누기)
4) 간단한 앙상블(SimpleEnsemble)
5) (옵션) Optuna 로 하이퍼파라미터 튜닝
6) train_and_score: 모델들 학습 → 검증 성능 비교 → 베스트 선택
7) save_artifacts: 베스트 모델/리더보드 저장

※ XGBoost/LightGBM/Optuna 는 설치되어 있지 않으면
   자동으로 건너뛰도록 만들어졌습니다.
============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd

# 평가 지표 계산을 위해 scikit-learn 함수 사용
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 기본 선형회귀/랜덤포레스트
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# XGBoost / LightGBM 은 있을 수도, 없을 수도 있어요. (try/except)
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

# Optuna(하이퍼파라미터 자동 탐색기)도 선택사항
try:
    import optuna
except Exception:
    optuna = None


# ------------------------------------------------------------
# 1) 평가 지표: RMSE / MAE / MAPE
# ------------------------------------------------------------
def rmse(a, b):
    """
    RMSE (Root Mean Squared Error)
    - 예측이 실제와 얼마나 다른지, '제곱 평균 오차의 제곱근'
    - 값이 작을수록 좋아요.
    """
    a = np.array(a); b = np.array(b)
    return float(np.sqrt(mean_squared_error(a, b))) if len(a) else float("nan")


def mae(a, b):
    """
    MAE (Mean Absolute Error)
    - 예측과 실제의 차이의 '절대값'을 평균낸 값
    - 쉬운 직관: 평균적으로 몇 개(또는 몇 단위) 만큼 틀렸나?
    """
    a = np.array(a); b = np.array(b)
    return float(mean_absolute_error(a, b)) if len(a) else float("nan")


def mape(a, b):
    """
    MAPE (Mean Absolute Percentage Error)
    - 퍼센트(%) 기준 오차. 10%면 '평균적으로 10% 틀렸다'는 뜻.
    - 실제값이 0이면 나눗셈이 안 되므로 1로 바꿔서 안전 처리해요.
    """
    a = np.array(a); b = np.array(b)
    if len(a) == 0:
        return float("nan")
    denom = np.where(a == 0, 1, a)  # 0인 곳은 1로 치환(분모 안전장치)
    return float(np.mean(np.abs((a - b) / denom)) * 100.0)


# ------------------------------------------------------------
# 2) 모델 후보를 만들어 주는 함수
# ------------------------------------------------------------
def get_candidates():
    """
    사용할 수 있는 모델 목록을 튜플로 모아 반환해요.
    각 원소: (이름, 모델객체, fit(학습)할 때 넣을 추가 파라미터 딕셔너리)

    - LinearRegression: 가장 기본적인 선형 모델
    - RandomForest: 비선형 패턴도 잘 잡는 나무 앙상블
    - XGBoost / LightGBM: 빠르고 강력한 부스팅 모델(설치된 경우만 사용)
    """
    models = []

    # 1) 선형회귀 (설정할 게 거의 없음)
    models.append(("LinearRegression", LinearRegression(), {}))

    # 2) 랜덤포레스트 (나무 300그루, 멀티코어 사용)
    models.append(("RandomForest", RandomForestRegressor(
        n_estimators=300,      # 나무 개수
        max_depth=None,        # 깊이 제한 없음(과적합 시 줄이기)
        random_state=42,
        n_jobs=-1              # CPU 코어 모두 사용
    ), {}))

    # 3) XGBoost (있을 때만)
    if XGBRegressor is not None:
        models.append(("XGBoost", XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",  # 빠른 히스토그램 분할
            n_jobs=-1
        ), {"verbose": False}))  # fit에 넣을 추가 인자 예시

    # 4) LightGBM (있을 때만)
    if LGBMRegressor is not None:
        models.append(("LightGBM", LGBMRegressor(
            n_estimators=600,
            max_depth=-1,         # 자동
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ), {}))

    return models


# ------------------------------------------------------------
# 3) 시계열 분할: 앞부분(학습) / 뒷부분(검증)
# ------------------------------------------------------------
def time_split(X, y, valid_ratio=0.2):
    """
    시간 순서를 지키기 위해, 앞쪽은 '학습', 뒤쪽은 '검증'으로 나눠요.
    (시계열은 랜덤 섞기를 안 하는 게 일반적)

    valid_ratio=0.2 이면 데이터의 20%를 검증용으로 사용.
    """
    n = len(X)
    v = max(1, int(n * valid_ratio))  # 검증 샘플 개수(최소 1)
    t = n - v                         # 학습 샘플 개수
    return (X[:t], y[:t], X[t:], y[t:])


# ------------------------------------------------------------
# 4) 간단한 앙상블: 여러 모델 예측을 '가중 평균'
# ------------------------------------------------------------
class SimpleEnsemble:
    """
    여러 모델의 예측을 섞어서 하나로 만드는 간단한 앙상블.
    - weights: 가중치(값이 크면 그 모델을 더 신뢰한다는 뜻)
    - 여기서는 모델별 검증 RMSE 의 역수를 가중치로 사용(좋을수록 큰 가중)
    """
    def __init__(self, models, weights):
        self.models = models
        # 가중치 합이 1이 되도록 정규화(합이 0이면 분모를 아주 작은 값으로)
        self.weights = np.array(weights, dtype=float) / max(np.sum(weights), 1e-9)

    def predict(self, X):
        # 각 모델의 예측을 모아서(열방향) 가중 평균
        preds = [m.predict(X) for m in self.models]              # 리스트 길이 = 모델 수
        return np.sum(np.array(preds).T * self.weights, axis=1)  # (샘플, 모델) · (모델,) → (샘플,)


# ------------------------------------------------------------
# 5) Optuna 로 하이퍼파라미터 튜닝(선택)
# ------------------------------------------------------------
def _tune_with_optuna(name, base_model, X_tr, y_tr, X_va, y_va, n_trials=20):
    """
    특정 모델에 대해 Optuna 로 '좋은 하이퍼파라미터'를 찾아요.
    - name: 모델명 문자열 (RandomForest/XGBoost/LightGBM)
    - base_model: 원래 모델(대체로 무시하고 새로 만듦)
    - X_tr, y_tr: 학습 세트
    - X_va, y_va: 검증 세트
    - n_trials: 시도 횟수(많을수록 더 꼼꼼하지만 시간이 오래 걸림)

    반환:
      - 튜닝이 가능하면 '최적 모델' 객체를 반환
      - Optuna가 없거나 모델이 매칭되지 않으면 None
    """
    if optuna is None:
        return None  # Optuna 설치 안 되어 있으면 스킵

    # 탐색 목표 함수: 검증 RMSE 를 최소화
    def objective(trial):
        if name == "RandomForest":
            # 탐색 범위 정의(대략적인 합리적 구간)
            n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
            max_depth    = trial.suggest_int("max_depth", 6, 24, step=2)
            m = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

        elif name == "XGBoost" and XGBRegressor is not None:
            n_estimators = trial.suggest_int("n_estimators", 300, 900, step=100)
            max_depth    = trial.suggest_int("max_depth", 4, 10)
            lr           = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
            subsample    = trial.suggest_float("subsample", 0.7, 1.0)
            colsample    = trial.suggest_float("colsample_bytree", 0.7, 1.0)
            lam          = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            m = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                reg_lambda=lam,
                random_state=42,
                tree_method="hist",
                n_jobs=-1
            )

        elif name == "LightGBM" and LGBMRegressor is not None:
            n_estimators = trial.suggest_int("n_estimators", 400, 1400, step=200)
            lr           = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
            num_leaves   = trial.suggest_int("num_leaves", 31, 255, step=16)
            subsample    = trial.suggest_float("subsample", 0.7, 1.0)
            colsample    = trial.suggest_float("colsample_bytree", 0.7, 1.0)
            m = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=lr,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample,
                random_state=42,
                n_jobs=-1
            )
        else:
            # 이 함수가 지원하지 않는 모델이면 큰 숫자(나쁜 점수) 반환
            return 1e9

        # 학습 후 검증세트 예측 → RMSE 반환
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        return rmse(y_va, p)

    # Optuna 실행(최소화)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # 최적 파라미터로 '다시' 모델을 만들어 학습해 반환
    best_params = study.best_params
    if name == "RandomForest":
        m = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=42,
            n_jobs=-1
        )
    elif name == "XGBoost" and XGBRegressor is not None:
        m = XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            reg_lambda=best_params["reg_lambda"],
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        )
    elif name == "LightGBM" and LGBMRegressor is not None:
        m = LGBMRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=42,
            n_jobs=-1
        )
    else:
        return None

    # 최적 모델은 다시 전체 학습세트에 맞춰서 반환
    m.fit(X_tr, y_tr)
    return m


# ------------------------------------------------------------
# 6) 학습 & 성능 비교 → 베스트 모델 선택
# ------------------------------------------------------------
def train_and_score(X, y, valid_ratio=0.2, use_optuna=False, optuna_trials=15, build_ensemble=True):
    """
    여러 모델을 학습시키고, 검증 성능(RMSE/MAE/MAPE)을 비교해
    '가장 좋은 모델'을 찾아 반환해요.

    입력:
      - X, y: 학습 데이터(배열/넘파이)
      - valid_ratio: 검증 비율(0.2 = 20%)
      - use_optuna: True면 모델별 튜닝 시도
      - optuna_trials: 튜닝 시도 횟수
      - build_ensemble: True면 간단 앙상블도 후보로 추가

    반환:
      - best_model: 가장 성능 좋은 모델(단일 또는 Ensemble)
      - lb: 성능 리더보드(DataFrame, rmse 오름차순 정렬)
    """
    # 시간 순서 기반 분할(앞: 학습, 뒤: 검증)
    X_tr, y_tr, X_va, y_va = time_split(X, y, valid_ratio=valid_ratio)

    rows = []            # 각 모델의 성적표를 담을 리스트(나중에 DataFrame으로)
    best = (None, None, float("inf"))  # (이름, 모델, 현재까지의 최소 RMSE)
    fitted = []          # 학습 완료된 (이름, 모델) 저장
    va_preds = []        # 검증 예측 결과(앙상블 만들 때 사용)

    # 모델 후보들을 하나씩 학습/평가
    for name, mdl, fit_params in get_candidates():
        try:
            # Optuna 튜닝을 켜면 먼저 튜닝을 시도
            if use_optuna:
                tuned = _tune_with_optuna(name, mdl, X_tr, y_tr, X_va, y_va, n_trials=optuna_trials)
                if tuned is not None:
                    mdl = tuned  # 튜닝 성공 시 그 모델로 교체

            # 모델 학습
            mdl.fit(X_tr, y_tr, **fit_params)

            # 검증 예측
            pred = mdl.predict(X_va)

            # 성적표 한 줄 작성
            row = {
                "model": name,
                "rmse": rmse(y_va, pred),
                "mae":  mae(y_va, pred),
                "mape": mape(y_va, pred)
            }
            rows.append(row)

            # 앙상블 후보를 위해 저장
            fitted.append((name, mdl))
            va_preds.append(pred)

            # 베스트 갱신(더 작은 RMSE가 나오면 교체)
            if row["rmse"] < best[2]:
                best = (name, mdl, row["rmse"])

        except Exception:
            # 어떤 모델이 실패하더라도 전체 파이프라인은 계속 가요.
            rows.append({"model": name, "rmse": np.nan, "mae": np.nan, "mape": np.nan})

    # ---- 간단 앙상블 후보 추가 (원하면) ----
    # 2개 이상 모델이 성공했을 때만 앙상블 시도
    if build_ensemble and len(va_preds) >= 2:
        # 모델별 RMSE의 역수를 가중치로 사용(좋을수록 큰 가중)
        rmses = [rmse(y_va, p) for p in va_preds]
        weights = [1.0 / max(r, 1e-6) for r in rmses]  # 0 나눔 방지

        ens = SimpleEnsemble([m for _, m in fitted], weights)
        ens_pred = ens.predict(X_va)

        row = {
            "model": "Ensemble",
            "rmse": rmse(y_va, ens_pred),
            "mae":  mae(y_va, ens_pred),
            "mape": mape(y_va, ens_pred)
        }
        rows.append(row)

        # 앙상블이 제일 좋으면 베스트로 교체
        if row["rmse"] < best[2]:
            best = ("Ensemble", ens, row["rmse"])

    # 리더보드 테이블 만들기(작은 rmse 순)
    lb = pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)

    # best[1] = 베스트 모델 객체
    return best[1], lb


# ------------------------------------------------------------
# 7) 산출물 저장(베스트 모델/피처명/매핑/리더보드)
# ------------------------------------------------------------
def save_artifacts(out_dirs, best_model, feature_names, mapping, leaderboard_df):
    """
    학습 결과를 디스크에 저장해요.

    - out_dirs: 저장할 폴더 목록(예: ['artifacts', 'models'])
      두 폴더 모두에 동일한 파일을 만들어 둡니다(복구/공유 편의).
    - best_model: train_and_score 에서 뽑힌 최고 모델(또는 앙상블)
    - feature_names: 모델 입력 컬럼 이름 리스트
    - mapping: 날짜/타깃/카테고리 매핑 딕셔너리 (재현/예측 시 필요)
    - leaderboard_df: 성능 표(DataFrame)

    생성 파일:
      - best_model.pkl: {model, feature_names, mapping} 를 pickle 로 저장
      - leaderboard.csv: 성능 표 (UTF-8-SIG, 엑셀 호환)
      - leaderboard.parquet: 파케이(있으면)
    """
    payload = {
        "model": best_model,
        "feature_names": feature_names,
        "mapping": mapping
    }

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

        # 1) 베스트 모델 패키지 저장
        with open(os.path.join(d, "best_model.pkl"), "wb"):
            # pickle.dump: 파이썬 객체를 파일로 직렬화해서 저장
            pass
        with open(os.path.join(d, "best_model.pkl"), "wb") as f:
            pickle.dump(payload, f)

        # 2) 리더보드 저장 (CSV)
        leaderboard_df.to_csv(
            os.path.join(d, "leaderboard.csv"),
            index=False,
            encoding="utf-8-sig"  # 엑셀에서 한글 안깨지도록
        )

        # 3) 리더보드 저장 (Parquet, 선택사항)
        try:
            leaderboard_df.to_parquet(
                os.path.join(d, "leaderboard.parquet"),
                index=False
            )
        except Exception:
            # pyarrow 같은 의존성이 없을 수 있으니 실패해도 그냥 넘어감
            pass
