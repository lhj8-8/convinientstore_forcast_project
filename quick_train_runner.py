#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
자동 학습 런처 (train_cli.py 예시)
------------------------------------------------------------
이 스크립트는 CSV를 읽어 자동으로 컬럼 매핑 → 피처 생성 →
모델 후보 학습(옵션: Optuna 튜닝) → 아티팩트/모델 저장을
한 번에 수행합니다.

[사용 예]
python train_cli.py --data ./data/sample_sales.csv \
                    --project . \
                    --valid_ratio 0.2 \
                    --use_optuna --optuna_trials 20

필수:
  --data         학습에 사용할 CSV 파일 경로

선택:
  --project      작업 루트 폴더(기본: 현재 폴더 ".")
  --valid_ratio  검증 비율(0.05~0.4 권장, 기본 0.2)
  --use_optuna   Optuna 튜닝 사용 플래그(지정 시 on)
  --optuna_trials Optuna 시도 횟수(기본 15)

출력:
  프로젝트 폴더 아래에
    artifacts/   (로그/리더보드 등 중간 산출물)
    models/      (best_model.pkl 등 모델 파일)
이 생성됩니다.
============================================================
"""

import os
import argparse
import pandas as pd  # (필요하면 추후 사용, 지금은 임포트만)

from utils_io import read_csv_flexible, save_utf8sig, ensure_dirs, auto_map_columns
from preprocess import make_matrix
from train_core import train_and_score, save_artifacts


def main():
    """
    커맨드라인 인자를 파싱해서:
      1) CSV 로드
      2) 자동 컬럼 매핑
      3) 학습용 데이터셋(X, y) 구성
      4) 모델 학습(+옵션: Optuna 튜닝)
      5) 결과 저장(artifacts/, models/)
    를 순차 실행합니다.
    """
    # --------------------------------------------------------
    # 1) 커맨드라인 옵션 정의/파싱
    # --------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="학습에 사용할 CSV 경로 (예: ./data/sales.csv)")
    ap.add_argument("--project", default=".", help="작업 루트 폴더(artifacts/models 생성 위치). 기본값='.'")
    ap.add_argument("--valid_ratio", type=float, default=0.2, help="검증 데이터 비율(기본 0.2)")
    ap.add_argument("--use_optuna", action="store_true", help="Optuna 튜닝 사용 여부(플래그 지정 시 사용)")
    ap.add_argument("--optuna_trials", type=int, default=15, help="Optuna 시도 횟수(기본 15)")
    args = ap.parse_args()

    # --------------------------------------------------------
    # 2) 작업 루트 이동 (상대 경로 혼동 방지)
    # --------------------------------------------------------
    proj = os.path.abspath(args.project)  # 절대경로로 변환
    os.chdir(proj)                        # 여길 기준으로 파일 읽고/저장

    # --------------------------------------------------------
    # 3) CSV 로드 + 컬럼 자동 매핑
    # --------------------------------------------------------
    data = read_csv_flexible(args.data)
    mapping = auto_map_columns(data)

    # --------------------------------------------------------
    # 4) 피처 구성(X, y, feat_names 생성)
    # --------------------------------------------------------
    df, X, y, feat_names = make_matrix(data, mapping)

    # --------------------------------------------------------
    # 5) 출력 폴더 준비 (없으면 생성)
    # --------------------------------------------------------
    artifacts = os.path.join(proj, "artifacts")  # 리더보드/로그 등
    models_dir = os.path.join(proj, "models")    # best_model.pkl 저장 위치
    ensure_dirs(artifacts, models_dir)

    # --------------------------------------------------------
    # 6) 모델 학습(+옵션: Optuna) & 리더보드 획득
    # --------------------------------------------------------
    best_model, lb = train_and_score(
        X, y,
        valid_ratio=args.valid_ratio,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials
    )

    # --------------------------------------------------------
    # 7) 산출물 저장 (모델/메타데이터/리더보드)
    # --------------------------------------------------------
    save_artifacts([artifacts, models_dir], best_model, feat_names, mapping, lb)

    # --------------------------------------------------------
    # 8) 콘솔 로그(요약)
    # --------------------------------------------------------
    print("✅ training done.")
    print(" - artifacts:", artifacts)
    print(" - models   :", models_dir)
    try:
        print(lb.head())
    except Exception:
        print(lb)


if __name__ == "__main__":
    main()
