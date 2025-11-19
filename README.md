# cstore_suite_final

편의점 수요예측 & 자동 발주 추천 플랫폼 (Colab 원클릭 실행 버전)

## 실행 순서 (Colab)
1. zip 업로드 후 아래 셀 실행:
    ```python
    %cd /content
    !unzip -o cstore_suite_final.zip -d /content/
    %cd /content/cstore_suite_final/cstore_suite
    ```
2. `cstore_autotrain_suite.ipynb` 열고
   - [1] 설치
   - [2] 환경 진단
   - [3] Streamlit 실행
   순서대로 돌립니다.

## 주요 기능
- CSV 업로드 후 자동 컬럼 매핑 (date/region/brand/item/target)
- 시계열 피처 (lag1/7/14, rolling mean/std 등)
- 모델 후보: LinearRegression, RandomForest, XGBoost, LightGBM
- Optuna 기반 하이퍼파라미터 튜닝 (체크박스)
- RMSE/MAE/MAPE 리더보드 및 상위 모델 앙상블
- 반복 예측(AR식 전가)로 향후 n일 수요 예측
- 안전재고(ROP), MOQ, 포장단위 반영한 발주 추천
- cloudflared 버튼으로 퍼블릭 URL 생성 (ngrok 토큰 불필요)

폴더 구조:
- `data/` 샘플 CSV
- `models/` 학습된 best_model.pkl 저장
- `artifacts/` 리더보드 등 산출물 저장
- `app_streamlit_pro.py` Streamlit 앱 본체
- `cstore_autotrain_suite.ipynb` 단일 런처
