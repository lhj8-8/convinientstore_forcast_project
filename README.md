# 편의점 수요예측 & 발주 프로젝트

> 편의점/리테일 매장의 **수요예측 + 발주 추천**을 한 번에 처리하는 Pro 버전 대시보드 & 학습 파이프라인  
> (Streamlit 앱 + 자동 학습 스크립트 + 피처 엔지니어링 + 모델/리더보드 저장)

---
##  목차
- [프로젝트 개요](#-개요)
- [팀원 소개](#-팀원-소개)
- [주요 특징](#-주요-특징)
- [폴더 및 파일 구조](#-폴더-및-파일-구조)
- [사용 기술 스택](#사용-기술-스택)
- [프로젝트 설계, 구현](#프로젝트-설계-구현)
- [주요기능 실행 화면](#주요기능-실행-화면)
- [설치 방법](#설치-방법)
- [향후 개선 아이디어](#향후-개선-아이디어)
---

## 프로젝트 개요

- 프로젝트 목표 : 데이터 기반 발주 자동화를 목표로 한 4인 팀 프로젝트입니다.
- 개발 기간 : 25/11/02 ~ 25/12/02  


이 프로젝트는 편의점/리테일 매장의 판매 데이터(CSV)를 기반으로

- **수요예측(일 단위)**  
- **재고를 고려한 발주량 추천**  
- **매출 금액 예측**  
- **우산/군고구마 ↔ 날씨 데이터 상관 분석**  

구성 요소는 크게 두 가지입니다.

1. `app_streamlit_pro.py`  
   - Streamlit 기반 웹 대시보드  
   - CSV 업로드 → 컬럼 자동 매핑 → 학습 → 예측/발주 → 분석(그래프) → 진단/로그까지 한 화면에서 처리
2. `quick_train_runner.py`  
   - 커맨드라인에서 **자동 학습**을 실행하는 런처  
   - 배치 학습/서버 환경에서 모델만 먼저 학습시킬 때 사용

이외에 다음과 같은 유틸/핵심 모듈로 구성되어 있습니다.

- `utils_io.py` : CSV 인코딩 유연 로딩, 폴더 생성, 컬럼 자동 매핑
- `preprocess.py` : 날짜/시계열 피처, lag/rolling 통계, 원-핫 인코딩
- `train_core.py` : 모델 후보(Linear, RF, XGBoost, LightGBM), Optuna 튜닝, 간단 앙상블, 리더보드 저장

---

## 팀원 소개
<div align="center">

<table>
  <tr>
    <!-- PM / 데이터 기획 -->
    <td align="center" width="230" style="vertical-align: top;">
      <b>진민경 (PM)</b>
      <div style="width:60%;margin:6px auto;border-bottom:1px solid #aaa;"></div>
      <sub><b>PM / 데이터 기획</b></sub><br>
      <sub>프로젝트 전체 일정·업무 관리,<br>요구사항 정의 및 화면 설계,및 데이터 전처리 총괄<br>발표 자료 및 문서화 총괄</sub><br><br>
      <a href="https://github.com/wjsalsrud02-lang">
        <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
    <!-- MLOps / 인프라 -->
    <td align="center" width="230" style="vertical-align: top;">
      <b>이다미</b>
      <div style="width:60%;margin:6px auto;border-bottom:1px solid #aaa;"></div>
      <sub><b>MLOps / 인프라</b></sub><br>
      <sub>Colab 실행 환경 및 의존성 관리,<br>
      데이터 전처리 보조<br>
      모델/아티팩트 저장 구조 설계</sub><br><br>
      <a href="https://github.com/2dam2">
        <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
    <!-- ML / 분석 & 시각화 -->
    <td align="center" width="230" style="vertical-align: top;">
      <b>이혜지</b>
      <div style="width:60%;margin:6px auto;border-bottom:1px solid #aaa;"></div>
      <sub><b>ML / 분석 & 시각화</b></sub><br>
      <sub>피처 엔지니어링(preprocess) 설계,<br>
      모델 성능 분석 및 리더보드 해석,<br>
      우산·군고구마 상관 분석 그래프 구현</sub><br><br>
      <a href="https://github.com/lhj8-8">
        <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
    <!-- Full Stack / Streamlit -->
    <td align="center" width="230" style="vertical-align: top;">
      <b>박종훈</b>
      <div style="width:60%;margin:6px auto;border-bottom:1px solid #aaa;"></div>
      <sub><b>Full Stack / Streamlit</b></sub><br>
      <sub>메인 대시보드 개발,<br>
      예측·발주 로직 및 세그먼트 구현,<br>
      ngrok 연동 및 UI/UX 개선</sub><br><br>
      <a href="https://github.com/dailyhune">
        <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
  </tr>
</table>

</div>


---

## 주요 특징

- **여러 CSV를 한 번에 업로드/선택 → 자동 결합**
  - 업로드/선택 시 `source` 열(파일명)을 자동 추가해서 원본을 추적할 수 있음
- **컬럼 자동 매핑**
  - `auto_map_columns()`를 통해 `date / target / region / brand / item` 역할 자동 추정
  - 한국어/영어 컬럼명을 모두 고려 (예: `날짜`, `일자`, `sales`, `판매수량`, `지점`, `brand`, `상품명` 등)
- **학습 파이프라인 내장**
  - 시간 피처(연/월/일/요일/주차/주말) + lag(1, 7, 14) + rolling 평균/표준편차(7, 14) 자동 생성
  - 분류형 변수(`region`, `brand`, `item`)는 원-핫 인코딩
- **모델 후보 및 앙상블**
  - `LinearRegression`, `RandomForestRegressor`
  - 설치되어 있으면 `XGBRegressor`, `LGBMRegressor`도 자동 사용
  - 검증 RMSE 기반 가중 평균 **SimpleEnsemble** 후보까지 포함
- **Optuna 하이퍼파라미터 튜닝(선택)**
  - RandomForest / XGBoost / LightGBM에 대해 Optuna로 n_estimators, depth, learning_rate 등 탐색
- **예측 & 발주 추천**
  - 14일 고정 **반복(autoregressive) 예측**
  - 재고 컬럼(재고, stock…)을 자동 탐지하여 **재고 소진 예상일수 / 권장 발주량 계산**
  - 가격 컬럼(가격, 단가, price…) 자동 검색, 없으면 직접 입력 → **예상 매출 금액 계산**
- **상품/세그먼트별 분석 그래프**
  - **우산**: 월별 강수량 ↔ 우산 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
  - **군고구마**: 월별 기온 ↔ 군고구마 판매량 (산점도 + 회귀선 + 일별 선형 그래프)
  - **전체**: 우산/군고구마 제외 전체 상품의 일별 총 판매량 선형 그래프
- **Colab/로컬 환경 모두 고려**
  - `__file__`이 없는 환경(예: Colab) 방어
  - `ngrok`/`cloudflared`를 이용한 퍼블릭 URL 열기 기능

---

## 폴더 및 파일 구조

예시 구조는 다음과 같습니다.

```bash
project_root/
├─ app_streamlit_pro.py        # 메인 Streamlit 앱
├─ quick_train_runner.py       # 커맨드라인 자동 학습 런처
├─ utils_io.py                 # CSV IO + 컬럼 자동 매핑
├─ preprocess.py               # 피처 엔지니어링(시간/lag/rolling/원-핫)
├─ train_core.py               # 모델 학습/튜닝/앙상블/리더보드 저장
├─ requirements.txt            # Python 의존성 목록 (예시 아래 참고)
├─ data/
│   ├─ sample_sales.csv        # 예시 판매 데이터
│   ├─ usan.csv                # 우산/강수량 데이터 (예시)
│   └─ gungoguma.csv           # 군고구마/기온 데이터 (예시)
├─ artifacts/                  # 학습 리더보드, 로그 등 (자동 생성)
└─ models/                     # best_model.pkl 등 모델 파일 (자동 생성)
```

---

## 사용 기술 스택

- Python 3.10+
- 데이터 처리: `pandas`, `numpy`
- 웹 대시보드: `streamlit`, `altair`
- 머신러닝:
  - `scikit-learn` (LinearRegression, RandomForestRegressor 등)
  - 선택: `xgboost`, `lightgbm`
- 하이퍼파라미터 튜닝: `optuna` (선택)
- 터널링(옵션): `pyngrok`

---

## 프로젝트 설계, 구현
> PPT나 ERD, 플로우차트 이미지 등을 여기에 삽입할 예정  

----------------------------------------------------------

---

## 주요기능 실행 화면

### 미구현

### 미구현



---
## 설치 방법

### 1) 가상환경 생성(선택)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scriptsctivate
```

### 2) 의존성 설치

`requirements.txt` 예시는 아래와 같이 구성할 수 있습니다.

```txt
pandas==2.2.2
numpy==1.26.4
streamlit>=1.36.0
altair>=5,<6
scikit-learn>=1.4.0

# 선택(설치되어 있으면 추가 모델 사용)
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.0.0

# 퍼블릭 URL 옵션
pyngrok>=7.0.0
cloudflared
```

설치:

```bash
pip install -r requirements.txt
```

---

## 실행 방법

### 방법 A. Streamlit 앱으로 실행

```bash
streamlit run app_streamlit_pro.py
```

기본 포트는 8501입니다.

브라우저에서:

- 로컬: `http://localhost:8501`
- Colab/원격: 아래의 **퍼블릭 URL 기능(ngrok/cloudflared)** 사용

### 방법 B. CLI 자동 학습만 따로 실행

```bash
python quick_train_runner.py   --data ./data/sample_sales.csv   --project .   --valid_ratio 0.2   --use_optuna   --optuna_trials 20
```

실행 결과:

- `artifacts/` 에 리더보드 CSV/Parquet 등
- `models/` 에 `best_model.pkl` 저장  
  → Streamlit 앱에서 예측/발주 탭에서 이 모델을 자동 사용

---

## 데이터 형식 & 컬럼 매핑 규칙

### 기본적으로 필요한 컬럼 역할

`auto_map_columns(df)` 가 다음 역할을 자동으로 추정합니다.

- `date`   : 날짜 (`date`, `일자`, `날짜`, `기준일` 등)
- `target` : 맞출 값 (판매량, 수요량 등)
  - 후보: `qty`, `sales_qty`, `sales`, `판매수량`, `수량`, `demand`, `target`, `y` …
- `region` : 점포/지역 (`지점`, `점포`, `매장`, `지역`, `시도` 등)
- `brand`  : 브랜드/회사명 (`brand`, `브랜드`, `회사`, `제조사` 등)
- `item`   : 상품/품목 (`item`, `상품`, `품목`, `sku`, `상품명`, `제품명` 등)

자동 매핑 로직은

1. 컬럼명을 소문자로 변환  
2. 후보 리스트와 **정확히 일치**하는 컬럼을 우선 매칭  
3. 없으면 **부분 포함(contains)** 으로 완화 탐색  
4. 같은 컬럼이 여러 역할로 중복되면 날짜/타깃을 우선 보존하고,  
   나머지(region/brand/item)는 다른 컬럼으로 대체 시도

를 수행합니다.

> **특수 규칙:**  
> `seoul_gyeonggi_with_demand.csv`, `usan.csv`, `gungoguma.csv` 같이  
> 강수량이 있는 데이터에서 `target`이 `강수량`으로 잘못 잡히는 문제를 방지하기 위해,  
> 데이터에 `일일판매량` 컬럼이 있는 경우 `target`을 강제로 `일일판매량`으로 교체하도록 보정되어 있습니다.

### 재고/가격/날씨 컬럼 자동 인식

예측·발주 탭(③)과 분석 탭(④)에서 아래 컬럼을 자동으로 탐지합니다.

- **재고 컬럼** 후보
  - `"재고", "재고수", "재고수량", "현재재고", "onhand", "on_hand", "stock", "inventory"` 등이름 포함
  - 해당 컬럼의 마지막 값 → 현재 재고(on-hand)로 사용
- **가격 컬럼** 후보
  - `"price", "가격", "단가", "판매가", "amount", "금액"` 포함
  - 마지막 값 × 예측수량 × 정확도(보정 계수) → 금액 예측
- **날씨(강수량/기온) 컬럼** 후보
  - 우산 탭:  
    - 강수량 후보: `"rain", "precip", "precipitation", "강수", "강수량", "일강수량", "강우", "강우량"`
  - 군고구마 탭:  
    - 기온 후보: `"온도", "tmin", "temp_min", "min_temp", "최저", "최저기온", "일최저기온", "temperature", "temp"`

---

## Streamlit 앱 구성 (탭별 상세 설명)

### ① 데이터 탭 — CSV 업로드/선택 + 자동 매핑

- **멀티 업로드**
  - 여러 CSV를 한 번에 업로드(`accept_multiple_files=True`)
  - 체크박스 `파일명(source) 열 추가`가 켜져 있으면 각 행에 `source` 컬럼으로 파일명 기록
  - 업로드한 파일은 `data/` 폴더에 그대로 저장
- **data 폴더에서 선택**
  - `data/` 아래의 CSV 파일 목록을 읽어와 다중 선택
  - 선택한 여러 파일을 한 번에 읽어 concat → `df` 세션 상태에 저장
- **컬럼 자동 매핑**
  - `auto_map_columns(df)` 결과를 세션에 저장 (`st.session_state["mapping"]`)
  - 현재 매핑 결과를 표로 보여줌 (날짜/타깃/지역/브랜드/상품)

### ② 학습/모델 탭 — 모델 학습

- **옵션**
  - `Optuna 하이퍼파라미터 튜닝 사용` 체크박스
  - 튜닝 시도 횟수 슬라이더 (`5 ~ 60`, 기본 15)
  - 검증 비율(`valid_ratio`) 슬라이더 (`0.05 ~ 0.4`, 기본 0.2)
- **학습 흐름**
  1. `make_matrix(df, mapping)` 으로 X, y, feat_names 구성
  2. `train_and_score(X, y, valid_ratio, use_optuna, optuna_trials)` 호출
  3. 여러 모델 후보 학습 + 검증
  4. 최종 베스트 모델/리더보드 반환
  5. `save_artifacts([ARTI_DIR, MODELS_DIR], best_model, feat_names, mapping, lb)` 로 저장
- **리더보드 표시**
  - RMSE/MAE/MAPE 기준 성능 테이블 출력

### ③ 예측·발주 탭 — 반복 예측 + 자동 발주량/매출 계산

- **세그먼트 선택**
  - 매핑된 `region`, `brand`, `item` 컬럼이 있으면 각자 selectbox 제공
  - `<전체>` 옵션을 선택하면 필터링 없이 전체 사용
- **반복(autoregressive) 예측**
  - 14일(`horizon_days = 14`) 고정
  - `lag1`, `lag7`, `lag14`, `rmean7`, `rmean14`, `rstd7`, `rstd14` 등 피처를
    직접 구성해서 **하루씩 미래로 밀면서 반복 예측**
- **정확도(보정계수) 슬라이더**
  - 0.5 ~ 2.0 사이에서 선택
  - 금액 예측 시 수요량 × 가격 × 정확도로 반영
- **재고 자동 인식**
  - 세그먼트 필터링된 df에서 재고 컬럼을 자동 탐지
  - 있으면 마지막 값을 **현재 재고**로 사용 / 없으면 사용자 입력
- **가격 자동 인식 + 금액 예측**
  - 세그먼트 필터링된 df에서 가격 컬럼을 자동 탐지
  - 없으면 직접 숫자 입력
  - `예측수량 × 가격 × 정확도` → `금액예측` 컬럼 생성
- **지표 출력**
  - 예측 기간(일): 14
  - 재고 소진 예상일수
  - 2주 총 예상 매출 (원)
- **결과 테이블**
  - 인덱스: 날짜
  - 컬럼: `예측수량`, `금액예측`

### ④ 분석(그래프) 탭 — 우산/군고구마/전체 분석

3개의 서브 탭으로 구성됩니다.

1. **☔ 우산: 한 달 강수량 vs 판매량**
   - 특정 한 달(YYYY-MM)을 선택
   - 우산 상품만 필터 (item 컬럼에 `우산/umbrella`가 들어가는 행)
   - 일별로
     - 강수량 평균
     - 우산 판매량 합계를 집계
   - **그래프**
     - X축: 강수량, Y축: 우산 판매량 (산점도 + 회귀선)
     - 일별 우산 판매량 **선형(line) 그래프**
2. **군고구마: 한 달 기온 vs 판매량**
   - 군고구마 상품 필터 (`고구마/군고구마/sweet/goguma` 포함)
   - 일별 기온 평균 vs 군고구마 판매량
   - **그래프**
     - X축: 기온, Y축: 군고구마 판매량 (산점도 + 회귀선)
     - 일별 군고구마 판매량 **선형(line) 그래프**
3. **전체: 우산·군고구마 제외 전체 상품 일별 판매량(선형)**
   - item 컬럼에서 우산/군고구마 관련 키워드를 포함하는 행을 제외
   - 일별 `target` 합계를 계산
   - **그래프**
     - X축: 날짜, Y축: 일 판매량 합계 (선형 그래프)

### ⑤ 진단/로그 탭 — 디렉터리 상태 & 퍼블릭 URL

- `data`, `artifacts`, `models` 폴더 경로 및 파일 목록 표시
- 퍼블릭 URL 모드 선택
  - `ngrok` / `cloudflared` 중 선택
- 버튼
  - `퍼블릭 URL 열기`
    - ngrok 모드: `NGROK_AUTHTOKEN` 입력 후 `start_ngrok()` 실행
    - cloudflared 모드: `cloudflared tunnel --url http://localhost:8501`
  - `퍼블릭 URL 닫기`
    - ngrok: `ngrok.kill()` 호출
    - cloudflared: 세션 상태에 저장된 프로세스를 `terminate()`

---

## 내부 모듈 상세

### 1) `utils_io.py`

- `read_csv_flexible(path_or_buf)`
  - `utf-8-sig`, `utf-8`, `cp949`, `euc-kr`, `latin1` 순으로 인코딩을 시도
  - 파일 경로뿐 아니라 `BytesIO`, 업로드된 파일 객체도 지원
- `save_utf8sig(df, path)`
  - 폴더가 없으면 생성 후, UTF-8-SIG로 CSV 저장
- `ensure_dirs(*dirs)`
  - 인자로 받은 경로들을 모두 `os.makedirs(exist_ok=True)`
- `auto_map_columns(df)`
  - 컬럼 후보 리스트(한/영 혼합)를 기반으로 `date/target/region/brand/item` 자동 추정
  - 중복되는 경우 날짜/타깃을 우선으로 하고 나머지는 미사용 컬럼으로 재할당

### 2) `preprocess.py`

- `add_time_features(df, date_col)`
  - 날짜 열에서 `year, month, day, dow, week, is_weekend` 파생
- `add_lag_features(df, date_col, target_col, group_keys=None, lags=(1,7,14), rolls=(7,14))`
  - 그룹별(예: region+item) 날짜 순으로 정렬 후
  - `lag1, lag7, lag14`, `rmean7, rmean14`, `rstd7, rstd14` 생성
- `make_matrix(df, mapping)`
  - 숫자형 정리 → 시간 피처 → lag/rolling → 원-핫 인코딩까지 처리
  - NaN이 있는 초기 구간 제거 후
  - `X`, `y`, `feat_names` 반환

### 3) `train_core.py`

- 평가 지표
  - `rmse`, `mae`, `mape`
- 모델 후보
  - `LinearRegression`
  - `RandomForestRegressor`
  - (선택) `XGBRegressor`
  - (선택) `LGBMRegressor`
- `time_split(X, y, valid_ratio)`
  - 앞 부분은 학습, 뒷 부분은 검증으로 분할
- `SimpleEnsemble(models, weights)`
  - 모델별 검증 RMSE 역수를 가중치로 하는 간단 앙상블
- `_tune_with_optuna(name, base_model, X_tr, y_tr, X_va, y_va, n_trials)`
  - 모델별 하이퍼파라미터 탐색
- `train_and_score(X, y, valid_ratio, use_optuna, optuna_trials, build_ensemble=True)`
  - 모델 학습 + 검증 성능 비교 → 리더보드 생성
  - 필요 시 앙상블까지 후보에 포함 → 최종 best 모델 반환
- `save_artifacts(out_dirs, best_model, feature_names, mapping, leaderboard_df)`
  - 각 폴더에 `best_model.pkl`, `leaderboard.csv`, (가능하면) `leaderboard.parquet` 저장

### 4) `quick_train_runner.py`

- CLI 인자
  - `--data` (필수): 학습에 사용할 CSV 경로
  - `--project` (선택): 작업 루트 폴더 (기본 `.`)
  - `--valid_ratio` (선택): 검증 비율 (기본 0.2)
  - `--use_optuna` (선택): 플래그 지정 시 Optuna 튜닝 사용
  - `--optuna_trials` (선택): Optuna 시도 횟수 (기본 15)
- 동작 순서
  1. `--project` 기준으로 작업 디렉토리 이동
  2. CSV 로드 (`read_csv_flexible`)
  3. 자동 컬럼 매핑 (`auto_map_columns`)
  4. `make_matrix`로 피처 구성
  5. `train_and_score` 호출
  6. `save_artifacts`로 저장

---

## 향후 개선 아이디어

- 프로모션/행사 정보, 날씨(기온/강수량) 등을 포함한 다변량 수요모델
- 점포별/카테고리별 발주 정책(리드타임, 안전재고, MOQ 등) 추가
- Hugging Face Space로 Streamlit 앱 배포 자동화
- 모델 자동 재학습 파이프라인(CI/CD, 스케줄링) 연동
