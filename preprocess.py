import pandas as pd, numpy as np

def add_time_features(df, date_col):
    """
    [무엇을 하나요?]
    - 날짜 열(date_col)에서 '연도/월/일/요일/몇 주차/주말 여부' 같은
      쉬운 달력 정보를 뽑아 표에(데이터프레임에) 붙여줘요.

    [왜 필요하죠?]
    - 기계는 '2025-01-15' 같은 날짜 글자를 잘 못 이해해요.
      대신 '2025년', '1월', '15일', '수요일', '3주차' 처럼 숫자 정보가 있으면
      규칙(계절/요일 패턴)을 더 잘 배울 수 있어요.

    [입력]
    - df: 원래 데이터 표 (DataFrame)
    - date_col: 날짜가 들어있는 열 이름 (예: 'date')

    [출력]
    - 달력 정보 열이 추가된 새 표 (원본은 건드리지 않아요)
    """
    df = df.copy()  # 원본을 망가뜨리지 않으려고 복사본을 만들어요.

    # 날짜 글자를 진짜 '날짜'로 바꿔요. 이상한 값은 NaT(비어있음)로 처리.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 날짜가 비어있는 행은 계산이 안 되니 빼고, 날짜순으로 정렬해요.
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # 달력에서 바로 꺼낼 수 있는 정보들을 새 열로 만들어요.
    df["year"]  = df[date_col].dt.year        # 몇 년도인지
    df["month"] = df[date_col].dt.month       # 몇 월인지(1~12)
    df["day"]   = df[date_col].dt.day         # 며칠인지(1~31)
    df["dow"]   = df[date_col].dt.dayofweek   # 요일(월=0 ... 일=6)
    # '몇 주차'는 ISO 달력 기준이에요. 예: 1월의 첫 주가 1이 아니라 52일 수도 있어요.
    df["week"]  = df[date_col].dt.isocalendar().week.astype(int)
    # 토/일이면 주말(1), 아니면 0
    df["is_weekend"] = (df["dow"]>=5).astype(int)

    return df


def add_lag_features(df, date_col, target_col, group_keys=None, lags=(1,7,14), rolls=(7,14)):
    """
    [무엇을 하나요?]
    - '어제/일주일 전/보름 전' 같은 과거 값(=지연값, lag)을 만들어서 붙이고,
      최근 7일/14일의 평균·표준편차(흔들림)도 같이 붙여줘요.

    [왜 필요하죠?]
    - 수요는 어제/지난주와 비슷하게 움직이는 경향이 있어요.
      과거 값을 힌트로 주면 '내일'을 맞추기 쉬워져요.
      - lag7: 7일 전 값 → '지난주 같은 요일'의 힌트
      - rmean7: 최근 7일 평균 → 최근 흐름(평균)
      - rstd7: 최근 7일 흔들림(표준편차) → 변동성 크기

    [group_keys가 뭐죠?]
    - 점포/브랜드/상품마다 따로 과거를 보라고 지정하는 열들이에요.
      예) ["region", "item"]이면 지역+상품별로 각각 어제/지난주를 계산해요.
      (그룹 없이 통으로 계산하면 서로 다른 점포/상품의 값이 섞여서 의미가 흐려질 수 있어요.)

    [입력]
    - df: 표
    - date_col: 날짜 열 이름
    - target_col: 맞추고 싶은 숫자(판매량 등) 열
    - group_keys: 그룹핑할 열 목록(없어도 됨)
    - lags: 만들 lag 목록(기본 1, 7, 14)
    - rolls: 굴리는 창 크기(rolling window) 목록(기본 7, 14)

    [출력]
    - lag/rmean/rstd 열이 추가된 표(날짜순)
    """
    df = df.copy()

    # group_keys 중 표에 실제로 존재하는 것만 남겨요.
    group_keys = [c for c in (group_keys or []) if c in df.columns]

    # 그룹이 있으면 그룹별로, 없으면 전체를 하나의 그룹처럼 처리해요.
    if group_keys:
        g = df.groupby(group_keys, group_keys=False)  # group_keys=False: 키를 인덱스로 올리지 말기
    else:
        g = [(None, df)]  # '그룹이 하나'라고 가정한 리스트. 아래 for문과 호환되게 만들어요.

    out = []  # 그룹별로 처리한 결과를 모아둔 뒤, 마지막에 합쳐요.

    # pandas의 groupby는 (키, 부분표) 형태로 반복됩니다.
    # 위에서 g를 리스트로 맞춰줬기 때문에 둘 모두 같은 방식으로 순회 가능해요.
    for _, part in (g if isinstance(g, list) else g):
        part = part.sort_values(date_col).copy()  # 날짜순으로 정렬

        # (1) lag 열들 만들기: 예) lag1(어제), lag7(지난주), lag14(보름 전)
        for l in lags:
            part[f"lag{l}"] = part[target_col].shift(l)
            # shift(l)은 위에서 l칸 밀어요. 오늘 행에는 'l일 전 값'이 들어감.

        # (2) rolling 평균/표준편차: 최근 w일 평균/흔들림
        for w in rolls:
            # min_periods를 w의 절반 이상(최소 2)으로 줘서
            # 초반부 데이터가 너무 작을 때도 값이 조금이라도 나오도록 배려.
            part[f"rmean{w}"] = part[target_col].rolling(w, min_periods=max(2, w//2)).mean()
            part[f"rstd{w}"]  = part[target_col].rolling(w, min_periods=max(2, w//2)).std()

        out.append(part)

    # 그룹별로 만든 표들을 위아래로 이어붙이고, 다시 날짜순 정렬
    return pd.concat(out, axis=0).sort_values(date_col)


def make_matrix(df, mapping):
    """
    [무엇을 하나요?]
    - 모델 학습용 '입력 X'와 '정답 y'를 만드는 공장입니다.
      1) 날짜/타깃 열 이름을 mapping에서 읽고,
      2) add_time_features / add_lag_features로 숫자 힌트를 추가하고,
      3) (있다면) region/brand/item을 '원-핫 인코딩(가짜 열)'으로 바꿔서 X에 붙여요.
      4) y는 타깃 값(판매량 등)으로 설정해요.

    [입력]
    - df: 원본 표
    - mapping: {'date':..., 'target':..., 'region':..., 'brand':..., 'item':...}
               (region/brand/item은 없어도 됨)

    [출력]
    - df: 피처가 붙은 표(초기 lag로 NaN인 맨 앞부분은 제거됨)
    - X: 모델에 들어갈 숫자 배열(2차원)
    - y: 정답 벡터(1차원)
    - feat_names: X의 열 이름 목록(모델 해석/재현에 필요)
    """
    df = df.copy()

    # 매핑에서 열 이름 꺼내오기
    date_col   = mapping.get("date")
    target_col = mapping.get("target")
    region_col = mapping.get("region")
    brand_col  = mapping.get("brand")
    item_col   = mapping.get("item")

    # 날짜/타깃은 필수! 없으면 진행 못 해요.
    if not date_col or not target_col:
        raise ValueError("date/target 컬럼 매핑이 필요합니다.")

    # --- (1) 숫자형 정리 ---
    # 타깃은 반드시 숫자여야 해요. 글자가 섞여 있으면 NaN으로 바뀜 → 0으로 채움.
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # (선택) 분류형 열들은 글자(문자열)로 통일해요.
    # 이렇게 해야 '원-핫 인코딩'이 잘 됩니다.
    if region_col and region_col in df: df[region_col] = df[region_col].astype(str)
    if brand_col  and brand_col  in df: df[brand_col]  = df[brand_col].astype(str)
    if item_col   and item_col   in df: df[item_col]   = df[item_col].astype(str)

    # --- (2) 달력 피처 붙이기 ---
    df = add_time_features(df, date_col)

    # --- (3) 과거/최근 통계 피처 붙이기 ---
    # 그룹키: 존재하는 것만 사용 (예: ['region','brand','item'] 중 실제 있는 열만)
    df = add_lag_features(
        df, date_col, target_col,
        [c for c in [region_col, brand_col, item_col] if c]
    )

    # --- (4) lag/rolling 때문에 앞부분에 생긴 비어있는 행 제거 ---
    # 첫 몇 행은 lag1/lag7 같은 게 채울 수 없어서 NaN이 돼요 → 학습에 못 쓰니 제거.
    drop_cols = [c for c in df.columns if c.startswith("lag") or c.startswith("rmean") or c.startswith("rstd")]
    df = df.dropna(subset=drop_cols)

    # --- (5) 숫자 피처 목록 만들기 ---
    # 달력 숫자 + lag/rolling 숫자들을 모아서 X의 기본 뼈대를 만들어요.
    num_cols = ["year","month","day","dow","week","is_weekend"] + drop_cols
    num_cols = [c for c in num_cols if c in df.columns]  # 혹시 빠진 게 있으면 걸러줌

    # 숫자 피처를 먼저 행렬로 변환
    X_num = df[num_cols].values
    feat_names = list(num_cols)  # 나중에 해석/재현할 때 필요

    # --- (6) 분류형(문자) → 원-핫 인코딩 ---
    # 예: region이 '서울','경기'면 'region_서울','region_경기' 같은 가짜 열을 만들어요(0/1)
    cat_cols = [c for c in [region_col, brand_col, item_col] if c and c in df.columns]
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols].astype(str), dummy_na=False)
        # 숫자 피처(X_num) 오른쪽에 원-핫 피처를 붙여요.
        X = np.hstack([X_num, dummies.values])
        feat_names += list(dummies.columns)  # 새로 생긴 열 이름도 기록
    else:
        X = X_num  # 분류형이 없으면 숫자만 사용

    # --- (7) 정답 y 만들기 ---
    y = df[target_col].values  # 우리가 맞추고 싶은 값(예: 판매량)

    return df, X, y, feat_names
