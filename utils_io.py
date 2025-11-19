#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_io.py — 입출력/컬럼 자동 매핑 유틸 모음 (상세 주석)

이 파일은 다음 기능을 제공합니다.
1) read_csv_flexible: 여러 인코딩 후보로 CSV를 '안전하게' 읽기
2) save_utf8sig     : UTF-8-SIG(엑셀 호환)로 CSV 저장
3) ensure_dirs      : 폴더가 없으면 만들어 주기
4) auto_map_columns : 날짜/타깃/지역/브랜드/상품 컬럼 자동 추정

※ 주의: 아래 auto_map_columns()는 원본 코드의 locals() 기반 충돌 해결을
   '안전한 딕셔너리 기반'으로 고쳤습니다. (Python에서 locals() 수정은
   함수 스코프에서 보장이 되지 않습니다.)
"""

import os
import re
import glob
import pandas as pd
from typing import Optional, Dict, List, Union, IO

# 1) CSV 읽기 시도할 인코딩 후보들
# - utf-8-sig: 엑셀에서 잘 열리는 UTF-8 with BOM
# - utf-8    : 범용
# - cp949/euc-kr: 윈도우/국내 환경에서 자주 쓰는 한글 인코딩
# - latin1   : 마지막 안전망(손실 없이 읽히나 글자가 깨질 수 있음)
ENCODINGS: List[str] = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]


def read_csv_flexible(path_or_buf: Union[str, os.PathLike, IO[bytes], IO[str]]) -> pd.DataFrame:
    """
    여러 인코딩을 순차적으로 시도하여 CSV를 안전하게 읽습니다.
    - 첫 번째로 성공하는 인코딩 결과를 반환합니다.
    - 모두 실패하면 마지막 예외를 다시 던집니다.
    - 문자열 경로뿐 아니라 BytesIO/파일 객체도 지원합니다.

    Parameters
    ----------
    path_or_buf : str 또는 파일 객체
        CSV 파일 경로 또는 파일 객체/버퍼(예: BytesIO, UploadedFile 등)

    Returns
    -------
    pd.DataFrame
        읽어들인 데이터프레임
    """
    last_e: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            # 파일 객체일 경우 매번 처음부터 다시 읽도록 커서 이동
            if hasattr(path_or_buf, "seek"):
                try:
                    path_or_buf.seek(0)
                except Exception:
                    # seek을 지원하지 않으면 그냥 진행
                    pass
            return pd.read_csv(path_or_buf, encoding=enc)
        except Exception as e:
            # 실패하면 다음 인코딩으로 넘어가고, 마지막 예외를 저장
            last_e = e
    if last_e is not None:
        # 모든 인코딩이 실패 → 마지막 에러를 그대로 올림(디버깅에 유용)
        raise last_e
    # 이론상 도달하지 않지만, 안전망으로 한 번 더 시도
    return pd.read_csv(path_or_buf)


def save_utf8sig(df: pd.DataFrame, path: str) -> None:
    """
    DataFrame을 UTF-8-SIG로 저장합니다.
    - 디렉토리가 없으면 먼저 만들어 줍니다.
    - 엑셀에서 한글 깨짐을 방지하는 인코딩입니다.

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터프레임
    path : str
        저장 경로(파일명 포함)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_dirs(*dirs: str) -> None:
    """
    전달된 모든 경로에 대해 폴더가 없으면 생성합니다.
    - 여러 경로를 한 번에 처리할 수 있습니다.

    Example
    -------
    ensure_dirs("data", "artifacts", "models")
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --- Column auto-mapping helpers --------------------------------------------
# 한국어/영어로 자주 쓰이는 열 이름 후보 리스트
_CAND_DATE   = ["date", "일자", "날짜", "dt", "기준일"]
_CAND_TARGET = ["qty", "sales_qty", "sales", "판매수량", "수량", "demand", "target", "y"]
_CAND_REGION = ["region", "지점", "점포", "매장", "지역", "시도", "광역", "구분"]
_CAND_BRAND  = ["brand", "브랜드", "회사", "제조사"]
_CAND_ITEM   = ["item", "상품", "품목", "sku", "상품명", "제품명"]


def _guess_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    """
    컬럼 이름 목록(cols)에서 후보(candidates)와 '가장 잘 맞는' 컬럼을 추정합니다.
    1) 전부 소문자로 바꾼 뒤 '정확히 같은 이름' 우선 매칭
    2) 없으면 '포함(contains)' 매칭으로 완화 탐색

    Parameters
    ----------
    cols : List[str]
        실제 데이터프레임의 컬럼명 리스트
    candidates : List[str]
        우리가 찾고 싶은 의미의 후보명들

    Returns
    -------
    Optional[str]
        매칭된 컬럼명(없으면 None)
    """
    lower = {c.lower(): c for c in cols}  # 소문자 → 원래 컬럼명 매핑

    # (1) 정확 일치 우선
    for c in candidates:
        if c in lower:
            return lower[c]

    # (2) 부분 포함(완화 매칭)
    for c in candidates:
        for col in cols:
            if c in col.lower():
                return col

    return None


def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    날짜/타깃/지역/브랜드/상품 컬럼명을 자동으로 추정합니다.
    - 정/부분일치로 각각 한 개씩 찾습니다.
    - 중복(같은 컬럼이 두 역할로 선택) 발생 시, 날짜/타깃을 우선 보존하고
      나머지(region/brand/item)는 '아직 사용되지 않은' 다른 컬럼으로
      대체 시도합니다. (원본 로직의 locals() 수정 버그를 제거)

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임

    Returns
    -------
    Dict[str, Optional[str]]
        {'date': ..., 'target': ..., 'region': ..., 'brand': ..., 'item': ...}
        값이 None일 수 있습니다.
    """
    cols = list(df.columns)

    # 1) 1차 자동 추정
    date   = _guess_col(cols, _CAND_DATE)
    target = _guess_col(cols, _CAND_TARGET)
    region = _guess_col(cols, _CAND_REGION)
    brand  = _guess_col(cols, _CAND_BRAND)
    item   = _guess_col(cols, _CAND_ITEM)

    # 2) 충돌(중복) 처리 — 안전한 딕셔너리 방식
    picks = {
        "date": date,
        "target": target,
        "region": region,
        "brand": brand,
        "item": item,
    }

    # None이 아닌 값들만 뽑아 중복 여부 확인
    chosen_non_null = [p for p in picks.values() if p]
    has_dup = len(set(chosen_non_null)) != len(chosen_non_null)

    if has_dup:
        # 날짜/타깃 최우선 보호
        used = set([p for p in (date, target) if p])
        # 충돌 가능성이 있는 키들(우선순위 낮음)
        for key in ["region", "brand", "item"]:
            val = picks.get(key)
            # 이미 사용된 컬럼과 겹치면 다른 후보를 찾아봄
            if val and val in used:
                # 아직 쓰지 않은 임의의 컬럼을 순회하며 대체
                replace = None
                for c in cols:
                    if c not in used and c != val:
                        replace = c
                        break
                picks[key] = replace  # 못 찾으면 None이 들어갑니다.
                if replace:
                    used.add(replace)
            elif val:
                used.add(val)

    return picks
