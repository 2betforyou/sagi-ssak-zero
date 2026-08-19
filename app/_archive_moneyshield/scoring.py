"""그래프 특징 기반 계좌 위험도 스코어링 (0~100).

주의: 초기 버전은 IsolationForest(비지도 이상탐지)만 사용했으나, 검증 과정에서
거래가 유난히 많은 '정상 허브 계좌'까지 통계적 극단값으로 잡혀 실제 대포통장
패턴(빠른 패스스루·사이클)보다 우선순위가 높게 나오는 문제가 발견되었다.
IsolationForest는 방향성 없이 "분포에서 먼 값"을 이상치로 보기 때문에,
degree가 낮은 게 위험 신호인 대포통장 계좌와 degree가 높은 게 정상인 허브
계좌를 함께 이상치로 취급해버린 것이 원인이다. 그래서 대포통장 특유의
도메인 신호(패스스루 비율·보유시간·사이클 여부) 방향이 뚜렷한 항목만 뽑아
해석 가능한 가중합 규칙으로 스코어링한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import QUICK_MATCH_WINDOW_MINUTES

W_QUICK_PASSTHROUGH = 0.65
W_CYCLE = 0.35


def score_accounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 보유시간이 짧을수록(0에 가까울수록) 1, 매칭 탐색 윈도우 이상이면 0
    speed_factor = np.clip(1 - df["hold_minutes"] / QUICK_MATCH_WINDOW_MINUTES, 0, 1)
    quick_signal = df["passthrough_ratio"] * speed_factor
    cycle_signal = df["in_cycle"].astype(float)

    risk_raw = W_QUICK_PASSTHROUGH * quick_signal + W_CYCLE * cycle_signal
    df["risk_score"] = np.clip(risk_raw * 100, 0, 100).round(1)

    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df
