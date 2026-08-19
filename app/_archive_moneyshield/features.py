"""계좌(노드) 단위 그래프 특징 추출."""
from __future__ import annotations

from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

FAST_CYCLE_WINDOW_MINUTES = 120  # 이 시간 내에 완성되는 사이클만 "이상 사이클"로 간주
QUICK_MATCH_WINDOW_MINUTES = 30  # 입금 후 이 시간 내 나간 돈만 "패스스루"로 간주
QUICK_MATCH_AMOUNT_TOLERANCE = 0.10  # 입금액 대비 ±10% 이내 출금만 동일 자금으로 간주


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _fast_cycle_nodes(g: nx.DiGraph) -> set[str]:
    """정상 거래망에서도 우연히 생기는 장기 사이클과, 자금세탁 특유의
    '단시간에 완성되는' 사이클을 구분하기 위해 사이클을 구성하는 엣지들의
    타임스탬프 범위가 FAST_CYCLE_WINDOW_MINUTES 이내인 경우만 위험 신호로 채택한다."""
    try:
        cycles = list(nx.simple_cycles(g, length_bound=6))
    except TypeError:
        cycles = [c for c in nx.simple_cycles(g) if len(c) <= 6]

    fast_nodes: set[str] = set()
    for cycle in cycles:
        edge_times = []
        ok = True
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if not g.has_edge(u, v):
                ok = False
                break
            edge_times.append(_parse(g[u][v]["timestamp"]))
        if not ok or not edge_times:
            continue
        span_minutes = (max(edge_times) - min(edge_times)).total_seconds() / 60
        if span_minutes <= FAST_CYCLE_WINDOW_MINUTES:
            fast_nodes.update(cycle)

    return fast_nodes


def _quick_passthrough(in_edges, out_edges) -> tuple[float, float]:
    """개별 입금 건 각각에 대해, 금액이 비슷하고 짧은 시간 내에 나간 출금 건이
    있는지 확인한다 (전형적인 대포통장 패스스루 신호). 전체 기간을 뭉뚱그려 계산하면
    무관한 거래끼리 우연히 매칭되는 착시가 생기므로 건별로 매칭한다.

    반환: (quick_passthrough_ratio, min_gap_minutes)
    """
    if not in_edges or not out_edges:
        return 0.0, np.nan

    best_ratio = 0.0
    min_gap = np.nan
    for _, _, din in in_edges:
        in_amt = din["amount"]
        in_t = _parse(din["timestamp"])
        for _, _, dout in out_edges:
            out_amt = dout["amount"]
            out_t = _parse(dout["timestamp"])
            gap = (out_t - in_t).total_seconds() / 60
            if gap < 0 or gap > QUICK_MATCH_WINDOW_MINUTES:
                continue
            if out_amt < in_amt * (1 - QUICK_MATCH_AMOUNT_TOLERANCE) * 3:
                # 스머핑(분할 출금) 대비 하한은 널널하게, 상한은 입금액을 넘지 않도록
                if out_amt > in_amt * (1 + QUICK_MATCH_AMOUNT_TOLERANCE):
                    continue
                ratio = min(out_amt / in_amt, 1.0)
                best_ratio = max(best_ratio, ratio)
                min_gap = gap if np.isnan(min_gap) else min(min_gap, gap)

    return best_ratio, min_gap


def extract_features(g: nx.DiGraph) -> pd.DataFrame:
    scoreable_nodes = [n for n in g.nodes if g.nodes[n].get("label") != "external_source"]
    fast_cycle_nodes = _fast_cycle_nodes(g)

    rows = []
    for node in scoreable_nodes:
        in_edges = list(g.in_edges(node, data=True))
        out_edges = list(g.out_edges(node, data=True))

        in_amount = sum(d["amount"] for _, _, d in in_edges)
        out_amount = sum(d["amount"] for _, _, d in out_edges)

        quick_ratio, min_gap = _quick_passthrough(in_edges, out_edges)
        unique_counterparties = len({u for u, _, _ in in_edges} | {v for _, v, _ in out_edges})

        rows.append(
            {
                "account_id": node,
                "label": g.nodes[node].get("label", "normal"),
                "ring_id": g.nodes[node].get("ring_id"),
                "in_degree": len(in_edges),
                "out_degree": len(out_edges),
                "in_amount": in_amount,
                "out_amount": out_amount,
                "passthrough_ratio": quick_ratio,
                "hold_minutes": min_gap,
                "unique_counterparties": unique_counterparties,
                "in_cycle": node in fast_cycle_nodes,
            }
        )

    df = pd.DataFrame(rows)
    # 빠른 패스스루 매칭이 전혀 없었던 계좌(정상 대다수)는 정의상 "보유시간이
    # 매칭 탐색 윈도우보다 길다"는 뜻이다. 매칭된 값들의 중앙값으로 채우면
    # (매칭 자체가 대부분 대포통장에서만 발생하므로) 오히려 작은 값이 되어
    # 정상 계좌를 왜곡시키므로, 탐색 윈도우보다 명확히 큰 고정값으로 채운다.
    no_match_value = QUICK_MATCH_WINDOW_MINUTES * 20
    df["hold_minutes"] = df["hold_minutes"].fillna(no_match_value)
    return df
