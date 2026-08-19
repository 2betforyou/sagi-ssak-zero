"""Synthetic 계좌 송금 네트워크 생성.

대회에서 별도 데이터가 제공되지 않아, Elliptic / PaySim의 통계적 특성
(정상 거래는 성기고 무작위적, 자금세탁 거래는 짧은 시간에 다단계로
스머핑·사이클 패턴을 형성)을 참고해 자체 synthetic 데이터를 생성한다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import networkx as nx


@dataclass
class Account:
    account_id: str
    label: str  # "normal" | "mule"
    ring_id: str | None = None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_network(
    n_normal: int = 60,
    n_rings: int = 6,
    ring_size_range: tuple[int, int] = (4, 6),
    seed: int = 42,
) -> nx.DiGraph:
    rng = random.Random(seed)
    g = nx.DiGraph()

    normal_ids = [f"N{i:03d}" for i in range(n_normal)]
    for nid in normal_ids:
        g.add_node(nid, label="normal", ring_id=None)

    base_time = datetime(2026, 8, 1, 9, 0, 0)

    # 정상 거래: 시간 간격이 크고(수 시간~수일), 상대방이 매번 무작위
    for _ in range(n_normal * 3):
        src, dst = rng.sample(normal_ids, 2)
        ts = base_time + timedelta(
            days=rng.randint(0, 20), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        amount = rng.choice([30000, 50000, 120000, 250000, 500000])
        g.add_edge(src, dst, amount=amount, timestamp=_iso(ts))

    # 대포통장 자금세탁 링: 소스(도박사이트 환전)에서 시작해 짧은 시간에
    # 여러 계좌를 거쳐(레이어링) 최종적으로 인출계좌로 모이거나 순환(사이클)됨
    for r in range(n_rings):
        ring_id = f"R{r+1}"
        size = rng.randint(*ring_size_range)
        ring_ids = [f"M{ring_id}_{i:02d}" for i in range(size)]
        for mid in ring_ids:
            g.add_node(mid, label="mule", ring_id=ring_id)

        start_time = base_time + timedelta(days=rng.randint(0, 20), hours=rng.randint(0, 23))
        big_amount = rng.choice([8_000_000, 15_000_000, 30_000_000, 50_000_000])

        source = f"SRC_{ring_id}"  # 불법도박 환전상 (그래프 외부 소스로 표시)
        g.add_node(source, label="external_source", ring_id=ring_id)

        cur_amount = big_amount
        t = start_time
        prev = source
        for mid in ring_ids:
            t = t + timedelta(minutes=rng.randint(3, 15))  # 초단기 패스스루
            fee_cut = rng.uniform(0.02, 0.05)
            cur_amount = round(cur_amount * (1 - fee_cut), -3)
            g.add_edge(prev, mid, amount=cur_amount, timestamp=_iso(t))
            prev = mid

        # 절반은 순환(사이클)형: 마지막 계좌가 다시 앞쪽 계좌로 송금
        if rng.random() < 0.5 and size >= 3:
            t = t + timedelta(minutes=rng.randint(3, 10))
            cycle_amount = round(cur_amount * rng.uniform(0.3, 0.6), -3)
            g.add_edge(ring_ids[-1], ring_ids[0], amount=cycle_amount, timestamp=_iso(t))

        # 스머핑: 마지막 계좌가 다수의 현금인출용 계좌로 소액 분산
        cashout_n = rng.randint(2, 4)
        for c in range(cashout_n):
            cashout_id = f"CO_{ring_id}_{c}"
            g.add_node(cashout_id, label="mule", ring_id=ring_id)
            t = t + timedelta(minutes=rng.randint(1, 8))
            split_amount = round(cur_amount / cashout_n, -3)
            g.add_edge(ring_ids[-1], cashout_id, amount=split_amount, timestamp=_iso(t))

    return g


if __name__ == "__main__":
    graph = generate_network()
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
