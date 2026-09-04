from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

__all__ = [
    "vector",
    "dot",
    "norm",
    "angle_between",
    "normalize",
    "project",
    "reject",
    "skew",
    "cross",
    "plane_normal",
    "row_echelon",
    "rank",
    "det"
]

def vector(x) -> np.array:
    """입력한 숫자을 벡터로 인식"""
    
    v = np.array(x, dtype=float)                    
    # 입력한 x를 float 타입의 NumPy 배열로 바꾸는 명령어

    if v.ndim != 1:
        raise ValueError("백터는 1차원이어야 합니다.")
    if v.size == 0:
        raise ValueError("빈 벡터는 사용할 수 없습니다.")
    return v

def dot(a,b) -> float:
    """같은 차원인 두 벡터의 곱을 구한다."""
    a, b = vector(a),vector(b) 
    if a.shape != b.shape:
        raise ValueError(
            f"{a}와 {b}의 차원이 다릅니다: "
            f"{a}는 {a.shape}차원이고, {b}는 {b.shape}차원이므로 "
            f"연산이 이루어질 수 없습니다. "
            f"{a}와 {b}가 서로 같은 차원이 되도록 수정하시오."
        )
    return float(np.sum(a*b))
    # 스칼라를 return


def norm(v) -> float:
    """벡터의 크기(길이)를 구한다"""
    v = vector(v)
    return float(np.sqrt(dot(v,v)))


def angle_between(a,b, degrees=True) -> float:
    """두 벡터 사이의 각도를 라디안으로 구한다."""
    a, b = vector(a), vector(b)
    if a.shape != b.shape:
        raise ValueError(
            f"{a}와 {b}의 차원이 다릅니다: "
            f"{a}는 {a.shape}차원이고, {b}는 {b.shape}차원이므로 "
            f"연산이 이루어질 수 없습니다. "
            f"{a}와 {b}가 서로 같은 차원이 되도록 수정하시오."
        )
    if norm(a) == 0 or norm(b) == 0:
        raise ValueError("영벡터는 정규화 할 수 없습니다.")
    return float(np.degrees(np.arccos(dot(a, b) / (norm(a) * norm(b)))))


def normalize(a) -> np.ndarray:
    """두 벡터의 정규화"""
    v = vector(a)
    if norm(a) == 0 or norm(a) <= 1e-12:
        raise ValueError("영에 가까운 영벡터 ᄄᆃ는 영벡터와의 각도는 정의할 수 없습니다.")
    return (v / norm(v))

    
def project(a,b) -> np.ndarray:
    """벡터 a를 벡터 b 위에 정사영 한다"""
    a, b = vector(a), vector(a)
    if a.shape != b.shape:
        raise ValueError(
            f"{a}와 {b}의 차원이 다릅니다: "
            f"{a}는 {a.shape}차원이고, {b}는 {b.shape}차원이므로 "
            f"연산이 이루어질 수 없습니다. "
            f"{a}와 {b}가 서로 같은 차원이 되도록 수정하시오."
        )
    if norm(b) == 0:
        raise ValueError("영벡터 위에는 정사영 할 수 없습니다.")
    return dot(a,b)/dot(b,b)*b


def reject(a,b) -> np.ndarray:
    """벡터 a에서 b방향의 정사영 성분을 제거한다."""
    a, b = vector(a), vector(b)
    if a.shape != b.shape:
        raise ValueError(
            f"{a}와 {b}의 차원이 다릅니다: "
            f"{a}는 {a.shape}차원이고, {b}는 {b.shape}차원이므로 "
            f"연산이 이루어질 수 없습니다. "
            f"{a}와 {b}가 서로 같은 차원이 되도록 수정하시오."
        )
    if norm(b) == 0:
        raise ValueError("영벡터 위에는 정사영 할 수 없습니다.")
    p = project(a,b)
    return a - p


def skew(a) -> np.ndarray:
    "반대칭행렬로 외적을 구한다"
    v = vector(a)
    if v.size != 3:
        raise ValueError("skew는 3차원 벡터에 대해서만 정의됩니다.")
    x,y,z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=float)


def cross(a, b) -> np.ndarray:
    a,b = vector(a), vector(b)
    if a.size != 3 or b.size !=3:
        raise ValueError("외적은 3차원 벡터에서만 정의됩니다.")
    return skew(a) @ b
    

def plane_normal(a,b,c) -> np.ndarray:
    """두 벡터가 만드는 평면의 단위 법선벡터를 구한다."""
    P1 = vector(a)
    P2 = vector(b)
    P3 = vector(c)

    u = P2 - P1
    v = P3 - P1

    n = cross(u,v)

    if norm(n) == 0:
        raise ValueError("두 벡터가 평행하므로 평면의 법선 벡터를 정할 수 없습니다.")

    return normalize(n)


def row_echelon(M) -> np.ndarray:
    q = np.array(M, dtype=float)

    pivots = []
    swaps = 0
    for i in range(3):
        if q[i,i] == 0:
            for r in range(i + 1, 3):
                if q[r,i] !=0:
                    q[[i,r]] = q[[r,i]]
                    swaps += 1
                    break

        if q[i,i] == 0:
            continue
        
        pivots.append(i)
        
        for row in range(i + 1, 3):
            if q[row, i] != 0:
                k = q[row, i] / q[i,i]
                q[row] = q[row] - k * q[i]
    
    return q, pivots, swaps


def rank(M):
    U, pivots, swaps = row_echelon(M)
    return len(pivots)


def det(M):
    U, pivots, swaps = row_echelon(M)
    diagonal = np.diag(U)
    result = np.prod(diagonal)
    return ((-1)**swaps) * result
    




    

    



    
    
    
 

    
                         

