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
    "det",
    "gauss_eliminate"
    "inverse_gauss_jordan"
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
    q = np.array(M, dtype=float, copy=True)

    if q.ndim != 2:
        raise ValueError("2차원 행렬이어야 합니다.")

    m, n = q.shape
    pivots = []
    swaps = 0
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot_row = row + np.argmax(np.abs(q[row:, col]))

        if abs(q[pivot_row, col]) <= 1e-8:
            continue

        # 행 교환
        if pivot_row != row:
            q[[row, pivot_row]] = q[[pivot_row,row]]
            swaps += 1

        pivots.append(col)
        
        # 피벗 아래 소거
        for r in range(row + 1, m):
            factor = q[r, col] / q[row, col]
            q[r] -= factor * q[row]
            q[r,col] = 0.0

        row += 1
    
    return q, pivots, swaps


def rank(M):
    U, pivots, swaps = row_echelon(M)
    return len(pivots)


def det(M):
    U, pivots, swaps = row_echelon(M)
    diagonal = np.diag(U)
    result = np.prod(diagonal)
    return ((-1)**swaps) * result
    

def inverse_gauss_jordan(A):
    A = np.array(A, dtype=float, copy=True)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("정방행렬만 역행렬을 구할 수 있습니다.")

    n = A.shape[0]
    # [A | I] 형태의 확장행렬 만들기
    augmented = np.hstack([A, np.eye(n)])

    for col in range(n):
        #
        pivot_row = col + np.argmax(np.abs(augmented[col:, col]))

        if augmented[pivot_row, col] == 0:
            raise ValueError("행렬이 특이하여 역행렬을 구할 수 없습니다.")
        
        #피벗 행을 현재 행으로 이동
        augmented[[col, pivot_row]] = augmented[[pivot_row, col]]

        #피벗 행 전체를 나눠서 피벗을 1로 만들기

        pivot = augmented[col, col]
        augmented[col] /= pivot

        for row in range(n):

            if row == col:
                continue

            factor = augmented[row, col] / augmented[col, col]
            augmented[row] -= factor * augmented[col]   

    # [I | A^-1]에서 오른쪽 절반 반환
    return augmented[:, n:]


def gauss_eliminate(A, b, pivoting: bool = True, verbose: bool = False):
    """가우스 소거법 + 후진대입으로 Ax = b 를 푼다.

    Parameters
    ----------
    pivoting : True 면 부분 피벗팅을 적용한다. False 면 피벗을 그대로 쓴다
               (문제 4-4 에서 두 경우의 오차를 비교하므로 **둘 다 동작해야 한다**).
    verbose  : True 면 각 소거 단계의 첨가행렬 [A|b] 를 출력한다
               (문제 4-1 이 요구하는 '단계별 출력').

    Returns
    -------
    x : 해 벡터
    steps : 단계별 첨가행렬 [A|b] 스냅샷 리스트 (초기 상태 포함)

    피벗이 0 이면 해가 유일하지 않다 -> ZeroDivisionError.
    """
    # TODO: 문제 4-1
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    if A.ndim !=2 or A.shape[0] != A.shape[1]:
        raise ValueError("A는 정사각행렬이어야 합니다.")

    n = A.shape[0]

    if b.shape != (n,):
        raise ValueError("b는 A의 행 개수와 같은 길이의 벡터여야 합니다.")

    augmented = np.column_stack([A, b])
    steps = [augmented.copy()]

    if verbose:
        print("초기 첨가행렬:")
        print(augmented)
    
    for col in range (n):
        if pivoting:
            pivot_row = col + np.argmax(np.abs(augmented[col:, col]))

            if pivot_row != col:
                augmented[[col, pivot_row]] = augmented [[pivot_row, col]]

                steps.append(augmented.copy())

                if verbose:
                    print(f"{col}번 행과 {pivot_row}번 행 교환:")
                    print(augmented)

        if augmented[col, col] == 0:
            raise ValueError(
                "현재 피벗이 0이므로 소거를 진행할 수 없습니다."
            )

        for row in range(col + 1, n):
            factor = augmented[row, col] / augmented[col, col]
            augmented[row] -= factor * augmented[col]

            steps.append(augmented.copy())

            if verbose:
                print(f"{col}번 피벗으로 {row}번 행 소거:")
                print(augmented)

    x = np.zeros(n)

    for row in range(n-1, -1, -1):
        known = augmented[row, row + 1:n] @ x[row + 1:]
        x[row] = (augmented[row, n] - known) / augmented[row, row]

    return x, steps



    

    



    
    
    
 

    
                         

