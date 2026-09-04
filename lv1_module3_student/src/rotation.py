from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from .vectors import det, normalize,skew


__all__=[
    "rot_x",
    "rot_y",
    "rot_z"
    "inverse_gauss_jordan",
    "orthogonality_error",
    "rodrigues",
    "gram_schmidt",
    "is_rotation",
    "axis_angle_from_matrix",
    "quaternion_from_axis_angle",
]

def rot_x(theta) -> np.ndarray:
    """입력된 각도만큼 x축으로 회전하는 회전행렬을 반환한다."""
    return np.array([
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta), np.cos(theta)]
    ])


def rot_y(theta) -> np.ndarray:
    """입력된 각도만큼 y축으로 회전하는 회전행렬을 반환한다."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,1,0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rot_z(theta) -> np.ndarray:
    """입력된 각도만큼 z축으로 회전하는 회전행렬을 반환한다."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0,0,1]
    ])


def orthogonality_error(R) -> float:
    """직교성 이탈 지표: || R^T R - I ||_F (프로베니우스 노름)."""
    R = np.asarray(R, dtype=float)
    E = R.T @ R - np.eye(R.shape[0])
    return float(np.sqrt(np.sum(E * E)))


def rodrigues(axis, theta):
    axis = np.array (axis, dtype=float)
    k = axis / np.linalg.norm(axis)
    K = skew(k)
    I = np.eye(3)
    return I + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def gram_schmidt(A) -> np.ndarray:
    A = np.array(A, dtype=float, copy=True)

    n_cols = A.shape[1]
    Q = np.zeros_like(A)

    for j in range(n_cols):
        v = A[:,j].copy()

        for i in range(j):
            v -= (Q[:,i]@v)*Q[:,i]

        nv = np.sqrt(v @ v)

        if nv <= 1e-12:
            raise ValueError(
                f"{j}번 열이 앞선 열들에 종속이라 직교화할 수 없습니다."
            )

        Q[:,j] = v/nv

    return Q


def is_rotation():
    pass


def inverse_gauss_jordan(A):
    A = np.array(A, dtype=float, copy=True)
    u, pivots, swaps = row_echelon(A)
    m = A.shape[0]
    n = A.shape[1]
    row = 0
    for r in range(m):
        for c in range(n):
            if A[r,c] != 0:
                row += 1
            if c == m:
                break
        if r == m:
            break

        
    """현재 열에서 피벗 찾기
        ↓
필요하면 행 교환
        ↓
피벗 아래의 값을 0으로 만들기
        ↓
다음 행으로 이동
        ↓
다음 열에서 반복"""
    pass


def axis_angle_from_matrix(R, atol: float = 1e-8):
    """고유값 분해로 회전축을, 대각합으로 회전각을 복원한다."""
    R = np.ndarray(R, dtype=float)

    if R.shape != 3:
        raise ValueError("회전행렬의 형태는 3x3이어야 합니다.")
    
    #고유값, 고유벡터
    eig, eig_v = np.linalg.eig(R)
    
    # 각도 구하기
    traceR = np.sum(R(1,1),R(2,2),R(3,3))
    theta = np.degrees(np.arccos((traceR-1) / 2))
    
    # 단위 회전축 구하기
    v = np.array([
        [R(3,2)-R(2,3)]
        [R(1,3)-R(2,1)]
        [R(2,1)-R(1,2)], dtype:=float
    ])
    k = (1/2*np.sin(theta))*v

    return k, theta

def quaternion_from_axis_angle(axis, angle: float) ->np.ndarray:
    """축-각에서 단위 쿼터니언을 만든다."""
    pass