from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from .rotation import rot_z

__all__=[
    "make_T",
    "inv_T",
    "to_homogeneous",
    "transform_point",
    "transform_direction",
    "transform_points",
    "inv_T_batch",
    "least_squares_normal_equation",
    "rmse",
]


def make_T(R,t) -> np.ndarray:
    """회전행렬과 병진으로 이루어진 4x4 변환행렬을 생성합니다."""
    t = np.asarray(t, dtype=float)

    if R.shape != (3,3):
        raise ValueError("R은 3x3 회전행렬이어야 합니다.")
    if t.shape != (3,):
        raise ValueError("t은 shape (3,)의 벡터여야 합니다.")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    """4x4 변환행렬의 역행렬을 계산합니다."""
    R = T[:3, :3]
    t = T[:3, 3]
    if T.shape != (4,4):
        raise ValueError("T는 4x4 변환행렬이어야 합니다.")
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def inv_T_batch(Ts) -> np.ndarray:
    """(N, 4, 4) 동차변환 묶음을 반복문 없이 한번에 역변환한다."""
    # 1. 입력 검증 (N, 4, 4) 형태가 맞는지 검증
    Ts = np.asarray(Ts, dtype=float)
    if Ts.ndim != 3 or Ts.shape[-2:] != (4,4):
        raise ValueError("Ts는 shape (N, 4, 4)의 배열이어야 합니다.")
    
    # 2. 회전행렬 R과 병진 벡터 t 추출
    R = Ts[:, :3, :3]  # shape (N, 3, 3)
    t = Ts[:, :3, 3]   # shape (N, 3)   

    # 3. R의 배치 전치 구하기
    R_T = np.swapaxes(R, 1, 2)  # shape (N, 3, 3)

    # 4. -R^T * t 의 배치 연산 구하기
    neg_R_Tt = -np.einsum('nij,nj->ni', R_T, t)  # shape (N, 3)

    # 5. 결과를 담을 (N, 4, 4) 빈 배열 ᄄᆃ는 영행렬 생성 후 채우기
    N = Ts.shape[0]
    inv_Ts = np.zeros((N, 4, 4), dtype=Ts.dtype)

    inv_Ts[:, :3, :3] = R_T
    inv_Ts[:, :3, 3] = neg_R_Tt
    inv_Ts[:, 3, 3] = 1.0 # 마지막 행은 [0, 0, 0, 1]로

    return inv_Ts


def transform_point(T,v) -> np.ndarray:
    """점 변환 (w=1): 회전과 병진이 모두 적용된다. 반환은 shape (3,)의 벡터이다."""
    v = np.asarray(v, dtype=float)
    T = np.asarray(T, dtype=float)

    if v.shape != (3,):
        raise ValueError("v는 shape (3,)의 벡터여야 합니다.")
    if T.shape != (4,4):
        raise ValueError("T는 4x4 변환행렬이어야 합니다.")
    
    p_homo = to_homogeneous(v,1)
    new_p = T @ p_homo
        
    return new_p[:3]
 
    
def to_homogeneous(P, w:float = 1.0) -> np.ndarray:
    """3D 점들을 동차좌표로 변환합니다."""
    P = np.asarray(P, dtype=float)

    if P.shape == (3,):
        return np.append(P, w)
        
    if P.ndim == 2 or P.shape[1] == 3:
        w_column = np.full((P.shape[0], 1), w)
        return np.hstack((P, w_column))
        
    raise ValueError("P는 shape (N, 3)의 배열이어야 합니다.")    

    
def transform_direction(T, v) -> np.ndarray:
    """4x4 변환행렬 T를 이용하여 3D 방향 벡터 v를 변환합니다."""
    T = np.asarray(T, dtype=float)
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError("v는 shape (3,)의 벡터이어야 합니다.")
    if T.shape != (4,4):
        raise ValueError("T는 4x4 변환행렬이어야 합니다.")
    R = T[:3, :3]
    new_v = R @ v
    return new_v
    

def transform_points(T, P, w:float = 1.0) -> np.ndarray:
    """4x4 변환행렬 T를 이용하여 3D 점들 P를 변환합니다."""
    P_homogeneous = to_homogeneous(P, w)
    new_P_homogeneous = T @ P_homogeneous.T
    new_P = new_P_homogeneous.T[:, :3]

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P는 shape (N, 3)의 배열이어야 합니다.")
    if T.shape != (4,4):
        raise ValueError("T는 4x4 변환행렬이어야 합니다.")

    return new_P


def least_squares_normal_equation(A, b):
    """정규방정식 (A^T A) x = A^T b 를 직접 세워 최소장승해를 구한다."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    x = np.linalg.solve(A.T @ A, A.T @ b)
    residual = b - A @ x

    return x, residual


def rmse(residual):
    residual = np.asarray(residual, dtype=float)
    return float(np.sqrt(np.mean(residual ** 2)))

