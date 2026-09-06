codefrom __future__ import annotations
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
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,-s],
                     [0.0, s, c]], dtype=float)


def rot_y(theta) -> np.ndarray:
    """입력된 각도만큼 y축으로 회전하는 회전행렬을 반환한다."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def rot_z(theta) -> np.ndarray:
    """입력된 각도만큼 z축으로 회전하는 회전행렬을 반환한다."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


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

        if nv <= 1e-8:
            raise ValueError(
                f"{j}번 열이 앞선 열들에 종속이라 직교화할 수 없습니다."
            )

        Q[:,j] = v/nv

    return Q


def is_rotation(R, atol: float = 1e-8) -> bool :
    """회전행렬 판정: 직교(R^T R = I) **그리고** det(R) = +1 이면 True.

    det = -1 이면 직교이긴 하지만 반사가 섞여 있어 회전이 아니다.
    3x3 이 아니면 False.
    """
    R = np.array(R, dtype=float)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False
    if np.linalg.det(R) < 0:
        return False
    return True


def axis_angle_from_matrix(R, atol: float = 1e-8):
    """고유값 분해로 회전축을, 대각합으로 회전각을 복원한다."""
    R = np.array(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("회전행렬의 형태는 3x3이어야 합니다.")
    
    
    
    # 각도 구하기
    # 대각 합:
    traceR = R[0,0] + R[1,1] + R[2,2]
    cos_theta = np.clip((traceR - 1) / 2, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    
    if theta <= atol:
    # 회전이 없으면 축이 유일하지 않으므로 +x축으로 정한다.
        return np.array([1.0, 0.0, 0.0]), 0.0

    #고유값, 고유벡터
    eig, eig_v = np.linalg.eig(R)

    idx = np.argmin(np.abs(eig-1))

    # 단위 회전축 구하기
    k = eig_v[:, idx].real
    k = k / np.linalg.norm(k)
    
    v = np.array([
        [R[2,1]-R[1,2]],
        [R[0,2]-R[2,0]],
        [R[1,0]-R[0,1]],
    ], dtype=float)

    if np.pi - theta <= atol:
        idx = np.argmax(np.abs(k))

        if k[idx] < 0:
            k = -k

        return k, float(np.pi)
    
    if k @ v < 0:
        k = -k

    return k, theta


def quaternion_from_axis_angle(axis, angle: float) ->np.ndarray:
    """축-각에서 단위 쿼터니언을 만든다."""
    # k 는 단위 회전축
    # theta는 라디안 각도
    # \(q=(k_x\sin(\theta/2),\;k_y\sin(\theta/2),\;k_z\sin(\theta/2),\;\cos(\theta/2))\) 해당 식을 적용할 예정

    # 입력된 축을 numpy 배열로 바꾼다.
    # 실수형
    axis = np.array(axis, dtype=float)

    #축은 x,y,z 성분을 가진 벡터이므로 (3,) 인지 확인한다.
    # [x ...]
    # [y ...]
    # [z ...] ?
    if axis.shape != (3,):
        raise ValueError("회전축은 원소가 3개인 벡터여야 합니다.")

    #축의 길이를 구하고, 단위벡터로 만든다.
    length = np.linalg.norm(axis)

    # 길이가 0이면 벡터도 0, 즉 회전축은 0일 수 없음.
    if length == 0:
        raise ValueError("회전축은 영벡터일 수 없습니다.")

    # 단위 회전축을 구하기 위해 길이로 나눈다.
    k = axis / length

    # 쿼터니언 공식에 들어갈 theta/2를 준비한다
    half_angle = angle / 2
    s = np.sin(half_angle)
    w = np.cos(half_angle)

    xyz = k * s

    # 쿼터니언 벡터대로 입력한다.
    return np.array([xyz[0], xyz[1], xyz[2], w])



     
