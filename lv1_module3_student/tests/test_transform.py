"""문제 5 — 동차변환 inv_T 검증 (pytest). [학생 작성용 템플릿]

지시문이 요구하는 것은 `inv_T` 검증이지만,
점/방향 구분과 벡터화, 최소자승까지 함께 검증해 두면 이후 문제에서 안전하다.

실행: 프로젝트 루트에서  pytest -v
"""

import numpy as np
import pytest

from src.rotation import rot_x, rot_y, rot_z
from src.transform import (
    inv_T,
    least_squares_normal_equation,
    make_T,
    transform_direction,
    transform_point,
    transform_points,
)


@pytest.fixture
def T():
    """테스트에 쓸 대표 동차변환 하나."""
    R = rot_z(0.9) @ rot_y(-0.35) @ rot_x(1.3)
    return make_T(R, [0.35, -0.15, 0.55])


def test_inv_T_gives_identity(T):
    # TODO: inv_T(T) @ T 와 T @ inv_T(T) 가 모두 4x4 단위행렬인지 검사
    T_inv = inv_T(T)
    identity = np.eye(4)
    assert np.allclose(T_inv @ T, identity), "inv_T(T) @ T가 단위행렬이 아닙니다."
    assert np.allclose(T @ T_inv, identity), "T @ inv_T(T)가 단위행렬이 아닙니다."


def test_inv_T_matches_generic_inverse(T):
    # TODO: inv_T(T) 가 np.linalg.inv(T) 와 일치하는지 검사 (np.linalg 는 검산용)
    assert np.allclose(inv_T(T), np.linalg.inv(T)), "inv_T(T)가 Numpy 역행렬과 일치하지 않습니다."


def test_point_and_direction_differ(T):
    # TODO: 같은 벡터를 점(w=1)/방향(w=0)으로 변환하면 결과가 다르고,
    #       그 차이가 정확히 병진 벡터 T[:3, 3] 이며,
    #       방향 변환은 길이를 보존하는지 검사
    # 점: Rp + t, 방향: Rv
    v = np.array([1.0, 2.0, 3.0])

    point_result = transform_point(T,v)
    direction_result = transform_direction(T,v)
    assert not np.allclose(point_result, direction_result), "이동이 있는 변환에서 점과 방향의 결과가 같습니다."

    # 이동 벡터인지 확인: (Rv + t)-Rv = t
    translation = T[:3,3]

    assert np.allclose(point_result - direction_result, translation), "점과 방향의 변환 결과 차이가 이동 벡터와 다릅니다."
    assert np.isclose(np.linalg.norm(direction_result), np.linalg.norm(v),), "방향 변환 후 벡터의 길이가 달라졌습니다."


def test_transform_points_is_vectorized(T):
    # TODO: (N,3) 점군을 한 번에 변환한 결과가
    #       transform_point 를 반복문으로 돌린 결과와 같은지 검사
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-2.0, 0.5, 1.0],
        [0.3, -1.0, 2.0]
    ])
    actual = transform_points(T, points)

    expected = np.array([
        transform_point(T, point)
        for point in points
    ])

    assert actual.shape == points.shape, "변환 결과의 형태가 (N, 3)이 아닙니다."
    assert np.allclose(actual, expected), "일괄 변환 결과가 개별 변환 결과와 다릅니다."


def test_roundtrip_through_inverse(T):
    # TODO: T 로 보냈다가 inv_T(T) 로 되돌리면 원래 점군이 나오는지 검사
    # p' = Tp, T^-1p' = p

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-2.0, 0.5, 1.0],
        [0.3, -1.0, 2.0]
    ])

    transformed = transform_points(T, points
    )
    T_inv = inv_T(T)
    restored = transform_points(T_inv, transformed)
    
    assert restored.shape == points.shape, "복원된 점군의 형태가 원본과 다릅니다."
    assert np.allclose(restored, points), "변환 후 역변환한 점들이 원본과 다릅니다."


def test_least_squares_matches_lstsq():
    # TODO: 노이즈를 섞은 과결정 문제를 만들어
    #       least_squares_normal_equation 의 해가 np.linalg.lstsq 와 일치하고
    #       잔차가 A 의 열공간에 수직(A^T r = 0)인지 검사
    rng = np.random.default_rng(42)
    
    # 과결정 문제
    A = rng.normal(size=(8,3))
    x_true = np.array([1.0, -2.0, 0.5])

    # 잡음 추가
    noise = rng.normal(scale=0.1, size=8)
    b = A @ x_true + noise

    actual, residual = least_squares_normal_equation(A, b)

    #정규 방정식: A^TAx = A^Tb 을 이용해 검사
    #numpy 비교
    expected, _, _, _ = np.linalg.lstsq(A, b, rcond=None) # 검산용
    
    assert actual.shape == expected.shape, "최소자승 해의 형태가 올바르지 않습니다."
    assert np.allclose(actual, expected), "최소자승 해가 NumPy 결과와 다릅니다."

    # 최소자승 해: A^Tr = 0 성립
    residual = b - A @ actual

    assert np.allclose(A.T @ residual, np.zeros(3), atol=1e-8), "잔차가 A의 열공간에 수직이 아닙니다."
        

