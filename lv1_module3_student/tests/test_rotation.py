"""문제 3 — 회전 행렬의 수학적 성질 검증 (pytest). [학생 작성용 템플릿]

지시문이 요구하는 4가지를 각각 테스트 함수로 작성한다.

  1. 회전행렬의 열이 서로 직교하는 단위벡터인가   -> test_columns_are_orthonormal
  2. 행렬식이 1인가                               -> test_determinant_is_one
  3. 역행렬이 전치와 같은가                       -> test_inverse_equals_transpose
  4. 재직교화 결과가 직교행렬인가                 -> test_gram_schmidt_restores_orthogonality

작성 요령
--------
- `@pytest.mark.parametrize` 로 여러 축 x 여러 각도를 한 함수에서 검사하면
  테스트 하나가 여러 케이스를 담당한다 (아래 ANGLES / MAKERS 참고).
- 비교는 반드시 `np.isclose` / `np.allclose` 로 한다 (부동소수점).
- `np.linalg` 는 검산용으로만 쓰고, 쓸 때는 주석으로 검산임을 밝힌다.
- assert 에 실패 메시지를 붙이면 어디가 깨졌는지 바로 보인다.
- 4개는 **최소 개수**다. 반사 행렬 반례, 로드리게스 일치, 축·각 왕복 같은
  테스트를 더 붙이면 좋다.

실행: 프로젝트 루트에서  pytest -v
"""

import numpy as np
import pytest

from src.rotation import (
    axis_angle_from_matrix,
    gram_schmidt,
    is_rotation,
    orthogonality_error,
    rodrigues,
    rot_x,
    rot_y,
    rot_z,
)

ANGLES = [0.0, np.deg2rad(22.5), np.pi / 6, np.pi / 4, np.pi / 2, 2.0, np.pi, -1.234]
MAKERS = [rot_x, rot_y, rot_z]


@pytest.fixture
def rng():
    """난수는 반드시 시드를 고정한다."""
    return np.random.default_rng(42)


# --- 1. 열이 서로 직교하는 단위벡터인가 -------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_columns_are_orthonormal(maker, theta):
    R = maker(theta)
    # 1. 열벡터의 내적이 0인지 확인
    dot_products = R.T @ R
    identity = np.eye(R.shape[0])
    assert np.allclose(dot_products, identity), f"열벡터가 직교하지 않음: {dot_products}"


# --- 2. 행렬식이 1인가 --------------------------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_determinant_is_one(maker, theta):
    R = maker(theta)
    det_R = np.linalg.det(R)  # 검산용
    assert np.isclose(det_R, 1.0), f"행렬식이 1이 아님: det(R)={det_R}"


# --- 3. 역행렬 == 전치 --------------------------------------------------------

@pytest.mark.parametrize("maker", MAKERS)
@pytest.mark.parametrize("theta", ANGLES)
def test_inverse_equals_transpose(maker, theta):
    R = maker(theta)
    assert np.allclose(np.linalg.inv(R), R.T), "역행렬이 전치와 일치하지 않습니다"


# --- 4. 재직교화 결과가 직교행렬인가 -----------------------------------------

def test_gram_schmidt_restores_orthogonality(rng):
    """Gram-Schmidt 재직교화가 직교행렬을 복원하는지 확인."""
    # 1. 임의의 직교행렬 R 생성
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)  # 검산용
    theta = rng.uniform(-np.pi, np.pi)
    R = rodrigues(axis, theta)

    # 2. R에 작은 잡음을 더해 직교성을 깨뜨린다
    noise = rng.normal(scale=1e-3, size=R.shape)
    R_noisy = R + noise

    # 3. Gram-Schmidt 재직교화 수행
    R_orthogonalized = gram_schmidt(R_noisy)

    # 4. 재직교화 결과가 직교행렬인지 확인
    error = orthogonality_error(R_orthogonalized)
    assert error < 1e-6, f"재직교화 후 직교성 오차가 너무 큼: {error}"
    
# --- 여기부터는 추가 테스트 (권장) -------------------------------------------

def test_reflection_is_not_a_rotation():
    """det = -1 인 반사 행렬은 직교여도 회전이 아니다."""
    R = np.diag([1, 1, -1])
    assert is_rotation(R) is False, "반사 행렬은 회전이 아니다"


@pytest.mark.parametrize("theta", ANGLES)
def test_rodrigues_matches_rot_z(theta):
    """Rodrigues 공식이 rot_z 와 일치하는지 확인."""
    R1 = rodrigues([0, 0, 1], theta)
    R2 = rot_z(theta)
    assert np.allclose(R1, R2), "Rodrigues 공식이 rot_z 와 일치하지 않는다"


def test_axis_angle_roundtrip(rng):
    """축·각 -> 회전행렬 -> 축·각 왕복이 일치하는지 확인."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)  # 검산용
    theta = rng.uniform(-np.pi, np.pi)
    R = rodrigues(axis, theta)
    axis2, theta2 = axis_angle_from_matrix(R)
    assert np.allclose(axis, axis2) or np.allclose(axis, -axis2), "축이 일치하지 않는다"
    assert np.isclose(theta, theta2), "각도가 일치하지 않는다"




