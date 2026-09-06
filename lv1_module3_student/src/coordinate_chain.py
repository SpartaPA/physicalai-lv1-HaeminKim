from __future__ import annotations
import numpy as np
from .rotation import axis_angle_from_matrix, rot_x, rot_y, rot_z
from .transform import inv_T, make_T, transform_points

__all__ = [
    "CoordinateChain",
    "default_chain",
    "camera_point_to_base",
    "base_point_to_camera"
]


class CoordinateChain:
    """부모 -> 자식 동차변환을 이름으로 등록하고, 임의의 두 프레임 사이 변환을 만든다."""


    def __init__(self, root: str = "base"):
        self.root = root
        self._parent : dict[str, str] = {}               # child -> parent
        self._T: dict[tuple[str, str], np.ndarray] = {}  # (parent, child) -> T


    def add(self, parent: str, child: str, T) -> "CoordinateChain":
        """parent 기준으로 표현된 child 프레임의 자세 T(parent<-child) 를 등록한다.

        체이닝이 되도록 self 를 돌려준다. 4x4 가 아니면 ValueError.
        """
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"4x4 동차변환이 필요합니다. 받은 shape={T.shape}")
        self._parent[child] = parent
        self._T[(parent, child)] = T
        return self


    def get(self, parent: str, child: str) -> np.ndarray:
        """등록해 둔 T(parent <- child) 를 그대로 돌려준다."""
        return self._T[(parent, child)]


    def frames(self) -> list[str]:
        """등록된 프레임 이름 목록 (root 포함)."""
        return [self.root] + list(self._parent.keys())
# -------------------------------------------------------------------------아래 구현
    
    
    def _path_to_root(self, frame: str) -> list[str]:
        """frame 에서 root 까지의 경로 [frame, ..., root] 를 만든다.
        root 에 연결되어 있지 않으면 KeyError.
        """
        # TODO: 문제 6-1
        path = [frame]

        while frame != self.root:
            if frame not in self._parent:
                raise KeyError(f"{frame}이 루트 {self.root}에 연결되어 있지 않습니다.")
                
            frame = self._parent[frame]

            if frame in path:
                raise KeyError("부모 관계가 순환하여 루트에 도달할 수 없습니다.")

            path.append(frame)

        return path


    def T_from_root(self, frame: str) -> np.ndarray:
        """root 기준 frame 의 자세 T(root <- frame).

        경로를 따라가며 등록된 변환을 곱한다. 곱하는 **순서**에 주의할 것:
        윗첨자/아랫첨자가 이웃끼리 상쇄되도록 놓으면 틀리지 않는다.
            T(base<-camera) = T(base<-link) @ T(link<-camera)
        """
        # TODO: 문제 6-1
        path = self._path_to_root(frame)

        result = np.eye(4)

        for i in range(len(path)-1):
            child = path[i]
            parent = path[i + 1]
            result = self.get(parent, child) @ result

        return result

    def T(self, target: str, source: str) -> np.ndarray:
        """source 좌표를 target 좌표로 바꾸는 변환 T(target <- source).

        힌트: T(target<-source) = inv(T(root<-target)) @ T(root<-source)
        """
        # TODO: 문제 6-1
        T_root_target = self.T_from_root(target)
        T_root_source = self.T_from_root(source)

        inv_T(T_root_target)
        return inv_T(T_root_target) @ T_root_source


    def transform(self, target: str, source: str, P, w: float = 1.0) -> np.ndarray:
        """source 프레임의 점(w=1) 또는 방향(w=0)을 target 프레임으로 변환한다.

        (3,) 와 (N,3) 을 모두 지원해야 하고, **반복문을 쓰지 않는다**.
        """
        # TODO: 문제 6-2
        P = np.asarray(P, dtype= float)

        if P.ndim == 1:
            if P.shape != (3,):
                raise ValueError("점 하나는 shape (3,)이어야 합니다.")
        elif P.ndim == 2:
            if P.shape[1] != 3:
                raise ValueError("여러 점은 shape (N, 3)이어야 합니다.")
        else:
            raise ValueError("입력은 shape (3,) 또는 (N,3)이어야 합니다.")

        single_point = P.ndim == 1

        points = np.atleast_2d(P)

        T_matrix = self.T(target, source)
        result = transform_points(T_matrix, points, w=w)

        if single_point:
            return result[0]

        return result


    def axis_angle(self, target: str, source: str):
        """T(target <- source) 의 회전 부분에서 회전축과 회전각을 복원한다."""
        # TODO: 문제 6-4
        T_matrix = self.T(target, source)

        R = T_matrix[:3, :3]

        return axis_angle_from_matrix(R)


def default_chain() -> CoordinateChain:
    """과제에서 쓸 기본 체인(base -> link -> camera)을 만든다.

    지시문은 '임의의 회전·병진'을 쓰라고 하지만, 채점 수치를 맞추기 위해
    아래 값을 그대로 쓰기를 권장한다. (바꾸려면 노트북에도 그 값을 명시할 것)

    base -> link   : z축 30도 회전 후 (0.30, 0.00, 0.40) m 이동
    link -> camera : y축 -20도, x축 90도 회전(y 먼저 곱함) 후 (0.10, 0.05, 0.15) m 이동
    """
    # TODO: 문제 6-1
    #   T_base_link   = make_T(rot_z(...), [...])
    #   T_link_camera = make_T(rot_y(...) @ rot_x(...), [...])
    #   return CoordinateChain("base").add(...).add(...)
    R_base_link = rot_z(np.deg2rad(30))
    T_base_link = make_T(
        R_base_link,
        [0.30, 0.00, 0.40],
    )
    
    R_link_camera = (
        rot_y(np.deg2rad(-20)) @ rot_x(np.deg2rad(90))
    )

    T_link_camera = make_T(R_link_camera, [0.10, 0.05, 0.15])

    chain = CoordinateChain("base")
    chain.add("base", "link", T_base_link)
    chain.add("link", "camera", T_link_camera)

    return chain


def camera_point_to_base(p_cam, chain: CoordinateChain | None = None) -> np.ndarray:
    """카메라 기준 좌표 -> base 기준 좌표. (3,) 와 (N,3) 모두 지원.

    chain 이 None 이면 default_chain() 을 쓴다.
    """
    # TODO: 문제 6-1
    if chain is None:
        chain = default_chain()

    return chain.transform("base", "camera", p_cam)


def base_point_to_camera(p_base, chain: CoordinateChain | None = None) -> np.ndarray:
    """base 기준 좌표 -> 카메라 기준 좌표. 왕복 검증(문제 6-2)에 쓴다."""
    # TODO: 문제 6-2
    if chain is None:
        chain = default_chain()

    return chain.transform("camera", "base", p_base)

