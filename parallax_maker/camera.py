# (c) 2024 Niels Provos

import numpy as np

#: Largest supported pitch; beyond this the image's top or bottom rays would
#: no longer reach the vertical card planes in front of the camera.
MAX_PITCH_DEGREES = 60.0


class Camera:
    """Pinhole camera for the card scene.

    World frame (OpenCV convention): x right, y down, z forward and
    horizontal; the ground is horizontal and the cards are vertical planes
    ``z = const``. The reference camera sits at ``(0, 0, -distance)`` and may
    be pitched up (positive) or down about the x axis; ``camera_position``
    is where the camera currently is (camera navigation moves it; the
    orientation stays the reference pitch).
    """

    __slots__ = (
        "_camera_position",
        "_camera_distance",
        "_max_distance",
        "_focal_length",
        "_sensor_width",
        "_pitch",
    )

    def __init__(
        self,
        distance=100,
        max_distance=500,
        focal_length=100,
        sensor_width=35.0,
        pitch=0.0,
    ):
        """
        Initializes a Camera object.

        It isn't really a camera but encapsulates related properties.

        Args:
            distance (float): The distance of the camera from the scene.
            max_distance (float): The maximum distance/depth of the scene.
            focal_length (float): The focal length of the camera.
            sensor_width (float, optional): The width of the camera sensor. Defaults to 35.0.
            pitch (float, optional): Upward tilt in degrees (negative looks down). Defaults to 0.

        Returns:
            None
        """
        self.camera_position = np.array([0, 0, -distance], dtype=np.float32)
        self.camera_distance = distance
        self.max_distance = max_distance
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.pitch = pitch

    def focal_length_px(self, image_width):
        """
        Calculate the focal length in pixels.

        Args:
            image_width (int): The width of the image.

        Returns:
            float: The focal length in pixels.
        """
        return (image_width * self.focal_length) / self.sensor_width

    def camera_matrix(self, image_width, image_height):
        """
        Calculate the camera matrix.

        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image.

        Returns:
            np.ndarray: The camera matrix.
        """

        fl_px = self.focal_length_px(image_width)
        return np.array(
            [[fl_px, 0, image_width / 2], [0, fl_px, image_height / 2], [0, 0, 1]],
            dtype=np.float32,
        )

    def rotation_camera_to_world(self):
        """Rotation taking camera-frame directions (x right, y down, z along the
        optical axis) to world directions; a pitch about the x axis."""
        theta = np.radians(self._pitch)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def rotation_world_to_camera(self):
        return self.rotation_camera_to_world().T

    def reference_position(self):
        """Where the cards are built from: ``(0, 0, -distance)``."""
        return np.array([0, 0, -self._camera_distance], dtype=np.float32)

    def horizon_row(self, image_width, image_height):
        """Image row where the horizon (horizontal directions) projects."""
        fl_px = self.focal_length_px(image_width)
        return image_height / 2 + fl_px * np.tan(np.radians(self._pitch))

    def pitch_for_horizon(self, row, image_width, image_height):
        """Pitch (degrees) that puts the horizon at image ``row``."""
        fl_px = self.focal_length_px(image_width)
        return float(np.degrees(np.arctan((row - image_height / 2) / fl_px)))

    def pitch_fits(self, image_width, image_height, pitch=None, focal_length=None):
        """Whether every image row's ray still points forward (towards the
        card planes) at ``pitch``/``focal_length`` (defaults: the current
        values): the pitch plus half the vertical field of view must stay
        below 90 degrees."""
        pitch = self._pitch if pitch is None else pitch
        focal_length = self._focal_length if focal_length is None else focal_length
        fl_px = image_width * focal_length / self._sensor_width
        half_fov = np.degrees(np.arctan((image_height / 2) / fl_px))
        return abs(pitch) + half_fov < 89.0

    def backproject_to_depth(self, points, z, image_width, image_height):
        """World points where the reference camera's rays through image
        ``points`` (N x 2, pixels) meet the vertical plane at depth ``z``."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        fl_px = self.focal_length_px(image_width)
        directions = np.stack(
            [
                (points[:, 0] - image_width / 2) / fl_px,
                (points[:, 1] - image_height / 2) / fl_px,
                np.ones(len(points)),
            ],
            axis=1,
        )
        directions = directions @ self.rotation_camera_to_world().T
        origin = self.reference_position().astype(np.float64)
        if (directions[:, 2] <= 0).any():
            raise ValueError("pitch too steep: image rays do not reach the card plane")
        t = (z - origin[2]) / directions[:, 2]
        return (origin + t[:, None] * directions).astype(np.float32)

    def to_json(self):
        return {
            "position": self._camera_position.tolist(),
            "camera_distance": self._camera_distance,
            "max_distance": self._max_distance,
            "focal_length": self._focal_length,
            "pitch": self._pitch,
        }

    @staticmethod
    def from_json(data):
        camera = Camera()
        if "position" in data:
            camera.camera_position = np.array(data["position"], dtype=np.float32)
        if "camera_distance" in data:
            camera.camera_distance = data["camera_distance"]
        if "max_distance" in data:
            camera.max_distance = data["max_distance"]
        if "focal_length" in data:
            camera.focal_length = data["focal_length"]
        if "pitch" in data:
            camera.pitch = data["pitch"]

        return camera

    def __str__(self):
        return (
            f"Camera(distance={self._camera_distance}, max_distance={self._max_distance}, "
            f"focal_length={self._focal_length}, pitch={self._pitch}, "
            f"position={self._camera_position})"
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Camera):
            return False
        return (
            self.camera_distance == other.camera_distance
            and self.max_distance == other.max_distance
            and self.focal_length == other.focal_length
            and self.pitch == other.pitch
        )

    @property
    def camera_position(self):
        return self._camera_position

    @camera_position.setter
    def camera_position(self, value):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != (3,)
        ):
            raise ValueError(
                "camera_position must be a numpy array of dtype 'float32' with shape (3,)"
            )
        self._camera_position = value

    @property
    def camera_distance(self):
        return self._camera_distance

    @camera_distance.setter
    def camera_distance(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("camera_distance must be a non-negative number")
        self._camera_distance = value

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("max_distance must be a non-negative number")
        self._max_distance = value

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("focal_length must be a positive number")
        self._focal_length = value

    @property
    def sensor_width(self):
        return self._sensor_width

    @sensor_width.setter
    def sensor_width(self, value):
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("sensor_width must be a positive number")
        self._sensor_width = value

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if not isinstance(value, (float, int)) or abs(value) > MAX_PITCH_DEGREES:
            raise ValueError(
                f"pitch must be a number of degrees within ±{MAX_PITCH_DEGREES}"
            )
        self._pitch = float(value)
