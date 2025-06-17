# (c) 2024 Niels Provos

import numpy as np


class Camera:
    __slots__ = (
        "_camera_position",
        "_camera_distance",
        "_max_distance",
        "_focal_length",
        "_sensor_width",
    )

    def __init__(
        self, distance=100, max_distance=500, focal_length=100, sensor_width=35.0
    ):
        """
        Initializes a Camera object.

        It isn't really a camera but encapsulates related properties.

        Args:
            distance (float): The distance of the camera from the scene.
            max_distance (float): The maximum distance/depth of the scene.
            focal_length (float): The focal length of the camera.
            sensor_width (float, optional): The width of the camera sensor. Defaults to 35.0.

        Returns:
            None
        """
        self.camera_position = np.array([0, 0, -distance], dtype=np.float32)
        self.camera_distance = distance
        self.max_distance = max_distance
        self.focal_length = focal_length
        self.sensor_width = sensor_width

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

    def to_json(self):
        return {
            "position": self._camera_position.tolist(),
            "camera_distance": self._camera_distance,
            "max_distance": self._max_distance,
            "focal_length": self._focal_length,
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

        return camera

    def __str__(self):
        return (
            f"Camera(distance={self._camera_distance}, max_distance={self._max_distance}, "
            f"focal_length={self._focal_length}, position={self._camera_position})"
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
