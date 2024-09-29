import random

class PoissonDiskSampling:
    """
    Poisson Disk Sampling
    """

    def __init__(self, r_min: float, r_max: float, random_seed: int = 0):
        """
        Args:
            r_min (float): minimum radius between points
            r_max (float): maximum radius between points
            random_seed (int): random seed
        """
        self.r_min = r_min
        self.r_max = r_max
        random.seed(random_seed)

    def random(self, n: int, max_iter: int = 30) -> list[tuple[float, float]]:
        """
        Generate n random points in the region of [0, 1) x [0, 1)

        Args:
            n (int): number of points
            max_iter (int): maximum number of iterations of attempts to place a point

        Returns:
            list: list of (x, y) points
        """
        points = []
        for i in range(n):
            for attempt in range(max_iter):
                x, y = random.random(), random.random()
                if self.is_valid_point(x, y, points):
                    points.append((x, y))
                    break
                if attempt == max_iter - 1:
                    raise ValueError(f'Failed to place point after {max_iter} attempts at {i + 1}th point.')
        return points

    def is_valid_point(self, x: float, y: float, points: list[tuple[float, float]]) -> bool:
        """
        Check if the point (x, y) is valid according to the r_min and r_max constraints.

        Args:
            x (float): x-coordinate of the new point
            y (float): y-coordinate of the new point
            points (list): existing points

        Returns:
            bool: True if the point is valid, False otherwise
        """
        for px, py in points:
            dist_sq = (x - px) ** 2 + (y - py) ** 2
            if dist_sq < self.r_min ** 2 or dist_sq > self.r_max ** 2:
                return False
        return True
