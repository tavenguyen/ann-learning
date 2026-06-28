# DataPoints — Code and Usage

This file extracts the data generator classes from Model/model.py with example usage.

## Line
```python
class Line:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std=2.0, random_state=None):
        # Use local RNG for reproducibility
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        N = n_points
        K = n_classes
        D = n_dimensions

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype='uint8')
        for j in range(K):
            a = 2 * (j - 1)
            b0 = (j - 2) * 2
            ix = range(N * j, N * (j + 1))
            t = np.linspace(-10, 10, N)

            noise = rng.normal(loc=0.0, scale=noise_std, size=N)

            if D == 2:
                y = a * t + b0 + noise
                self.P[ix] = np.c_[t, y]
            else:
                raise NotImplementedError("Only D==2 supported for Line data generator.")
            self.L[ix] = j
```

Example:
```python
from Model.model import Line
ld = Line(50, 3, 2, noise_std=1.5, random_state=42)
P, L = ld.P, ld.L
```

## Quadratic
```python
class Quadratic:
    def __init__(self, n_points, a, b, c, x_min = -10.0, x_max = 10.0, noise_std = 2.0):
        self.a = a
        self.b = b
        self.c = c

        self.X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

        # noise ~ N(0, noise_std^2)
        noise = np.random.randn(n_points, 1) * noise_std
        self.Y = (a * self.X ** 2 + b * self.X + c) + noise
```

## SpiralData
```python
class SpiralData:
    def __init__(self, n_points, n_classes):
        dimensions = 2
        self.P = np.zeros((n_points * n_classes, dimensions))
        self.L = np.zeros(n_points * n_classes, dtype = 'uint8')

        for j in range(n_classes):
            ix = range(n_points * j , n_points * (j + 1))
            r = np.linspace(0, 1, n_points)
            theta = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.6
        
            self.P[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
            self.L[ix] = j
```

## PointZone
```python
class PointZone:
    def __init__(self, n_points, n_classes, cluster_std=1.0, radius=3.0, random_state=None):
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        self.P = np.zeros((n_points * n_classes, 2))
        self.L = np.zeros(n_points * n_classes, dtype='uint8')
        
        for j in range(n_classes):
            offset_x = np.cos(j * np.pi / 2) * radius
            offset_y = np.sin(j * np.pi / 2) * radius
            
            x = rng.normal(loc=offset_x, scale=cluster_std, size=n_points)
            y = rng.normal(loc=offset_y, scale=cluster_std, size=n_points)
            
            ix = range(n_points * j, n_points * (j + 1))
            self.P[ix] = np.c_[x, y]
            self.L[ix] = j
```

## Circle2D
```python
class Circle2D:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std=2.0, radius_step=2.0, random_state=None, endpoint=False):
        """Generate concentric noisy circles (2D only).

        Parameters
        - n_points: points per class
        - n_classes: number of concentric circles
        - n_dimensions: must be 2
        - noise_std: std dev of radial noise
        - radius_step: base step between radii (radius = radius_step * (j+1))
        - random_state: optional int for reproducibility
        - endpoint: pass to np.linspace for theta (default False to avoid duplicate point)
        """
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        N = n_points
        K = n_classes
        D = n_dimensions

        if D != 2:
            raise NotImplementedError("Circle2D supports only 2D (n_dimensions==2)")

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype='uint8')

        for j in range(K):
            base_r = radius_step * (j + 1)
            r_noise = base_r + rng.normal(scale=noise_std, size=N)

            theta = np.linspace(0, 2 * np.pi, N, endpoint=endpoint)
            idx = range(N * j, N * (j + 1))

            # polar -> Cartesian
            self.P[idx] = np.c_[r_noise * np.cos(theta), r_noise * np.sin(theta)]
            self.L[idx] = j
```

Notes
- Use random_state or a local RNG if you need reproducibility across generators.
- Standardize features before training for better convergence.
