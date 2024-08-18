from abc import abstractmethod
import math
from typing import List, Tuple


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other: "Point"):
        return self.x == other.x and self.y == other.y

    def distance_to(self, other: "Point"):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Shape:
    def __init__(self):
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        pass

    @property
    @abstractmethod
    def height(self) -> float:
        pass

    @property
    @abstractmethod
    def boundingbox(self) -> Tuple[float, float, float, float]:
        """
        x_1, y_1, x_2, y_2
        """
        pass
    
    @property
    @abstractmethod
    def center(self) -> Point:
        pass

    @property
    @abstractmethod
    def points(self) -> List[Point]:
        pass

    @property
    @abstractmethod
    def area(self) -> float:
        pass

    def is_point_inside(self, point: Point):
        pass
    
    def is_inside(self, other: "Shape", margin: float=20) -> bool:
        pass

    def union(self, other: "Shape") -> "Shape":
        pass

    def intersect(self, other: "Shape") -> "Shape":
        pass

class Rectangle(Shape):
    def __init__(self, x_1: float, y_1: float, x_2: float, y_2: float):
        super().__init__()
        # TODO: Check x1<x2 and y1<y2
        if x_1 >= x_2:
            x_2 = x_1
        if y_1 >= y_2:
            y_2 = y_1
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2

    @property
    def width(self) -> int:
        return self.x_2 - self.x_1

    @property
    def height(self) -> int:
        return self.y_2 - self.y_1

    @property
    def boundingbox(self) -> Tuple[float, float, float, float]:
        return (self.x_1, self.y_1, self.x_2, self.y_2)

    @property
    def center(self) -> Point:
        x_1, y_1, x_2, y_2 = self.boundingbox
        return Point((x_1 + x_2) / 2.0, (y_1 + y_2) / 2.0)

    @property
    def points(self) -> List[Point]:
        return [
            Point(self.x_1, self.y_1),
            Point(self.x_1, self.y_2),
            Point(self.x_2, self.y_2),
            Point(self.x_2, self.y_1),
        ]

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_point_inside(self, point: Point) -> bool:
        return (
            point.x > self.x_1
            and point.x < self.x_2
            and point.y > self.y_1
            and point.y < self.y_2
        )

    def is_inside(self, other: "Rectangle", margin: float=20) -> bool:
        return (
            self.x_1 >= other.x_1 - margin
            and self.y_1 >= other.y_1 - margin
            and self.x_2 <= other.x_2 + margin
            and self.y_2 <= other.y_2 + margin
        )

    def union(self, other: "Shape") -> Shape:
        if isinstance(other, Rectangle):
            return self.__class__(
                min(self.x_1, other.x_1),
                min(self.y_1, other.y_1),
                max(self.x_2, other.x_2),
                max(self.y_2, other.y_2),
            )
        else:
            raise Exception("Unsupported union shape")

    def intersect(self, other: "Shape") -> Shape:
        if isinstance(other, Rectangle):
            return self.__class__(
                max(self.x_1, other.x_1),
                max(self.y_1, other.y_1),
                min(self.x_2, other.x_2),
                min(self.y_2, other.y_2),
            )
        else:
            raise Exception("Unsupported intersect shape")

    def __str__(self):
        return f"Rectangle([{self.x_1}, {self.y_1}], [{self.x_2}, {self.y_2}])"

    def __repr__(self):
        return f"Rectangle([{self.x_1}, {self.y_1}], [{self.x_2}, {self.y_2}])"

class Quadrilateral(Shape):
    def __init__(self, points: List[Point]):
        super().__init__()
        self.points = points

class Polygon(Shape):
    def __init__(self, name, sides):
        super().__init__(name)
        self.sides = sides

    def get_perimeter(self):
        return sum(self.sides)
