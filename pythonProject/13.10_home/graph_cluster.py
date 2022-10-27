import sys
import matplotlib.pyplot as plt


class Cluster:

    def __init__(self, points):
        self.points = points
        self.adjacency_matrix = [[0 for _ in range(0, len(points))] for _ in range(0, len(points))]
        self.isolated_points = set([i for i in range(0, len(points))])
        self.not_isolated_points = set()

    def has_isolated_points(self):
        return self.get_isolated_point() != -1

    def is_point_isolated(self, index):
        has_rib = False

        for k in range(0, len(self.adjacency_matrix[index])):
            if self.adjacency_matrix[index][k] != 0:
                has_rib = True

        return not has_rib

    def connect_two_nearest_points(self):
        nearest_distance = sys.maxsize
        nearest_i = 0
        nearest_k = 0

        for i in range(0, len(self.adjacency_matrix)):
            for k in range(i + 1, len(self.adjacency_matrix[i])):

                if self.adjacency_matrix[i][k] == 0:
                    distance = self.points[i].distance_to_point(self.points[k])
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_i = i
                        nearest_k = k

        self.connect(nearest_i, nearest_k)

    def connect_not_isolated_point_to_nearest_isolated(self):
        nearest_distance = sys.maxsize
        nearest_i = 0
        nearest_k = 0

        for i in self.isolated_points:
            for k in self.not_isolated_points:
                distance = self.points[i].distance_to_point(self.points[k])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_i = i
                    nearest_k = k

        self.connect(nearest_i, nearest_k)

    def connect(self, i, k):
        self.adjacency_matrix[i][k] = 1
        self.adjacency_matrix[k][i] = 1

        self.isolated_points.discard(i)
        self.not_isolated_points.add(i)

        self.isolated_points.discard(k)
        self.not_isolated_points.add(k)

    def detach(self, i, k):
        self.adjacency_matrix[i][k] = 0
        self.adjacency_matrix[k][i] = 0

        if self.is_point_isolated(i):
            self.isolated_points.add(i)
            self.not_isolated_points.discard(i)
        if self.is_point_isolated(k):
            self.isolated_points.add(k)
            self.not_isolated_points.discard(k)

    def remove_n_longest_ribs(self, n):

        for _ in range(0, n):

            longest_distance = 0
            longest_i = 0
            longest_k = 0

            for i in range(0, len(self.adjacency_matrix)):
                for k in range(i + 1, len(self.adjacency_matrix[i])):
                    if self.adjacency_matrix[i][k] != 0:
                        distance = self.points[i].distance_to_point(self.points[k])
                        if distance > longest_distance:
                            longest_distance = distance
                            longest_i = i
                            longest_k = k

            self.detach(longest_i, longest_k)

    def get_isolated_point(self):
        for i in range(0, len(self.adjacency_matrix)):

            has_rib = False

            for k in range(0, len(self.adjacency_matrix[i])):

                if self.adjacency_matrix[i][k] != 0:
                    has_rib = True

            if not has_rib:
                return i

        return -1

    def show(self):

        for i in range(0, len(self.adjacency_matrix)):
            for k in range(i + 1, len(self.adjacency_matrix[i])):

                if self.adjacency_matrix[i][k] != 0:
                    plt.plot([self.points[i].x, self.points[k].x],
                             [self.points[i].y, self.points[k].y],
                             c='black')
        plt.scatter(
            [point.x for point in self.points],
            [point.y for point in self.points],
            c='blue'
        )

    def build_skeletons(self, k):

        self.remove_n_longest_ribs(n=k)
