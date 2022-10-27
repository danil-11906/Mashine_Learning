from graph_cluster import Cluster


class ShortestUnclosedPath:

    def __init__(self, points):
        self.points = points
        self.cluster = Cluster(self.points)

    def split_into_clusters(self, k):

        self.cluster.connect_two_nearest_points()

        while self.cluster.has_isolated_points():
            self.cluster.connect_not_isolated_point_to_nearest_isolated()

        self.cluster.build_skeletons(k - 1)
        self.cluster.show()

    def show_array(self):

        self.cluster.connect_two_nearest_points()

        while self.cluster.has_isolated_points():
            self.cluster.connect_not_isolated_point_to_nearest_isolated()

        self.cluster.show()
