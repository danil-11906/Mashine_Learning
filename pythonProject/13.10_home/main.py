import matplotlib.pyplot as plt
from clustering import ShortestUnclosedPath
from point import Point

points = [Point(0, 0) for i in range(0, 120)]

for i in range(0, 120):
    points[i].gen_coordinates(0, 40, 0, 40)

for i in range(20, 50):
    points[i].gen_coordinates(0, 40, 53, 100)

for i in range(50, 80):
    points[i].gen_coordinates(53, 100, 0, 40)

for i in range(80, 120):
    points[i].gen_coordinates(53, 100, 53, 100)

ShortestUnclosedPath(points).show_array()

plt.show()

ShortestUnclosedPath(points).split_into_clusters(4)

plt.style.use('Solarize_Light2')

plt.show()
