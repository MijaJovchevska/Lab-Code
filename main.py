import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from math import sqrt
from scipy.signal import find_peaks


def main():
    MAX_STEPS = 500
    step = 0
    g = nx.Graph()
    root = Node()
    tips = [root]
    position_dictionary = {root: root.get_coordinates()}
    while step < MAX_STEPS:
        for tip in tips:
            new_edges = tip.grow_from_here(g)
            g.add_edges_from(new_edges)
            for tup in new_edges:
                new_tip = tup[1]
                tips.append(new_tip)
                position_dictionary[new_tip] = new_tip.get_coordinates()
            tips.remove(tip)
        step += 1

    # paths = find_all_paths(g) # find the path segments (from branch node to branch node)

    paths = get_all_follow_paths(g, root)  # find the "main" paths i.e. the paths as they grow

    # plot the tortuosity indices vs the total length for each path
    all_arc_len = all_arc_lengths(paths, position_dictionary)
    tortuosity = tortuosity_index(paths, position_dictionary)
    plt.figure("Tortuosity Index")
    plt.scatter(all_arc_len, tortuosity)
    plt.xlabel("total length")
    plt.ylabel("tortuosity")
    plt.title("Long Vessels (subsampled)")
    plt.show()

    auto_correlations = list()
    path_lengths = list()
    for path in paths:
        if len(path) > 4:
            path_lengths.append(arc_length(path, position_dictionary))
            curvature_angles = calculate_curvature(path, position_dictionary)
            interpolated_curvature = interpolate_curvature(path, position_dictionary, curvature_angles)
            auto_correlation = calculate_autocorrelation(interpolated_curvature)
            auto_correlations.append(auto_correlation)

    min_values, peaks, total_lengths = find_autocorrelation_minima(auto_correlations, path_lengths)
    length_bins, persistence_lengths = calculate_persistence_length(min_values, peaks, total_lengths)

    plt.figure("Persistence Length")
    plt.scatter(length_bins, persistence_lengths)
    plt.xlabel("total distance (right bin boundary)")
    plt.ylabel("persistence length")
    plt.title("Long Vessels (subsampled)")
    plt.show()

    # plot the simulated networks
    # plt.figure()
    # nx.draw(g, position_dictionary, node_size=5)
    # nx.draw_networkx_nodes(g, position_dictionary, nodelist=[root], node_color="red", node_size=7)
    # #plt.show()
    # plt.savefig("rand_walk_graph.png")


class Node:
    STEP_LEN = 1
    MEAN_ANGLE = 0
    TERMINATION_DISTANCE = 0.8
    BRANCHING_PROBABILITY = 0.2
    PATH_NUM = 0

    def __init__(self, x=0, y=0, heading=0, previous_Node=None, path_name=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.previous_Node = previous_Node
        self.path_name = path_name
        return

    def get_coordinates(self):
        coordinate_tuple = (self.x, self.y)
        return coordinate_tuple

    # Returns a list of tuples (can have only one element)
    # def grow_from_here(self, graph):
    #     tup_list = []
    #     new_node = self.generate_new_node()
    #     node_placed = False
    #     if not new_node.is_too_close(graph):
    #         node_placed = True
    #         appended_tuple = (self, new_node)
    #         tup_list.append(appended_tuple)
    #     if self.branches():
    #         branched_node = self.generate_new_node()
    #         if node_placed:
    #             Node.PATH_NUM += 1
    #             branched_node.path_name = Node.PATH_NUM
    #         if not branched_node.is_too_close(graph):
    #             appended_tuple = (self, branched_node)
    #             tup_list.append(appended_tuple)
    #     return tup_list

    def grow_from_here(self, graph):
        tup_list = []
        if self.branches():
            branched_node1 = self.generate_new_node()
            node_placed = False
            if not branched_node1.is_too_close(graph):
                node_placed = True
                appended_tuple = (self, branched_node1)
                tup_list.append(appended_tuple)
            branched_node2 = self.generate_new_node()
            if node_placed:
                Node.PATH_NUM += 1
                branched_node2.path_name = Node.PATH_NUM
            if not branched_node2.is_too_close(graph):
                appended_tuple = (self, branched_node2)
                tup_list.append(appended_tuple)
        else:
            new_node = self.generate_new_node()
            if not new_node.is_too_close(graph):
                if abs(new_node.heading - self.heading) < 0.1:
                    graph.remove_node(self)
                    tup_list.append((self.previous_Node, new_node))
                    new_node.previous_Node = self.previous_Node
                else:
                    tup_list.append((self, new_node))
        return tup_list

    def generate_new_node(self):
        new_angle = np.random.normal(Node.MEAN_ANGLE, 0.629)
        angle = new_angle + self.heading
        v, h = np.sin(angle) * Node.STEP_LEN, np.cos(angle) * Node.STEP_LEN
        new_x = self.x + h
        new_y = self.y + v
        new_heading = angle
        return Node(new_x, new_y, new_heading, self, self.path_name)

    def is_too_close(self, graph):
        points = []
        for node in graph.nodes():
            points.append(node.get_coordinates())
        return self.check_distance(points) != 0

    def check_distance(self, points):
        point_of_interest = self.get_coordinates()
        if point_of_interest not in points:
            points.append(point_of_interest)

        # points needs to be a list of tuples, and point of interest needs to be a tuple present in points
        index = points.index(point_of_interest)

        # the point of interest is the last point in the points list
        # calculate the distance between the point of interest and every other point
        dists = squareform(pdist(points))[index]

        close_points = list(dists < Node.TERMINATION_DISTANCE)

        counter = close_points.count(True)

        return counter - 1

    def branches(self):
        return np.random.random() < Node.BRANCHING_PROBABILITY


def find_path(graph, start, branch_nodes):
    """
    Function for parsing graphs and extracting paths that end at branch nodes.

    :param graph: networkx graph containing the edges, nodes, and edge diameter (optional)
    :param start: start node = where to start the path
    :param branch_nodes: a list of the nodes with more than two connections

    :return valid_paths: a nested list of the parsed paths
    """

    queue = list()  # maintain a queue of paths
    valid_paths = list()  # store the parsed paths
    queue.append([start])  # push the first path into the queue

    # for path in queue:
    while queue:

        path = queue.pop()  # get the first path from the queue
        node = path[-1]  # get the last node from the path

        # enumerate all adjacent nodes in path
        for adjacent in graph[node]:

            # if the adjacent node is not already in the path:
            if adjacent not in path:

                # if the adjacent node is not a node with more than two connections,
                # construct a new path and push it into the queue
                if adjacent not in branch_nodes:
                    queue.append(path + [adjacent])

                # else, you've found a valid path
                else:
                    valid_paths.append(path + [adjacent])

    return valid_paths


def find_all_paths(graph):
    """
    Function for parsing graphs. It uses the find_path function to find all the paths in the graph.

    :param graph: networkx graph containing the edges, nodes, and edge diameter (optional)

    :return paths: a nested list of the parsed paths
    """

    branch_nodes = [node for node, degree in dict(graph.degree()).items() if degree > 2]
    boundary_nodes = [node for node, degree in dict(graph.degree()).items() if degree == 1]

    bb_nodes = branch_nodes + boundary_nodes

    paths = list()
    for node in bb_nodes:
        for path in find_path(graph, node, branch_nodes):
            if path[::-1] not in paths:
                paths.append(path)

    return paths


def calculate_distance(pos, x, y):
    """
    A function that calculates the distance between two points.

    :param pos: a dictionary of the position of the two points
    :param x, y: the two different points

    :return: the euclidean distance between the points
    """

    return sqrt((pos[x][0] - pos[y][0]) ** 2 + (pos[x][1] - pos[y][1]) ** 2)


def euclidean_length(path, position_dictionary):
    """
    Calculates the shortest distance between the start and end nodes of a path.

    :param path: the path of interest
    :param position_dictionary: the x,y coordinates of the points

    :return: The shortest distances of the path.
    """
    return calculate_distance(position_dictionary, path[0], path[-1])


def arc_length(path, position_dictionary):
    """
    Calculates the total length of a path.

    :param path: the path of interest
    :param position_dictionary: the x,y coordinates of the points

    :return arc_len: the total length of the path
    """

    arc_len = 0
    pt2 = path[0]

    for x in range(1, len(path)):
        pt1 = pt2
        pt2 = path[x]
        arc_len += calculate_distance(position_dictionary, pt1, pt2)

    return arc_len


def all_arc_lengths(paths, position_dictionary):
    """
    Calculates the total lengths of all paths. Used for plotting.

    :param paths: a list of all the paths
    :param position_dictionary: the x,y coordinates of the points

    :return: a list of all the total lengths
    """

    all_arc_len = list()
    for path in paths:
        all_arc_len.append(arc_length(path, position_dictionary))

    return all_arc_len


def tortuosity_index(paths, position_dictionary):
    """
    Calculates the tortuosity of a path as the ratio between the total and euclidean length. It uses the
    arc_length and euclidean_length functions.

    :param paths: a list of all the paths
    :param position_dictionary: the x,y coordinates of the points

    :return tortuosity: the tortuosity indices of each path
    """
    tortuosity = list()
    for path in paths:
        euclidean_len = euclidean_length(path, position_dictionary)
        arc_len = arc_length(path, position_dictionary)
        tortuosity.append(arc_len / euclidean_len)

    return tortuosity


def calculate_curvature(path, position_dictionary):
    """
    Calculates the path's curvature as the angle divided by the average length of the steps that subtend the angle.

    :param path: a path to calculate the curvature for
    :param position_dictionary: the x,y coordinates of the points

    :return norm_theta_list: a list of the curvature
    """

    maxstep = len(path) - 2

    norm_theta_list = list()  # a list containing the curvature values

    for step_number in range(0, maxstep):  # all steps in path ***(while statement?)

        # a, b, c are the three points (a is start of normal, b is midpoint, c is end of vector)
        a = position_dictionary[path[step_number]]
        b = position_dictionary[path[step_number + 1]]
        c = position_dictionary[path[step_number + 2]]

        # u = vector from a to b, v = vector from b to c
        u = (b[0] - a[0], b[1] - a[1])
        v = (c[0] - b[0], c[1] - b[1])

        cross_product = u[0] * v[1] - u[1] * v[0]  # cross product between u and v

        # a product between the length of u and the length of v
        mods = (np.sqrt(u[0] ** 2 + u[1] ** 2)) * (np.sqrt(v[0] ** 2 + v[1] ** 2))

        # the average length of the two vectors
        avg_len = ((np.sqrt(u[0] ** 2 + u[1] ** 2)) + (np.sqrt(v[0] ** 2 + v[1] ** 2))) / 2

        sin_theta = cross_product / mods
        theta = (np.arcsin(sin_theta)) * (180 / np.pi)  # the angle in degrees

        # the angle normalized by the average length of the two edges that subtend the angle
        curvature = theta / avg_len

        norm_theta_list.append(curvature)

    return norm_theta_list


def arc_len_at_node(path, position_dictionary):
    """
    Calculates the arc length at each node in the path.

    :param path: path of interest
    :param position_dictionary: the x,y coordinates of the points

    :return arc_len: a list of the arc lengths at each node
    """
    # a list that stores the arc length from the start node to the current node we are at
    arc_len = list()
    total_distance = 0
    pt2 = path[0]

    # calculating the arc length for each node in path
    for i in range(1, len(path)):
        pt1 = pt2
        pt2 = path[i]
        total_distance += calculate_distance(position_dictionary, pt1, pt2)
        arc_len.append(total_distance)

    return arc_len


def interpolate_curvature(path, position_dictionary, curvature_angles, number_of_points=30):
    """
    Function for interpolating the curvature of a path. Given the path curvature as a function of arc length,
    interpolate_curvature just adds more points to curvature and arc length, making the curvature smoother.

    :param path: a path to calculate the curvature for
    :param position_dictionary: the x,y coordinates of the points
    :param number_of_points: an integer of the number of points for the interpolation
    :param curvature_angles: norm_theta_list from the calculate_curvature function

    :return interpolated_arclength: a list of the interpolated arc lengths
    :return interpolated_curvature: a list of the interpolated curvature
    """

    # don't take the last value so that arc_len and norm_theta_list are of equal length
    arc_len = arc_len_at_node(path, position_dictionary)[:-1]

    # the given arc length of the path and the arc length of the interpolation should have equal total length
    interp_arc_len = np.linspace(arc_len[0], arc_len[-1], number_of_points)  # arc length points

    interpolated_arclength = list()
    interpolated_curvature = list()

    for step in range(len(arc_len) - 1):

        # point a and point b from arc length
        arclength_pta = arc_len[step]
        arclength_ptb = arc_len[step + 1]

        # point a and point b from curvature
        curv_pta = curvature_angles[step]
        curv_ptb = curvature_angles[step + 1]

        # find the slope of the line connecting (arclength_pta,Ka) and (Sb,Kb)
        slope = (curv_ptb - curv_pta) / (arclength_ptb - arclength_pta)

        # find the interpolated curvature points
        for point in interp_arc_len:

            # if point falls between arclength_pta and arclength_ptb, proceed
            if arclength_pta <= point <= arclength_ptb:

                # the interpolated curvature point needs to be on the line connecting
                # (arclength_pta, curv_pta) and (arclength_ptb, curv_ptb), so we use the calculated slope from above
                interp_pt = curv_pta + slope * (point - arclength_pta)

                # append to the coordinates and interpolated arc length and curvature lists
                interpolated_arclength.append(point)
                interpolated_curvature.append(interp_pt)

            elif point > arclength_ptb:  # if point is bigger than arclength_ptb, then break the for loop
                break

    return interpolated_curvature


def calculate_autocorrelation(interpolated_curvature, number_of_points=30):
    """
    Calculates the auto correlation of the curvature of a path.

    :param number_of_points: default value is 30; also used for interpolating the curvature
    :param interpolated_curvature: the output of the interpolate_curvature function

    :return truncated_correlation: the normalized and truncated auto correlation
    """

    n = len(interpolated_curvature)

    # a list of the auto correlations
    auto_correlation = np.correlate(interpolated_curvature, interpolated_curvature, mode='full')

    # a list containing integers from 1 to n and back to 1
    norm_list = np.append(np.arange(1, n + 1, 1), (-1) * np.arange(-n + 1, 0, 1))

    # the magnitude of the autocorrelation is partly due to how many points it has been
    # summed over. so, we need to divide by the number of points it has summed over in
    # order to get rid of this inherent bias
    normed_correlation = auto_correlation / norm_list

    # the first (and probably biggest) autocorrelation value
    start_height = normed_correlation[number_of_points - 1]

    # normalize the autocorrelation by the start value
    fully_normed_correlation = normed_correlation / start_height

    # discard the mirrored part and last third of the autocorrelation
    truncated_correlation = fully_normed_correlation[number_of_points - 1: -int(number_of_points / 3)]

    return truncated_correlation


def find_autocorrelation_minima(auto_correlations, path_lengths, number_of_points=30):
    """
    Finds the minimum value of each auto correlation. It only considers the paths where the minimum goes below zero.

    :param auto_correlations: a list of the auto correlation for each path
    :param path_lengths: a list of the total lengths of each path
    :param number_of_points: default value is 30; also used for interpolating the curvature

    :return min_values: a list of all the minimum values of the auto correlations
    :return peaks: a list of the distance along the path where the minimum value occurs
    :return total_length: a list of the path lengths
    """
    min_values = list()  # storing the minimum values of the auto correlation for each path
    peaks = list()  # a list of the distance where the minimum occurs
    total_lengths = list()  # a list of the total length of the paths where a minimum was found

    for i in range(len(auto_correlations)):

        # find the first minimum value of the autocorrelation, if it's smaller than 0
        peak, min_value = find_peaks(-auto_correlations[i], height=0)

        # if the minimum value exists, then:
        if np.any(min_value["peak_heights"]):
            # append the minimum value to a list
            min_values.append(-min_value["peak_heights"][0])

            # append the length at which the minimum occurs to a list
            peaks.append(peak[0] * path_lengths[i] / number_of_points)

            # append the length of this path to a list
            total_lengths.append(path_lengths[i])

    return min_values, peaks, total_lengths


def calculate_persistence_length(min_values, peaks, total_lengths, length_bin=50):
    """
    Calculates the persistence lengths of all of the paths. First, it splits the paths in bins depending on their
    total length. Then, for each bin: 1) multiply the distance where the minimum occurs with the value of the minimum
    for each path, 2) sum these weighted peaks for each path in the bin and divide by the sum of the weights.

    :param min_values: the minimum values of the auto correlation
    :param peaks: the distance where the minimum occurs
    :param total_lengths: the total length of the paths considered
    :param length_bin: size of bins; default is 50

    :return length_bins: the right boundary of the bins of the total lengths
    :return persistence length: a list of the persistence lengths for each bin
    """
    persistence_length = list()
    length_bins = list()

    # floor and ceiling of min and max total length values, respectively
    # will use this for finding the total length bins
    max_length = np.ceil(max(total_lengths))
    min_length = np.floor(min(total_lengths))

    # find number of bins
    no_bins = int(np.ceil((max_length - min_length) / length_bin))

    for j in range(no_bins):

        weighted_peaks = list()
        min_values_list = list()

        for i in range(len(total_lengths)):

            # Split the total length of the vessels in bins of length 50
            if min_length + j * length_bin < total_lengths[i] <= min_length + (j + 1) * length_bin:
                # if total length falls in that range, then take the peak and weigh it based on its min_value
                weighted_peaks.append(peaks[i] * min_values[i])
                min_values_list.append(min_values[i])

        if len(min_values_list) != 0:
            persistence_length.append(sum(weighted_peaks) / sum(min_values_list))
            length_bins.append(min_length + (j + 1) * length_bin)

    return length_bins, persistence_length


def follow_path(node):
    """
    Follows the path as it grows. So, it splits the parent paths from the daughter paths.
    When a path branches, the new branch gets a different name, signifying the start of a new path.
    All the nodes with the same name belong to the same path.
    The function is recursive - it is meant to start with a boundary node and go to each node's previous
    node until it finds a node with a different name.

    :param node: the node from which to start following the path (usually a boundary node)

    :return path: the extracted path
    """
    if node.previous_Node is None:
        path = [node]
        return path

    elif node.previous_Node.path_name != node.path_name:
        path = [node]
        return path

    else:
        path = follow_path(node.previous_Node)
        path.append(node)
        return path


def get_all_follow_paths(graph, root):
    """
    Get all the follow paths from the follow_path function.

    :param graph: necessary for finding the boundary nodes
    :param root: the node the whole graph grows from. The end node of its path will be a boundary
    node, and we don't want to get duplicated paths, so we remove it.

    :return paths: a list of all the parsed paths
    """

    paths = list()
    boundary_nodes = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    if root in boundary_nodes:
        boundary_nodes.remove(root)
    for node in boundary_nodes:
        path = follow_path(node)
        if len(path) > 1:
            paths.append(path)

    return paths


main()
