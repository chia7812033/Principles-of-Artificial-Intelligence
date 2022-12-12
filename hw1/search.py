from collections import deque
import networkx as nx
from sklearn import neighbors
import numpy as np

loactions = ['Keelong', 'Taipei', 'Taoyuan', 'Hsinchu', 'Miaoli', 
            'Taichung','Changhua', 'Yunling', 'Chiayi', 'Tainan', 
            'Kaohsiung', 'Pingtung', 'Ilan', 'Nantou', 'Hualien', 
            'Taitung']

class Map():
    def __init__(self):
        self.map = nx.Graph()
        self.construct()
        self.v = self.map.number_of_nodes()

    def construct(self):
        self.map.add_edge(1,0, cost=30, time=20)
        self.map.add_edge(1,12, cost=65, time=40)
        self.map.add_edge(1,2, cost=55, time=35)
        self.map.add_edge(0,1, cost=30, time=20)
        self.map.add_edge(0,12, cost=65, time=45)
        self.map.add_edge(12,1, cost=65, time=40)
        self.map.add_edge(12,0, cost=65, time=45)
        self.map.add_edge(12,2, cost=100, time=70)
        self.map.add_edge(12,14, cost=110, time=110)
        self.map.add_edge(2,1, cost=55, time=35)
        self.map.add_edge(2,12, cost=100, time=70)
        self.map.add_edge(2,3, cost=45, time=45)
        self.map.add_edge(3,2, cost=45, time=45)
        self.map.add_edge(3,4, cost=35, time=35)
        self.map.add_edge(4,3, cost=35, time=35)
        self.map.add_edge(4,5, cost=45, time=40)
        self.map.add_edge(5,4, cost=45, time=40)
        self.map.add_edge(5,6, cost=30, time=24)
        self.map.add_edge(5,13, cost=50, time=50)
        self.map.add_edge(6,5, cost=30, time=24)
        self.map.add_edge(6,7, cost=35, time=35)
        self.map.add_edge(6,13, cost=75, time=45)
        self.map.add_edge(7,6, cost=35, time=35)
        self.map.add_edge(7,8, cost=35, time=35)
        self.map.add_edge(8,7, cost=35, time=35)
        self.map.add_edge(8,9, cost=45, time=40)
        self.map.add_edge(9,8, cost=45, time=40)
        self.map.add_edge(9,10, cost=45, time=30)
        self.map.add_edge(10,9, cost=45, time=30)
        self.map.add_edge(10,11, cost=22, time=20)
        self.map.add_edge(10,15, cost=150, time=140)
        self.map.add_edge(11,10, cost=22, time=20)
        self.map.add_edge(11,15, cost=130, time=120)
        self.map.add_edge(14,12, cost=110, time=110)
        self.map.add_edge(14,15, cost=170, time=170)
        self.map.add_edge(14,13, cost=250, time=240)
        self.map.add_edge(15,10, cost=150, time=140)
        self.map.add_edge(15,11, cost=130, time=120)
        self.map.add_edge(15,14, cost=170, time=170)
        self.map.add_edge(13,5, cost=50, time=50)
        self.map.add_edge(13,6, cost=75, time=45)
        self.map.add_edge(13,14, cost=250, time=240)

    def get_neighbors(self, n):
        return list(self.map.adj[n])

    def get_cost(self, start, end):
        return self.map[start][end]['cost']

    def get_time(self, start, end):
        return self.map[start][end]['time']

    def bfs(self, start, end):
        queue = deque()
        queue.append([start])   
        visited = set()  
        expanded = 0 
            
        while queue:
            current_path = queue.popleft()
            last_node = current_path[-1]
            neighbors = self.get_neighbors(last_node)
            expanded += 1

            if last_node == end: 
                cost = nx.path_weight(self.map, current_path, 'cost')
                self.print_path(current_path, cost)
                print('expaned nodes: %d' % expanded)
                return
        
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = current_path[:]
                    new_path.append(neighbor)
                    queue.append(new_path)
            visited.add(last_node)

    def dfs(self, start, end, path = [], visited = set(), expanded=0):
        path.append(start)
        visited.add(start)
        neighbors = self.get_neighbors(start)
        expanded += 1

        if start == end:
            cost = nx.path_weight(self.map, path, 'cost')
            self.print_path(path, cost)
            print('expaned nodes: %d' % expanded)
            return
        for neighbor in neighbors:
            if neighbor not in visited:
                self.dfs(neighbor, end, path, visited , expanded)
        path.pop() 
        return 

    def iddfs(self, start, end):
        depth = 1
        bottom_reached = False
        expanded = 0  
        while not bottom_reached:
            bottom_reached, expanded = self.IDDFS(start, end, 0, depth, expanded, [], set())
            depth += 1
        return 

    
    def IDDFS(self, start, end, current_depth, max_depth, expanded, path = [], visited = set()):

        path.append(start)
        visited.add(start)
        expanded += 1

        if start == end:
            cost = nx.path_weight(self.map, path, 'cost')
            self.print_path(path, cost)
            print('expaned nodes: %d' % expanded)
            return True, expanded

        neighbors = self.get_neighbors(start)
        if current_depth == max_depth:
            path.pop()
            if all(item in visited for item in neighbors):
                return True, expanded
            else:
                return False, expanded

        bottom_reached = False
        for neighbor in neighbors:
            if neighbor not in visited:
                if not bottom_reached:
                    bottom_reached, expanded = self.IDDFS(neighbor, end, current_depth + 1, max_depth, expanded, path, visited)

        path.pop()
        return bottom_reached, expanded

    def print_path(self, path, cost):
        for p in path:
            print(loactions[p], end=" ")
        print(cost)

    def real_shortest(self, s, e):
        print(nx.shortest_path(self.map, s, e, 'cost'))
        print(nx.shortest_path_length(self.map, s, e, 'cost'))
    
    def test(self, s, e):
        print('bsf')
        self.bfs(s, e)
        print('dfs')
        self.dfs(s, e, [], set(), 0)
        print('iddfs')
        self.iddfs(s, e)

a = Map()
a.test(3,0)
print()
a.test(0,3)
