import random as rand
import networkx
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

class TaskType(Enum):
    HC = "high-critical"
    LC = "low-critical"

@dataclass
class Resource:
    id: str

@dataclass
class Node:
    id: str
    wcet_hi: int
    wcet_lo: int
    resource: Resource = None

    def __str__(self) -> str:
        return f"Node: {self.id}, WCET_HI: {self.wcet_hi}, WCET_LO: {self.wcet_lo}, Resource: {self.resource.id if self.resource else None}"

@dataclass
class Edge:
    src: Node
    sink: Node

@dataclass
class Task:
    id: int
    period: int
    wcet : int
    nodes: list[Node]
    edges: list[Edge]
    task_type: TaskType

    def get_wcet(self) -> int:
        return sum([node.wcet_hi for node in self.nodes])
    
    def utilization(self) -> float:
        return self.get_wcet() / self.period
    
    def do_need_resource(self, resource: Resource) -> bool:
        return any([node.resource == resource for node in self.nodes])
    
    def nearest_deadline(self, time: int) -> int:
        return self.period - (time % self.period)

    def __str__(self) -> str:
        return f"Task: {self.id}, Period: {self.period}, WCET: {self.wcet}"

@dataclass
class Job:
    id: int
    task: Task
    arrival: int
    deadline: int

def FFT(mlg: int) -> tuple[list[int], list[tuple[int, int]]]:
    edges: list[tuple[int, int]] = []
    for i in range(2, 2 ** (mlg + 1)):
        edges.append((i // 2, i))

    label = 2**mlg
    for j in range(mlg):
        for k in range(2 ** (mlg)):
            edges.append((label + k, (2**mlg) + label + k))
            edges.append((label + k, ((label + k) ^ (2**j)) + (2**mlg)))
        label += 2**mlg
    nodes = [i + 1 for i in range(2 ** (mlg + 1) - 1 + mlg * (2 ** mlg))]
    return nodes, edges

def get_critical_path(nodes: list[Node], edges: list[Edge]) -> list[Node]:
    dp: dict[str, int] = {}
    degree_in: dict[str, int] = {}
    for edge in edges:
        degree_in[edge.sink.id] = degree_in.get(edge.sink.id, 0) + 1
    sources = [node for node in nodes if degree_in.get(node.id, 0) == 0]
    for node in sources:
        dp[node.id] = node.wcet_hi
    while sources:
        node = sources.pop()
        for edge in edges:
            if edge.src.id == node.id:
                degree_in[edge.sink.id] -= 1
                dp[edge.sink.id] = max(dp.get(edge.sink.id, 0), dp[node.id] + edge.sink.wcet_hi)
                if degree_in.get(edge.sink.id, 0) == 0:
                    sources.append(edge.sink)
    return max(dp.values())

resources_count = rand.randint(1, 5)
resources = [Resource(id=f"R{i + 1}") for i in range(resources_count)]

def generate_task(task_id: int, task_type: TaskType) -> Task:
    graph_nodes, graph_edges = FFT(1)
    nodes: list[Node] = []
    for node_id in graph_nodes:
        wcet_hi = rand.randint(5, 10)
        wcet_lo = wcet_hi if task_type == TaskType.LC else int(0.6 * wcet_hi)
        nodes.append(Node(id=f"T{task_id}-J{node_id}", wcet_hi=wcet_hi, wcet_lo=wcet_lo))

    edges: list[Edge] = []
    for edge in graph_edges:
        src = nodes[edge[0] - 1]
        sink = nodes[edge[1] - 1]
        edges.append(Edge(src=src, sink=sink))

    critical_path = get_critical_path(nodes, edges)
    x = 1
    while x < critical_path:
        x *= 2
    period = rand.choice([2 * x, 8 * x])
    wcet = sum([node.wcet_hi for node in nodes])

    critical_nodes_count = rand.randint(1, min(10, len(nodes)))
    critical_nodes = rand.choices(nodes, k=rand.randint(1, critical_nodes_count))
    for node in critical_nodes:
        node.resource = rand.choice(resources)

    return Task(id=f"T{task_id}", period=period, wcet=wcet, nodes=nodes, edges=edges, task_type=task_type)

# random number between 5 and 10
task_count = 10
tasks = []
for i in range(task_count):
    task_type = TaskType.HC if rand.random() < 0.5 else TaskType.LC
    tasks.append(generate_task(i, task_type))

# remove tasks with utilization > 1.0
trash_tasks = [task for task in tasks if task.utilization() > 1.0]
for trash_task in trash_tasks:
    tasks.remove(trash_task)

while sum([task.utilization() for task in tasks]) > 1:
    trash_task = rand.choice(tasks)
    tasks.remove(trash_task)

class CriticallyEDF:
    def __init__(self, tasks: list[Task], resources: list[Resource]):
        print("CriticallyEDF")
        for task in tasks:
            print(10 * "-")
            print(task)
            print(10 * "-")
            for node in task.nodes:
                print(node)
                
        
        self.tasks = tasks
        self.resources = resources

        self.counter = 1
        self.current_time = 0
        self.allocated_by: dict[str, Task] = {}
        self.execution_time: dict[str, int] = {}
        self.jobs: list[Job] = []
        self.in_degree: dict[str, int] = {}
        self.hyperperiod = np.lcm.reduce([task.period for task in tasks])

        self.fig, self.ax = plt.subplots()
        self.row: dict[str, int] = {}
        row_counter = 0
        for i, task in enumerate(tasks):
            for j, node in enumerate(task.nodes):
                self.row[node.id] = row_counter
                row_counter += 1

    def psi(self, resource: Resource, task: Task) -> int:
        result = math.inf
        for i, t in enumerate(self.tasks):
            if t == task:
                continue
            if t.do_need_resource(resource):
                result = min(result, t.nearest_deadline(self.current_time))

    def __resource_ceiling(self, resource: Resource) -> int:
        if not self.allocated_by.get(resource.id, None):
            return math.inf
        return self.current_time + self.psi(resource, self.allocated_by[resource.id])
    
    def __resource_request_deadline(self, resource: Resource) -> int:
        if not self.allocated_by.get(resource, None):
            return math.inf
        return self.allocated_by[resource.id].nearest_deadline(self.current_time) # TODO: check if this is correct

    def __system_ceiling(self) -> int:
        result = math.inf
        for resource in self.resources:
            result = min(result, self.__resource_ceiling(resource)) # TODO: after fixing `__resource_request_deadline`
        return result
    
    def __create_periodic_jobs(self):
        for task in self.tasks:
            if self.current_time % task.period == 0:
                job = Job(id=self.counter, task=task, arrival=self.current_time, deadline=self.current_time + task.period)
                self.jobs.append(job)
                self.counter += 1
                for node in task.nodes:
                    self.execution_time[node.id] = 0
                for edge in task.edges:
                    self.in_degree[edge.sink.id] = self.in_degree.get(edge.sink.id, 0) + 1

    def __execute_job(self, job: Job):
        selected_node = None
        for node in job.task.nodes:
            if self.in_degree.get(node.id, 0) > 0 or self.execution_time.get(node.id, 0) == node.wcet_lo:
                continue
            selected_node = node
            break
        if selected_node != None:
            print(f"Current Time: {self.current_time}, Executing Job: {job.id}, Task: {job.task.id}, Node: {selected_node.id}")
        if not selected_node:
            raise Exception("Not Schedulable 2")
        if selected_node.resource and self.allocated_by.get(selected_node.resource.id, None) not in [None, job.task]: 
            raise Exception("Not Schedulable 3")
        self.execution_time[selected_node.id] = self.execution_time.get(selected_node.id, 0) + 1
        self.ax.broken_barh([(self.current_time, 1)], (self.row[selected_node.id], 0.5), facecolors='gray')
        if self.execution_time[selected_node.id] == selected_node.wcet_lo:
            for edge in job.task.edges:
                if edge.src.id == selected_node.id:
                    self.in_degree[edge.sink.id] -= 1
            if selected_node.resource:
                self.allocated_by[selected_node.resource.id] = None
        for node in job.task.nodes:
            if self.execution_time.get(node.id, 0) < node.wcet_lo:
                return False
        return True
        
    def schedule(self):
        while self.current_time < self.hyperperiod:
            self.__create_periodic_jobs()
            if not self.jobs:
                self.current_time += 1
                continue

            job = min(self.jobs, key=lambda job: (job.task.task_type.value, job.deadline))
            if job.deadline >= self.__system_ceiling():
                raise Exception("Not Schedulable 1")
            if self.__execute_job(job):
                self.jobs.remove(job)
            
            self.current_time += 1

        self.visualize()
        return True 
    
    def visualize(self):
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Tasks')
        self.ax.set_title('Critically EDF')
        y_max = sum([len(task.nodes) for task in self.tasks])
        self.ax.set_xticks([i for i in range(0, self.hyperperiod, 10)])
        self.ax.set_yticks([i for i in range(y_max)])
        self.ax.set_yticklabels([node.id for task in self.tasks for node in task.nodes])
        self.ax.grid(True)
        plt.show()
    
cedf = CriticallyEDF(tasks, resources)
cedf.schedule()