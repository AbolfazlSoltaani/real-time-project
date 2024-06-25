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

    def map_to_color(self) -> str:
        if self.id == "R1":
            return 'red'
        if self.id == "R2":
            return 'blue'
        if self.id == "R3":
            return 'green'
        if self.id == "R4":
            return 'yellow'
        if self.id == "R5":
            return 'purple'
        return 'black'

@dataclass
class Node:
    id: str
    wcet_hi: int
    wcet_lo: int
    resource: Resource = None

    def get_exec_time(self, overrun: bool) -> int:
        return self.wcet_hi if overrun else self.wcet_lo

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
    active: bool = False
    quality: int = 100
    overrun: bool = False

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

def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources


def generate_task(task_id: int, task_type: TaskType, resources: list[Resource]) -> Task:
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

    period = rand.choice([x, 2 * x])
    wcet = sum([node.wcet_hi for node in nodes])

    # critical_nodes_count = rand.randint(1, min(10, len(nodes)))
    critical_nodes_count = len(nodes)
    # critical_nodes = rand.sample(nodes, k=rand.randint(1, critical_nodes_count))
    critical_nodes = nodes
    for node in critical_nodes:
        node.resource = rand.choice(resources)

    return Task(id=f"T{task_id}", period=period, wcet=wcet, nodes=nodes, edges=edges, task_type=task_type)

def generate_tasks(resources: list[Resource], task_count: int, ratio: float = 0.5, utilization_ub: int = 1) -> list[Task]:
    tasks = []
    for i in range(task_count):
        task_type = TaskType.HC if i < ratio * task_count else TaskType.LC
        tasks.append(generate_task(i, task_type, resources))

    # remove tasks with utilization > 1.0
    trash_tasks = [task for task in tasks if task.utilization() > 1.0]
    for trash_task in trash_tasks:
        tasks.remove(trash_task)

    while sum([task.utilization() for task in tasks]) > utilization_ub:
        trash_task = rand.choice(tasks)
        tasks.remove(trash_task)

    return tasks

class CriticallyEDF:
    def __init__(self, tasks: list[Task], resources: list[Resource], speedup_factor: int = 1, verbose: bool = False, overrun_chance: int = 0):
        if verbose:
            print("CriticallyEDF")
            for task in tasks:
                print(10 * "-")
                print(task)
                print(10 * "-")
                for node in task.nodes:
                    print(node)
                
        for task in tasks:
            for node in task.nodes:
                node.wcet_hi = node.wcet_hi // speedup_factor + 1
                node.wcet_lo = node.wcet_lo // speedup_factor + 1
            task.wcet = task.get_wcet()
        
        self.tasks = tasks
        self.resources = resources

        self.overrun_chance = overrun_chance

        self.counter = 1
        self.current_time = 0
        self.allocated_by: dict[str, Task] = {}
        self.execution_time: dict[str, int] = {}
        self.jobs: list[Job] = []
        self.done_jobs: list[Job] = []
        self.in_degree: dict[str, int] = {}
        self.hyperperiod = np.lcm.reduce([task.period for task in tasks])
        self.verbose = verbose

        self.fig, self.ax = plt.subplots()
        self.row: dict[str, int] = {}
        row_counter = 0
        for i, task in enumerate(tasks):
            for j, node in enumerate(task.nodes):
                self.row[node.id] = row_counter
                row_counter += 1

    def psi(self, resource: Resource, task: Task) -> int:
        result = math.inf
        for i, job in enumerate(self.jobs):
            if job.task == task:
                continue
            if job.task.do_need_resource(resource):
                result = min(result, job.task.nearest_deadline(self.current_time))
        return result

    def __resource_ceiling(self, resource: Resource) -> int:
        if not self.allocated_by.get(resource.id, None):
            return math.inf
        return self.current_time + self.psi(resource, self.allocated_by[resource.id])
    
    # def __resource_request_deadline(self, resource: Resource) -> int:
    #     if not self.allocated_by.get(resource, None):
    #         return math.inf
    #     return self.allocated_by[resource.id].nearest_deadline(self.current_time) # TODO: check if this is correct

    def __system_ceiling(self) -> int:
        result = math.inf
        for resource in self.resources:
            result = min(result, self.__resource_ceiling(resource)) # TODO: after fixing `__resource_request_deadline`
        return result
    
    def __create_periodic_jobs(self):
        for task in self.tasks:
            if self.current_time % task.period == 0:
                do_overrun = False if task.task_type == TaskType.LC or rand.randint(0, 100) >= self.overrun_chance else True
                job = Job(id=self.counter, task=task, arrival=self.current_time, deadline=self.current_time + task.period, active=False, overrun=do_overrun)
                self.jobs.append(job)
                self.counter += 1
                for node in task.nodes:
                    self.execution_time[node.id] = 0
                for edge in task.edges:
                    self.in_degree[edge.sink.id] = self.in_degree.get(edge.sink.id, 0) + 1

    def __execute_job(self, job: Job) -> bool:
        job.active = True
        selected_node = None
        for node in job.task.nodes:
            if self.in_degree.get(node.id, 0) > 0 or self.execution_time.get(node.id, 0) == node.get_exec_time(job.overrun):
                continue
            selected_node = node
            break
        if selected_node != None and self.verbose:
            print(f"Current Time: {self.current_time}, Executing Job: {job.id}, Task: {job.task.id}, Node: {selected_node.id}")
        if not selected_node:
            raise Exception("Not Schedulable 2")
        if selected_node.resource and self.allocated_by.get(selected_node.resource.id, None) not in [None, job.task]: 
            raise Exception("Not Schedulable 3")
        if selected_node.resource:
            self.allocated_by[selected_node.resource.id] = job.task
        self.execution_time[selected_node.id] = self.execution_time.get(selected_node.id, 0) + 1
        color = 'gray'
        if selected_node.resource:
            color = selected_node.resource.map_to_color()
        self.ax.broken_barh([(self.current_time, 1)], (self.row[selected_node.id], 0.5), facecolors=color)
        if self.execution_time[selected_node.id] == selected_node.get_exec_time(job.overrun):
            print(f"Node {selected_node.id} is Done")
            for edge in job.task.edges:
                if edge.src.id == selected_node.id:
                    self.in_degree[edge.sink.id] -= 1
            if selected_node.resource:
                self.allocated_by[selected_node.resource.id] = None
        if self.current_time > job.deadline:
            if job.task.task_type == TaskType.HC:
                raise Exception("Not Schedulable - Deadline Missed")
            job.quality = 100 - (self.current_time - job.deadline)
        for node in job.task.nodes:
            if self.execution_time.get(node.id, 0) < node.get_exec_time(job.overrun):
                return False
        return True
        
    def schedule(self):
        while self.current_time < self.hyperperiod or self.jobs:
            if self.current_time < self.hyperperiod:
                self.__create_periodic_jobs()
            if not self.jobs:
                self.current_time += 1
                continue
            
            job = None
            self.jobs.sort(key=lambda job: (job.task.task_type.value, job.deadline))
            for j in self.jobs:
                if j.deadline < self.__system_ceiling() or j.active:
                    job = j
                    break
            
            if not job:
                print("Not Schedulable - Deadlock")
                return False
            try:
                if self.__execute_job(job):
                    self.jobs.remove(job)
                    job.active = False
                    self.done_jobs.append(job)
            except Exception as e:
                if self.verbose:
                    print(e)
                return False
            
            self.current_time += 1

        if len(self.jobs) > 0:
            return False
        return True
    
    def quality_of_service(self) -> float:
        return sum([job.quality for job in self.done_jobs]) / len(self.done_jobs)
    
    def visualize(self, show: bool = False, save: bool = False, filename: str = None):
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Tasks')
        self.ax.set_title('Critically EDF')
        y_max = sum([len(task.nodes) for task in self.tasks])
        self.ax.set_xticks([i for i in range(0, self.hyperperiod, 4)] + [self.hyperperiod])
        self.ax.set_yticks([i for i in range(y_max)])
        self.ax.set_yticklabels([node.id for task in self.tasks for node in task.nodes])
        self.ax.grid(True)
        if show:
            plt.show()
        if save:
            self.fig.savefig(f"{filename}.png")
