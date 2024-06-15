import random as rand
import networkx
import numpy as np
import matplotlib.pyplot as plt


MAX_DEADLINE = 10000
MAX_PERIOD = 10000
MAX_EXECUTION_TIME = 10

class Resource:
    def __init__(self, id: int):
        self.id = id

    def __str__(self):
        return self.id

class Job:
    def __init__(self, id: int, critical: bool):
        self.id = id
        self.critical = critical
        self.wcet_low = rand.randint(1, MAX_EXECUTION_TIME)
        if critical:
            self.wcet_high = rand.randint(self.wcet_low, MAX_EXECUTION_TIME)
        self.resources: list[Resource] = []

    # TODO: check with TA how should this happens
    def create_critical_section(self, resource: list[Resource]) -> None:
        n = rand.randint(1, len(resources))
        self.resources = rand.sample(resources, n)


class Task:
    def __init__(
        self,
        id: int,
        deadline: int,
        period: int,
        jobs: list[Job],
        dependency_graph: list[tuple[int, int]],
    ):
        self.id = id
        self.deadline = deadline
        self.period = period
        self.jobs = jobs
        self.dependency_graph = dependency_graph

    @staticmethod
    def generate_task(id: int, critical: bool, resources: list[Resource]):
        m = rand.randint(1, 1)
        num_jobs = (m + 2) * (2**m) - 1
        jobs: list[Job] = []
        for i in range(num_jobs):
            j = Job(i + 1, critical)
            jobs.append(j)
        if critical:
            s = sum([j.wcet_high for j in jobs])
        else:
            s = sum([j.wcet_low for j in jobs])
        period = rand.randint(s, MAX_PERIOD)
        deadline = rand.randint(s, period)
        dependency_graph = Task.FFT(m)

        num_critical_section = rand.randint(1, min(num_jobs, 16))
        critical_job_ids = rand.sample(range(1, num_jobs + 1), num_critical_section)
        for job_id in critical_job_ids:
            job = jobs[job_id - 1]
            job.create_critical_section(resources)

        return Task(id, deadline, period, jobs, dependency_graph)

    @staticmethod
    def FFT(mlg: int):
        edges: list[tuple[int, int]] = []
        for i in range(2, 2 ** (mlg + 1)):
            edges.append((i // 2, i))

        label = 2**mlg
        for j in range(mlg):
            for k in range(2 ** (mlg)):
                edges.append((label + k, (2**mlg) + label + k))
                edges.append((label + k, ((label + k) ^ (2**j)) + (2**mlg)))
            label += 2**mlg

        return edges

class CEDF:
    def __int__(self, tasks: list[Task], resources: list[Resource]):
        self.tasks = tasks
        self.resources = resources
        self.is_free = [True] * len(resources)

    def psi(self, resource: Resource):
        pass

    def pi(self, resource: Resource, time: int):
        pass
    
    def schedule(self):
        pass


def show(t: Task):
    g = networkx.DiGraph()
    print(t.dependency_graph)
    g.add_nodes_from([job.id for job in t.jobs])
    for src, sink in t.dependency_graph:
        g.add_edge(src, sink)

    levels = np.zeros(len(t.jobs) + 1, dtype=int)
    for src, sink in t.dependency_graph:
        if levels[sink] == 0:
            levels[sink] = levels[src] + 1

    max_level = int(np.max(levels))
    levels_width = np.zeros(max_level + 1, dtype=int)
    for level in levels:
        levels_width[level] += 1
    levels_width[0] = 1

    max_width = max(levels_width)
    layout = networkx.spring_layout(g)
    x_array = np.zeros(max_level + 1, dtype=int)
    for i in range(len(t.jobs)):
        level = levels[i + 1]
        x: int | None = None
        x = (
            (x_array[level] - ((levels_width[level] - 1) / 2))
            * (max_width / levels_width[level])
            * 10
        )

        layout[i + 1] = (x, -10 * level)
        x_array[level] += 1

    node_attributes = {
        "node_color": "pink",
        "node_size": 400,
        "font_size": 12,
        "font_color": "black",
    }
    edge_attributes = {
        "edge_color": "gray",
        "width": 1,
        "arrows": True,
        "arrowstyle": "-|>",
        "arrowsize": 12,
    }
    plt.figure(figsize=(6, 6))
    networkx.draw_networkx(
        g, layout, with_labels=True, **node_attributes, **edge_attributes
    )

    plt.title(f"Task {t.id}")

    plt.show()

def generate_resources(num_resources: int) -> list[Resource]:
    resources: list[Resource] = []
    for i in range(num_resources):
        resources.append(Resource(i + 1))
    return resources


def generate_tasks(num_lc_tasks: int, num_hc_tasks: int, resources: list[Resource]) -> list[Task]:
    tasks: list[Task] = []
    tasks = [Task.generate_task(i + 1, True, resources) for i in range(num_hc_tasks)] + [
        Task.generate_task(i + 1 + num_lc_tasks, False, resources) for i in range(num_lc_tasks)
    ]
    return tasks


ratio = 1
num_resources = 3
num_lc_tasks = 1
num_hc_tasks = num_lc_tasks * ratio

# num_resources = rand.randint(1, 5)
# num_lc_tasks = rand.randint(5, 10)
# num_hc_tasks = num_lc_tasks * ratio

resources = generate_resources(num_resources)
tasks = generate_tasks(num_lc_tasks, num_hc_tasks, resources)
