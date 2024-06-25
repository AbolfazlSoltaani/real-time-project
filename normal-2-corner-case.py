from task import (
    Resource,
    Node,
    Edge,
    Task,
    Job,
    TaskType,
    generate_tasks,
    generate_resources,
    CriticallyEDF,
)
import random as rand

num_resources = rand.randint(1, 5)
resources = generate_resources(resource_count=num_resources)
num_tasks = rand.randint(5, 10)
tasks = generate_tasks(
    resources=resources, task_count=num_tasks, ratio=0.5, utilization_ub=2
)
num_resources = 3
resources = generate_resources(resource_count=num_resources)

node1 = [
    Node("T1-J1", wcet_hi=9, wcet_lo=5, resource=resources[0]),
    Node("T1-J2", wcet_hi=5, wcet_lo=3, resource=resources[2]),
    Node("T1-J3", wcet_hi=7, wcet_lo=4, resource=resources[1]),
    Node("T1-J4", wcet_hi=5, wcet_lo=3, resource=resources[0]),
    Node("T1-J5", wcet_hi=5, wcet_lo=3, resource=resources[1]),
]

nodes2 = [
    Node("T2-J1", wcet_hi=10, wcet_lo=10, resource=resources[2]),
    Node("T2-J2", wcet_hi=10, wcet_lo=10, resource=resources[1]),
    Node("T2-J3", wcet_hi=7, wcet_lo=7, resource=resources[0]),
    Node("T2-J4", wcet_hi=5, wcet_lo=5, resource=resources[0]),
    Node("T2-J5", wcet_hi=5, wcet_lo=5, resource=resources[2]),
]

task1 = Task(
    id="T1",
    period=32,
    nodes=node1,
    edges=[],
    wcet=31,
    task_type=TaskType.HC,
)

task2 = Task(
    id="T2",
    period=64,
    nodes=nodes2,
    edges=[],
    wcet=37,
    task_type=TaskType.LC,
)
task2.wcet = task2.get_wcet()

tasks = [task1, task2]

# speedup_factor_list = [1, 1.5, 1.6, 1.7, 2, 10, 100]
speedup_factor_list = [1.5]
for speedup_factor in speedup_factor_list:
    cedf = CriticallyEDF(
        tasks=tasks, resources=resources, speedup_factor=speedup_factor, verbose=True
    )
    if cedf.schedule():
        cedf.visualize(
            show=False, save=True, filename=f"normal-speedup{speedup_factor}"
        )
        print("Speedup factor passed ", speedup_factor)
        break
    else:
        print("Speedup factor failed ", speedup_factor)
        cedf.visualize(
            show=False, save=True, filename=f"normal-speedup{speedup_factor}"
        )
