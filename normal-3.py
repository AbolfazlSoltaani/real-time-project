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
    resources=resources, task_count=num_tasks, ratio=0, utilization_ub=2
)
num_resources = 3
resources = generate_resources(resource_count=num_resources)

speedup_factor_list = [1, 1.5, 1.6, 1.7, 2, 10, 100]
for speedup_factor in speedup_factor_list:
    cedf = CriticallyEDF(
        tasks=tasks, resources=resources, speedup_factor=speedup_factor, verbose=False
    )
    if cedf.schedule():
        cedf.visualize(
            show=False, save=True, filename=f"normal-speedup{speedup_factor}"
        )
        print(f"Schedulable with speedup factor {speedup_factor} and QoS is {cedf.quality_of_service()}")
        break
    else:
        print("Speedup factor failed ", speedup_factor)
        cedf.visualize(
            show=False, save=True, filename=f"normal-speedup{speedup_factor}"
        )
