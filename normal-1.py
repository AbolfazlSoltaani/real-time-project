from task import Resource, Node, Edge, Task, Job, TaskType, generate_tasks, generate_resources, CriticallyEDF
import random as rand

num_resources = rand.randint(1, 5)
resources = generate_resources(resource_count=num_resources)
num_tasks = rand.randint(5, 10)
tasks = generate_tasks(resources=resources, task_count=num_tasks, ratio=0.5)

cedf = CriticallyEDF(tasks=tasks, resources=resources, speedup_factor=1)
try:
    cedf.schedule()
    cedf.visualize(show=False, save=True, filename="normal-1")
except Exception as e:
    print(e)
    pass