import random as rand

def generate_resources(num_nodes, num_resources):
    resources = {}
    for i in range(1, num_resources + 1):
        resources[f"Resource_{i}"] = rand.randint(1, 16)
    return resources

def assign_resources_to_critical_sections(resources, num_critical_sections):
    critical_sections = {}
    for i in range(1, num_critical_sections + 1):
        critical_sections[f"Critical_Section_{i}"] = {}
        for resource, value in resources.items():
            critical_sections[f"Critical_Section_{i}"][resource] = rand.randint(0, value)
    return critical_sections

def main():
    num_nodes = rand.randint(1, 5)
    num_critical_sections = rand.randint(1, 16)
    num_resources = rand.randint(1, 5)

    resources = generate_resources(num_nodes, num_resources)
    critical_sections = assign_resources_to_critical_sections(resources, num_critical_sections)

    print("Generated Resources:")
    for resource, value in resources.items():
        print(f"{resource}: {value}")

    print("\nAssigned Resources to Critical Sections:")
    for section, resource_allocation in critical_sections.items():
        print(f"{section}: {resource_allocation}")

if __name__ == "__main__":
    main()
