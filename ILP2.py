import pulp


def my_hardcoded_solver():
    # Step 1: Create the optimization model
    model = pulp.LpProblem("Core", pulp.LpMinimize)

    # Step 2: Create PuLP variables (not dictionaries!)
    # For our hardcoded problem, we need specific variables:

    # Study time variables
    linear_algebra = pulp.LpVariable("linear_algebra", lowBound=0)  # x1
    data_structures = pulp.LpVariable("data_structures", lowBound=0)  # x2

    # Coursework time variables
    ds_assignment = pulp.LpVariable("ds_assignment", lowBound=0)  # y1

    # Slack variable
    slack = pulp.LpVariable("slack", lowBound=0)  # s

    # Total available hours (this is just a number)
    total_hours = 40.0

    model += linear_algebra + data_structures + ds_assignment + slack == total_hours

    model += linear_algebra >= 2
    model += data_structures >= 2.5
    model += ds_assignment >= 14.1

    objective = 0.25 * linear_algebra + 0.2 * data_structures + 0.9 * ds_assignment + 0.1 * slack

    model += objective

    status = model.solve()

    if status == pulp.LpStatusOptimal:
        result = {
            'feasible' : True,
            'linearAlgebra' : linear_algebra.varValue,
            'data structures' : data_structures.varValue,
            'ds assignement' : ds_assignment.varValue,
            'slack' : slack.varValue
        }
        for key, value in result.items():
            if not key == 'feasible':
                print(f"{key}, {value}")
        return result

    else:
        print("no solution found")
        return { 'feasible': False , 'status': pulp.LpStatus[status]}




    # TODO: Add constraints here
    # TODO: Add objective here
    # TODO: Solve here

    return "in progress"


if __name__ == "__main__":
    result = my_hardcoded_solver()
    print(result)


# This IS the core - the mathematical heart:
def solve_optimization_problem():
    """
    Core: min ∑(w_i * x_i) + ∑(u_j * y_j) + α * s
    s.t. ∑x_i + ∑y_j + s ≤ H
         x_i ≥ min_i * φ(perf_i)  ∀i
         y_j ≥ est_j * β_j        ∀j
         x_i, y_j, s ≥ 0
    """
    # This is where everything happens
    pass