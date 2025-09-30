"""
Elegant Resource Allocation using Mathematical Formulation
Clean separation: Helper Functions → Constraints → Objective → Solver
"""

import pulp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


# ================================
# DATA STRUCTURES
# ================================

@dataclass
class Subject:
    id: int
    name: str
    min_hours: float  # min_i
    priority: int  # p_i (1-5)
    performance: float  # perf_i (0-1)


@dataclass
class Coursework:
    id: int
    name: str
    estimated_hours: float  # est_j
    urgency: float  # u_j (0-1)
    complexity: float  # c_j (1-5)


@dataclass
class Parameters:
    total_hours: float  # H
    mode: str  # optimization mode
    alpha: float = 0.1  # slack penalty weight


@dataclass
class Solution:
    feasible: bool
    x: Dict[int, float]  # study time per subject
    y: Dict[int, float]  # coursework time per coursework
    s: float  # slack time
    objective_value: float


# ================================
# HELPER FUNCTIONS φ(perf_i) and β_j
# ================================

def phi(performance: float) -> float:
    """Performance adjustment function φ(perf_i) """
    if performance < 0.6:
        return 1.3
    elif performance <= 0.8:
        return 1.0
    else:
        return 0.7


def beta(complexity: float) -> float:
    """Complexity buffer function β_j"""
    return 1.0 + 0.2 * (complexity - 1) / 4


# ================================
# CONSTRAINT FUNCTIONS
# ================================

def add_time_budget_constraint(model: pulp.LpProblem, x: Dict, y: Dict, s: pulp.LpVariable, H: float):
    """
    Time Budget Constraint: ∑x_i + ∑y_j + s ≤ H // here the given should not be exceeded.
    """
    total_time = pulp.lpSum([x[i] for i in x]) + pulp.lpSum([y[j] for j in y]) + s
    model += total_time == H, "TimeBudget"


def add_subject_minimum_constraints(model: pulp.LpProblem, x: Dict, subjects: List[Subject]):
    """
    Subject Minimum Constraints: x_i ≥ min_i × φ(perf_i) ∀i ∈ I
    """
    for subject in subjects:
        adjusted_minimum = subject.min_hours * phi(subject.performance)
        model += x[subject.id] >= adjusted_minimum, f"SubjectMin_{subject.id}"


def add_coursework_minimum_constraints(model: pulp.LpProblem, y: Dict, coursework: List[Coursework]):
    """
    Coursework Minimum Constraints: y_j ≥ est_j × β_j ∀j ∈ J
    """
    for cw in coursework:
        adjusted_estimate = cw.estimated_hours * beta(cw.complexity)
        model += y[cw.id] >= adjusted_estimate, f"CourseworkMin_{cw.id}"


def add_non_negativity_constraints(model: pulp.LpProblem, x: Dict, y: Dict, s: pulp.LpVariable):
    """
    Non-negativity Constraints: x_i, y_j, s ≥ 0
    """

    for i in x:
        model += x[i] >= 0, f"NonNeg_x_{i}"
    for j in y:
        model += y[j] >= 0, f"NonNeg_y_{j}"
    model += s >= 0, "NonNeg_s"


# ================================
# OBJECTIVE FUNCTIONS
# ================================

def objective_harm_reduction(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                             coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """
    Harm Reduction Mode: min ∑x_i
    """
    return pulp.lpSum([x[i] for i in x])


def objective_balanced(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                       coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """
    Balanced Mode: min ∑(x_i/p_i) + ∑(u_j × y_j) + α × s
    """
    study_term = pulp.lpSum([x[subject.id] / subject.priority for subject in subjects])
    coursework_term = pulp.lpSum([cw.urgency * y[cw.id] for cw in coursework])
    slack_term = params.alpha * s

    return study_term + coursework_term + slack_term


def objective_perfection(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                         coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """
    Perfection Mode: min ∑(x_i/(p_i × perf_i)) + ∑(0.5 × y_j)
    Simplified version of learning efficiency maximization
    """
    study_term = pulp.lpSum([x[subject.id] / (subject.priority * subject.performance + 0.1)
                             for subject in subjects])
    coursework_term = pulp.lpSum([0.5 * y[cw.id] for cw in coursework])

    return study_term + coursework_term


# ================================
# DECISION VARIABLE CREATION
# ================================

def create_variables(subjects: List[Subject], coursework: List[Coursework]) -> Tuple[Dict, Dict, pulp.LpVariable]:
    """
    Create decision variables: x_i, y_j, s
    """
    # Study time variables x_i
    x = {subject.id: pulp.LpVariable(f"x_{subject.id}", lowBound=0, cat='Continuous')
         for subject in subjects}

    # Coursework time variables y_j
    y = {cw.id: pulp.LpVariable(f"y_{cw.id}", lowBound=0, cat='Continuous')
         for cw in coursework}

    # Slack time variable s
    s = pulp.LpVariable("s", lowBound=0, cat='Continuous')

    return x, y, s


# ================================
# MAIN SOLVER FUNCTION
# ================================

def solve_resource_allocation(subjects: List[Subject], coursework: List[Coursework],
                              params: Parameters) -> Solution:
    """
    Elegant main solver function that mirrors mathematical formulation
    """
    # Create the model
    model = pulp.LpProblem("ResourceAllocation", pulp.LpMinimize)

    # Create decision variables
    x, y, s = create_variables(subjects, coursework)

    # Add constraints
    add_time_budget_constraint(model, x, y, s, params.total_hours)
    add_subject_minimum_constraints(model, x, subjects)
    add_coursework_minimum_constraints(model, y, coursework)
    add_non_negativity_constraints(model, x, y, s)

    # Set objective function based on mode
    objective_functions = {
        "harm_reduction": objective_harm_reduction,
        "balanced": objective_balanced,
        "perfection": objective_perfection
    }

    if params.mode not in objective_functions:
        raise ValueError(f"Unknown optimization mode: {params.mode}")

    objective = objective_functions[params.mode](x, y, s, subjects, coursework, params)
    model += objective

    # Solve the model
    status = model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract solution
    if status == pulp.LpStatusOptimal:
        return Solution(
            feasible=True,
            x={i: x[i].varValue for i in x},
            y={j: y[j].varValue for j in y},
            s=s.varValue,
            objective_value=pulp.value(model.objective)
        )
    else:
        return Solution(
            feasible=False,
            x={}, y={}, s=0.0,
            objective_value=float('inf')
        )


# ================================
# UTILITY FUNCTIONS
# ================================

def print_mathematical_formulation(subjects: List[Subject], coursework: List[Coursework], params: Parameters):
    """
    Print the mathematical formulation for verification
    """
    print("MATHEMATICAL FORMULATION")
    print("=" * 50)

    # Variables
    print("Decision Variables:")
    for subject in subjects:
        print(f"  x_{subject.id} = hours for {subject.name}")
    for cw in coursework:
        print(f"  y_{cw.id} = hours for {cw.name}")
    print(f"  s = slack time")

    # Objective
    print(f"\nObjective ({params.mode} mode):")
    if params.mode == "harm_reduction":
        variables = " + ".join([f"x_{s.id}" for s in subjects])
        print(f"  min {variables}")
    elif params.mode == "balanced":
        study_terms = " + ".join([f"x_{s.id}/{s.priority}" for s in subjects])
        cw_terms = " + ".join([f"{cw.urgency}*y_{cw.id}" for cw in coursework])
        print(f"  min {study_terms} + {cw_terms} + {params.alpha}*s")

    # Constraints
    print(f"\nConstraints:")

    # Time budget
    all_vars = [f"x_{s.id}" for s in subjects] + [f"y_{cw.id}" for cw in coursework] + ["s"]
    print(f"  {' + '.join(all_vars)} ≤ {params.total_hours}")

    # Subject minimums
    for subject in subjects:
        adj_min = subject.min_hours * phi(subject.performance)
        print(f"  x_{subject.id} ≥ {adj_min:.2f}")

    # Coursework minimums
    for cw in coursework:
        adj_est = cw.estimated_hours * beta(cw.complexity)
        print(f"  y_{cw.id} ≥ {adj_est:.2f}")

    print("  All variables ≥ 0")


def print_solution(solution: Solution, subjects: List[Subject], coursework: List[Coursework]):
    """
    Print solution in a clean format
    """
    print("\nSOLUTION")
    print("=" * 50)

    if not solution.feasible:
        print("❌ INFEASIBLE - No solution exists")
        return

    print("✅ FEASIBLE SOLUTION FOUND")
    print(f"Objective Value: {solution.objective_value:.4f}")

    print(f"\nStudy Time Allocation:")
    total_study = 0
    for subject in subjects:
        hours = solution.x[subject.id]
        total_study += hours
        print(f"  {subject.name}: {hours:.2f} hours")

    print(f"\nCoursework Time Allocation:")
    total_coursework = 0
    for cw in coursework:
        hours = solution.y[cw.id]
        total_coursework += hours
        print(f"  {cw.name}: {hours:.2f} hours")

    print(f"\nSlack Time: {solution.s:.2f} hours")
    print(f"\nTotals:")
    print(f"  Study: {total_study:.2f} hours")
    print(f"  Coursework: {total_coursework:.2f} hours")
    print(f"  Slack: {solution.s:.2f} hours")
    print(f"  Total: {total_study + total_coursework + solution.s:.2f} hours")


def calculate_minimum_requirements(subjects: List[Subject], coursework: List[Coursework]) -> float:
    """
    Calculate total minimum requirements
    """
    study_min = sum(subject.min_hours * phi(subject.performance) for subject in subjects)
    coursework_min = sum(cw.estimated_hours * beta(cw.complexity) for cw in coursework)
    return study_min + coursework_min


# ================================
# EXAMPLE USAGE (for testing in separate file)
# ================================

def create_test_data():
    """Create test data for validation"""
    subjects = [
        Subject(1, "Linear Algebra", 2.0, 4, 0.65),
        Subject(2, "Data Structures", 2.5, 5, 0.70),
        Subject(3, "Networks", 1.5, 3, 0.85),
        Subject(4, "Software Engineering", 2.0, 4, 0.75)
    ]

    coursework = [
        Coursework(1, "DS Assignment", 12.0, 0.9, 4.5),
        Coursework(2, "Network Lab", 8.0, 0.6, 3.0),
        Coursework(3, "Math Problem Set", 6.0, 0.8, 2.5)
    ]

    return subjects, coursework


def run_example():
    """Example run for demonstration"""
    subjects, coursework = create_test_data()
    params = Parameters(total_hours=40.0, mode="balanced")

    print_mathematical_formulation(subjects, coursework, params)

    solution = solve_resource_allocation(subjects, coursework, params)

    print_solution(solution, subjects, coursework)

    min_req = calculate_minimum_requirements(subjects, coursework)
    print(f"\nMinimum Required: {min_req:.2f} hours")
    print(f"Available: {params.total_hours} hours")
    print(f"Feasible: {'Yes' if min_req <= params.total_hours else 'No'}")


if __name__ == "__main__":
    run_example()