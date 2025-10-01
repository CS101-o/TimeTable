"""
Resource allocation optimizer using linear programming.
Balances study time across subjects and coursework deadlines.
"""

import pulp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class Subject:
    id: int
    name: str
    min_hours: float  # min weekly study time
    priority: int  # 1-5, higher = more important
    performance: float  # current grade/performance 0-1


@dataclass
class Coursework:
    id: int
    name: str
    estimated_hours: float  # time estimate
    urgency: float  # 0-1, deadline pressure
    complexity: float  # 1-5 difficulty rating


@dataclass
class Parameters:
    total_hours: float
    mode: str
    alpha: float = 0.1  # slack penalty weight


@dataclass
class Solution:
    feasible: bool
    x: Dict[int, float]  # study allocation
    y: Dict[int, float]  # coursework allocation
    s: float  # unused time
    objective_value: float


# Struggling students need more time than the minimum
def phi(performance: float) -> float:
    if performance < 0.6:
        return 1.3
    elif performance <= 0.8:
        return 1.0
    else:
        return 0.7


# Complex assignments need buffer time beyond estimates
def beta(complexity: float) -> float:
    return 1.0 + 0.2 * (complexity - 1) / 4


def add_time_budget_constraint(model: pulp.LpProblem, x: Dict, y: Dict, s: pulp.LpVariable, H: float):
    """Total allocated time can't exceed available hours."""
    total_time = pulp.lpSum([x[i] for i in x]) + pulp.lpSum([y[j] for j in y]) + s
    model += total_time == H, "TimeBudget"


def add_subject_min_constraints(model: pulp.LpProblem, x: Dict, subjects: List[Subject]):
    """Each subject needs minimum study time (adjusted for performance)."""
    for subject in subjects:
        adj_min = subject.min_hours * phi(subject.performance)
        model += x[subject.id] >= adj_min, f"SubjectMin_{subject.id}"


def add_coursework_min_constraints(model: pulp.LpProblem, y: Dict, coursework: List[Coursework]):
    """Each assignment needs minimum time (with complexity buffer)."""
    for cw in coursework:
        adj_est = cw.estimated_hours * beta(cw.complexity)
        model += y[cw.id] >= adj_est, f"CourseworkMin_{cw.id}"


def add_non_neg_constraints(model: pulp.LpProblem, x: Dict, y: Dict, s: pulp.LpVariable):
    """All time allocations must be non-negative."""
    for i in x:
        model += x[i] >= 0, f"NonNeg_x_{i}"
    for j in y:
        model += y[j] >= 0, f"NonNeg_y_{j}"
    model += s >= 0, "NonNeg_s"


def obj_harm_reduction(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                       coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """Minimize total study time - bare minimum approach."""
    return pulp.lpSum([x[i] for i in x])


def obj_balanced(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                 coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """Balance study priorities, urgent work, and some buffer time."""
    study_term = pulp.lpSum([x[subject.id] / subject.priority for subject in subjects])
    coursework_term = pulp.lpSum([cw.urgency * y[cw.id] for cw in coursework])
    slack_term = params.alpha * s
    return study_term + coursework_term + slack_term


def obj_perfection(x: Dict, y: Dict, s: pulp.LpVariable, subjects: List[Subject],
                   coursework: List[Coursework], params: Parameters) -> pulp.LpAffineExpression:
    """Optimize learning efficiency - focus on high-priority subjects where you're doing well."""
    study_term = pulp.lpSum([x[subject.id] / (subject.priority * subject.performance + 0.1)
                             for subject in subjects])
    coursework_term = pulp.lpSum([0.5 * y[cw.id] for cw in coursework])
    return study_term + coursework_term


def create_variables(subjects: List[Subject], coursework: List[Coursework]) -> Tuple[Dict, Dict, pulp.LpVariable]:
    """Set up all decision variables."""
    x = {subject.id: pulp.LpVariable(f"x_{subject.id}", lowBound=0, cat='Continuous')
         for subject in subjects}

    y = {cw.id: pulp.LpVariable(f"y_{cw.id}", lowBound=0, cat='Continuous')
         for cw in coursework}

    s = pulp.LpVariable("s", lowBound=0, cat='Continuous')

    return x, y, s


def solve_allocation(subjects: List[Subject], coursework: List[Coursework],
                     params: Parameters) -> Solution:
    """Main solver - builds and solves the LP model."""
    model = pulp.LpProblem("ResourceAllocation", pulp.LpMinimize)

    x, y, s = create_variables(subjects, coursework)

    # Add all constraints
    add_time_budget_constraint(model, x, y, s, params.total_hours)
    add_subject_min_constraints(model, x, subjects)
    add_coursework_min_constraints(model, y, coursework)
    add_non_neg_constraints(model, x, y, s)

    # Pick objective based on mode
    objective_funcs = {
        "harm_reduction": obj_harm_reduction,
        "balanced": obj_balanced,
        "perfection": obj_perfection
    }

    if params.mode not in objective_funcs:
        raise ValueError(f"Unknown mode: {params.mode}")

    objective = objective_funcs[params.mode](x, y, s, subjects, coursework, params)
    model += objective

    # Solve it
    status = model.solve(pulp.PULP_CBC_CMD(msg=0))

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


def print_formulation(subjects: List[Subject], coursework: List[Coursework], params: Parameters):
    """Show the math formulation for debugging/verification."""
    print("MATHEMATICAL FORMULATION")
    print("=" * 50)

    print("Decision Variables:")
    for subject in subjects:
        print(f"  x_{subject.id} = hours for {subject.name}")
    for cw in coursework:
        print(f"  y_{cw.id} = hours for {cw.name}")
    print(f"  s = slack time")

    print(f"\nObjective ({params.mode} mode):")
    if params.mode == "harm_reduction":
        variables = " + ".join([f"x_{s.id}" for s in subjects])
        print(f"  min {variables}")
    elif params.mode == "balanced":
        study_terms = " + ".join([f"x_{s.id}/{s.priority}" for s in subjects])
        cw_terms = " + ".join([f"{cw.urgency}*y_{cw.id}" for cw in coursework])
        print(f"  min {study_terms} + {cw_terms} + {params.alpha}*s")

    print(f"\nConstraints:")
    all_vars = [f"x_{s.id}" for s in subjects] + [f"y_{cw.id}" for cw in coursework] + ["s"]
    print(f"  {' + '.join(all_vars)} <= {params.total_hours}")

    for subject in subjects:
        adj_min = subject.min_hours * phi(subject.performance)
        print(f"  x_{subject.id} >= {adj_min:.2f}")

    for cw in coursework:
        adj_est = cw.estimated_hours * beta(cw.complexity)
        print(f"  y_{cw.id} >= {adj_est:.2f}")

    print("  All variables >= 0")


def print_solution(solution: Solution, subjects: List[Subject], coursework: List[Coursework]):
    """Display results."""
    print("\nSOLUTION")
    print("=" * 50)

    if not solution.feasible:
        print("INFEASIBLE - no solution exists with these constraints")
        return

    print("Found feasible solution")
    print(f"Objective value: {solution.objective_value:.4f}")

    print(f"\nStudy time allocation:")
    total_study = 0
    for subject in subjects:
        hours = solution.x[subject.id]
        total_study += hours
        print(f"  {subject.name}: {hours:.2f} hours")

    print(f"\nCoursework time allocation:")
    total_cw = 0
    for cw in coursework:
        hours = solution.y[cw.id]
        total_cw += hours
        print(f"  {cw.name}: {hours:.2f} hours")

    print(f"\nSlack time: {solution.s:.2f} hours")
    print(f"\nTotals:")
    print(f"  Study: {total_study:.2f} hours")
    print(f"  Coursework: {total_cw:.2f} hours")
    print(f"  Slack: {solution.s:.2f} hours")
    print(f"  Total: {total_study + total_cw + solution.s:.2f} hours")


def calc_min_requirements(subjects: List[Subject], coursework: List[Coursework]) -> float:
    """Calculate minimum time needed to satisfy all constraints."""
    study_min = sum(subject.min_hours * phi(subject.performance) for subject in subjects)
    cw_min = sum(cw.estimated_hours * beta(cw.complexity) for cw in coursework)
    return study_min + cw_min


def make_test_data():
    """Test data for validation."""
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
    """Quick test run."""
    subjects, coursework = make_test_data()
    params = Parameters(total_hours=40.0, mode="balanced")

    print_formulation(subjects, coursework, params)

    solution = solve_allocation(subjects, coursework, params)

    print_solution(solution, subjects, coursework)

    min_req = calc_min_requirements(subjects, coursework)
    print(f"\nMinimum required: {min_req:.2f} hours")
    print(f"Available: {params.total_hours} hours")
    print(f"Feasible: {'Yes' if min_req <= params.total_hours else 'No'}")


if __name__ == "__main__":
    run_example()