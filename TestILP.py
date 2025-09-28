# test_scenarios.py
from ILP import *

def test_infeasible_case():
    subjects, coursework = create_test_data()
    params = Parameters(total_hours=25.0, mode="balanced")  # Too little time
    solution = solve_resource_allocation(subjects, coursework, params)
    print(f"25 hours scenario: Feasible = {solution.feasible}")

def test_harm_reduction_mode():
    subjects, coursework = create_test_data()
    params = Parameters(total_hours=40.0, mode="harm_reduction")
    solution = solve_resource_allocation(subjects, coursework, params)
    print(f"Harm reduction total study time: {sum(solution.x.values()):.2f}")

def test_perfection_mode():
    subjects, coursework = create_test_data()
    params = Parameters(total_hours=50.0, mode="perfection")  # Extra time
    solution = solve_resource_allocation(subjects, coursework, params)
    print(f"Perfection mode total study time: {sum(solution.x.values()):.2f}")

if __name__ == "__main__":
    test_infeasible_case()
    test_harm_reduction_mode()
    test_perfection_mode()