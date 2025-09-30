from typing import List
from ILP import solve_resource_allocation, Subject, Coursework, Parameters, Solution
from greedy_placement import (
    generate_time_blocks, generate_available_slots, place_blocks_greedy,
    ScheduleConfig, LectureSchedule, print_schedule
)
from datetime import datetime

def create_complete_schedule(
    subjects: List[Subject],
    coursework: List[Coursework],
    lecture_schedule: List[LectureSchedule],
    total_hours: float,
    optimization_mode: str = "balanced"
):
    """
    Complete scheduling pipeline: Phase 1 (ILP) → Phase 2 (Greedy)
    """

    # PHASE 1: Resource Allocation (ILP)
    print("=" * 80)
    print("PHASE 1: RESOURCE ALLOCATION")
    print("=" * 80)

    params = Parameters(total_hours=total_hours, mode=optimization_mode)
    solution = solve_resource_allocation(subjects, coursework, params)

    if not solution.feasible:
        print("❌ Phase 1 failed: No feasible resource allocation")
        return None

    print("\n✅ Phase 1 Complete - Time Allocations:")
    for subject in subjects:
        hours = solution.x[subject.id]
        print(f"  {subject.name}: {hours:.2f} hours")

    for cw in coursework:
        hours = solution.y[cw.id]
        print(f"  {cw.name}: {hours:.2f} hours")

    print(f"  Slack: {solution.s:.2f} hours")

    # Convert Phase 1 output to Phase 2 input format
    allocation_dict = {}
    for subject in subjects:
        allocation_dict[subject.name] = solution.x[subject.id]
    for cw in coursework:
        allocation_dict[cw.name] = solution.y[cw.id]

    # PHASE 2: Time Block Placement (Greedy)
    print("\n" + "=" * 80)
    print("PHASE 2: TIME BLOCK PLACEMENT")
    print("=" * 80)

    # Generate blocks from Phase 1 allocations
    blocks = generate_time_blocks(allocation_dict)
    print(f"\n✅ Generated {len(blocks)} time blocks")

    # Setup schedule configuration
    config = ScheduleConfig(week_start=datetime(2024, 1, 1, 9, 0))  # Adjust to your week

    # Generate available time slots (excluding lectures)
    available_slots = generate_available_slots(config, lecture_schedule)
    print(f"✅ Generated available time slots for 7 days")

    # Place blocks using greedy algorithm
    placed_blocks = place_blocks_greedy(blocks, available_slots, lecture_schedule, config)
    print(f"✅ Placed {len(placed_blocks)} blocks successfully")

    if len(placed_blocks) < len(blocks):
        print(f"⚠️  Warning: {len(blocks) - len(placed_blocks)} blocks could not be placed")

    # Print final schedule
    print_schedule(placed_blocks)

    return {
        'phase1_solution': solution,
        'allocation_dict': allocation_dict,
        'time_blocks': blocks,
        'placed_blocks': placed_blocks
    }

# Example usage
if __name__ == "__main__":
    # Define your subjects (from Phase 1)
    subjects = [
        Subject(1, "Linear Algebra", 2.0, 4, 0.65),
        Subject(2, "Data Structures", 2.5, 5, 0.70),
        Subject(3, "Networks", 1.5, 3, 0.85)
    ]

    # Define your coursework (from Phase 1)
    coursework = [
        Coursework(1, "DS Assignment", 12.0, 0.9, 4.5),
        Coursework(2, "Network Lab", 8.0, 0.6, 3.0)
    ]

    # Define YOUR actual lecture schedule
    my_lectures = [
        LectureSchedule("Linear Algebra", day_of_week=0, start_hour=10, start_minute=0, duration_hours=2),
        LectureSchedule("Data Structures", day_of_week=1, start_hour=14, start_minute=0, duration_hours=2),
        LectureSchedule("Networks", day_of_week=3, start_hour=10, start_minute=0, duration_hours=1.5),
        # Add your workshops here too
    ]

    # Run complete pipeline
    result = create_complete_schedule(
        subjects=subjects,
        coursework=coursework,
        lecture_schedule=my_lectures,
        total_hours=40.0,
        optimization_mode="balanced"
    )