import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random


@dataclass
class TimeBlock:
    subject: str
    start_time: datetime
    duration: float  # hours
    block_type: str  # 'study' or 'coursework'
    priority: int
    is_fixed: bool = False


@dataclass
class Subject:
    name: str
    min_time: float  # minimum effective study time (hours)
    priority: int  # 1-5, higher = more important
    difficulty: float  # 1-5, higher = needs more time when struggling
    last_performance: float  # 0-1, recent quiz/understanding score

    def __post_init__(self):
        # Validate inputs
        if self.min_time < 0:
            raise ValueError(f"Minimum time cannot be negative: {self.min_time}")
        if not 0 <= self.last_performance <= 1:
            raise ValueError(f"Performance must be between 0 and 1: {self.last_performance}")
        if not 1 <= self.priority <= 5:
            raise ValueError(f"Priority must be between 1 and 5: {self.priority}")
        if not 1 <= self.difficulty <= 5:
            raise ValueError(f"Difficulty must be between 1 and 5: {self.difficulty}")


@dataclass
class Coursework:
    name: str
    estimated_hours: float
    deadline_urgency: float  # 0-1, higher = more urgent
    complexity: float  # 1-5, affects time burn potential

    def __post_init__(self):
        if self.estimated_hours < 0:
            raise ValueError(f"Estimated hours cannot be negative: {self.estimated_hours}")
        if not 0 <= self.deadline_urgency <= 1:
            raise ValueError(f"Deadline urgency must be between 0 and 1: {self.deadline_urgency}")
        if not 1 <= self.complexity <= 5:
            raise ValueError(f"Complexity must be between 1 and 5: {self.complexity}")


class OptimizationMode(Enum):
    HARM_REDUCTION = "harm_reduction"  # Minimize study time, ensure survival
    BALANCED = "balanced"  # Balance study and coursework
    PERFECTION = "perfection"  # Maximize learning when time allows


def calculate_subject_minimums(subjects: List[Subject], mode: OptimizationMode) -> Dict[str, float]:
    """Calculate minimum viable study time per subject based on optimization mode"""
    minimums = {}

    for subject in subjects:
        base_min = subject.min_time

        # Adjust based on recent performance
        performance_factor = 1.0
        if subject.last_performance < 0.7:  # Struggling
            performance_factor = 1.3
        elif subject.last_performance > 0.9:  # Excelling
            performance_factor = 0.8

        # Adjust based on optimization mode
        if mode == OptimizationMode.HARM_REDUCTION:
            # Absolute minimum - just enough to not fall behind
            mode_factor = 0.7
        elif mode == OptimizationMode.BALANCED:
            mode_factor = 1.0
        else:  # PERFECTION
            mode_factor = 1.2

        minimums[subject.name] = base_min * performance_factor * mode_factor

    return minimums


def estimate_coursework_time(coursework_list: List[Coursework], buffer_factor: float = 1.3) -> float:
    """Estimate total coursework time with buffer for time burn"""
    if not coursework_list:
        return 0.0

    total_estimated = sum(cw.estimated_hours * (1 + cw.complexity * 0.2) for cw in coursework_list)
    return total_estimated * buffer_factor


def basic_schedule_optimizer(
        subjects: List[Subject],
        coursework_list: List[Coursework],
        available_hours: float,
        mode: OptimizationMode = OptimizationMode.BALANCED
) -> Dict[str, any]:
    """
    Core optimization algorithm - finds feasible time allocation
    """

    # Handle edge cases
    if not subjects and not coursework_list:
        raise Exception("Must have at least subjects or coursework to schedule")

    if available_hours <= 0:
        return {
            'feasible': False,
            'deficit': float('inf'),
            'recommendations': ["No time available for scheduling"],
            'risk_level': 'critical'
        }

    # Step 1: Calculate minimum study requirements
    min_study_per_subject = calculate_subject_minimums(subjects, mode) if subjects else {}
    total_min_study = sum(min_study_per_subject.values())

    # Step 2: Estimate coursework time needs
    estimated_coursework = estimate_coursework_time(coursework_list) if coursework_list else 0

    # Step 3: Check feasibility
    required_time = total_min_study + estimated_coursework

    if required_time > available_hours:
        return handle_infeasible_schedule(subjects, coursework_list, available_hours, min_study_per_subject)

    # Step 4: Allocate remaining time
    remaining_time = available_hours - required_time

    # Strategy: Give extra time to coursework first (your key insight)
    coursework_allocation = estimated_coursework + (remaining_time * 0.8)  # 80% to coursework
    study_bonus = remaining_time * 0.2  # 20% bonus to struggling subjects

    # Distribute study bonus to subjects that need it most
    final_study_allocation = distribute_study_bonus(subjects, min_study_per_subject, study_bonus)

    # Step 5: Generate actual time blocks using greedy placement
    time_blocks = greedy_time_block_placement(
        final_study_allocation,
        coursework_allocation,
        coursework_list,
        available_hours
    )

    return {
        'feasible': True,
        'study_allocation': final_study_allocation,
        'coursework_allocation': coursework_allocation,
        'time_blocks': time_blocks,
        'total_scheduled': sum(final_study_allocation.values()) + coursework_allocation,
        'buffer_time': available_hours - sum(final_study_allocation.values()) - coursework_allocation,
        'optimization_mode': mode.value
    }


def handle_infeasible_schedule(
        subjects: List[Subject],
        coursework_list: List[Coursework],
        available_hours: float,
        min_study_per_subject: Dict[str, float]
) -> Dict[str, any]:
    """
    When there's not enough time - intelligent degradation strategies
    """

    total_min_study = sum(min_study_per_subject.values())
    estimated_coursework = estimate_coursework_time(coursework_list) if coursework_list else 0

    # Strategy 1: Reduce coursework buffer
    reduced_coursework = sum(
        cw.estimated_hours for cw in coursework_list) * 1.1 if coursework_list else 0  # Smaller buffer

    if total_min_study + reduced_coursework <= available_hours:
        remaining = available_hours - total_min_study - reduced_coursework
        return {
            'feasible': True,
            'study_allocation': min_study_per_subject,
            'coursework_allocation': reduced_coursework + remaining,
            'warning': 'Reduced coursework buffer - risk of time burn',
            'risk_level': 'medium',
            'time_blocks': greedy_time_block_placement(
                min_study_per_subject,
                reduced_coursework + remaining,
                coursework_list,
                available_hours
            )
        }

    # Strategy 2: Reduce study minimums (harm reduction mode)
    emergency_study = {name: time * 0.6 for name, time in min_study_per_subject.items()}
    emergency_total = sum(emergency_study.values())

    if emergency_total + reduced_coursework <= available_hours:
        remaining = available_hours - emergency_total - reduced_coursework
        return {
            'feasible': True,
            'study_allocation': emergency_study,
            'coursework_allocation': reduced_coursework + remaining,
            'warning': 'Emergency mode - minimum viable study time',
            'risk_level': 'high',
            'time_blocks': greedy_time_block_placement(
                emergency_study,
                reduced_coursework + remaining,
                coursework_list,
                available_hours
            )
        }

    # Strategy 3: Complete infeasibility - need to drop something
    return {
        'feasible': False,
        'deficit': (emergency_total + reduced_coursework) - available_hours,
        'recommendations': generate_deficit_recommendations(subjects, coursework_list),
        'risk_level': 'critical'
    }


def distribute_study_bonus(
        subjects: List[Subject],
        base_allocation: Dict[str, float],
        bonus_time: float
) -> Dict[str, float]:
    """
    Intelligently distribute bonus study time based on need
    """

    if bonus_time <= 0 or not subjects:
        return base_allocation

    # Priority scoring: struggling subjects + high priority get more time
    priority_scores = {}
    for subject in subjects:
        performance_need = (1 - subject.last_performance)  # More time if performing poorly
        priority_weight = subject.priority / 5.0
        difficulty_factor = subject.difficulty / 5.0

        priority_scores[subject.name] = performance_need * 0.5 + priority_weight * 0.3 + difficulty_factor * 0.2

    total_priority = sum(priority_scores.values())

    # Distribute bonus proportionally
    final_allocation = base_allocation.copy()
    for subject_name, base_time in base_allocation.items():
        if total_priority > 0:
            bonus_share = (priority_scores[subject_name] / total_priority) * bonus_time
            final_allocation[subject_name] = base_time + bonus_share

    return final_allocation


def generate_deficit_recommendations(subjects: List[Subject], coursework_list: List[Coursework]) -> List[str]:
    """
    Generate actionable recommendations when schedule is impossible
    """
    recommendations = []

    # Find least critical subjects
    if subjects:
        low_priority_subjects = [s for s in subjects if s.priority <= 2]
        if low_priority_subjects:
            recommendations.append(f"Consider reducing time on: {', '.join(s.name for s in low_priority_subjects)}")

    # Find coursework that might be reducible
    if coursework_list:
        complex_coursework = [cw for cw in coursework_list if cw.complexity >= 4]
        if complex_coursework:
            recommendations.append(
                f"High-risk coursework (may burn time): {', '.join(cw.name for cw in complex_coursework)}")

    recommendations.append("Consider extending deadlines or seeking help with coursework")
    recommendations.append("Focus on highest priority subjects only")

    return recommendations


def greedy_time_block_placement(
        study_allocation: Dict[str, float],
        coursework_allocation: float,
        coursework_list: List[Coursework],
        available_hours: float,
        week_start: datetime = None
) -> List[TimeBlock]:
    """
    Greedy algorithm to place time blocks in actual schedule
    Key insight: Place high-priority, large blocks first for optimal fit
    """

    if week_start is None:
        week_start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    # Available time slots (9 AM to 11 PM, Monday to Sunday)
    available_slots = generate_available_slots(week_start)
    time_blocks = []

    # Step 1: Create all blocks that need to be placed
    all_blocks = []

    # Add coursework blocks (prioritize by urgency and size)
    if coursework_list and coursework_allocation > 0:
        coursework_blocks = create_coursework_blocks(coursework_list, coursework_allocation)
        all_blocks.extend(coursework_blocks)

    # Add study blocks
    if study_allocation:
        study_blocks = create_study_blocks(study_allocation)
        all_blocks.extend(study_blocks)

    # Step 2: Sort blocks for greedy placement
    # Priority order: urgent coursework → large coursework → difficult subjects → regular study
    all_blocks.sort(key=lambda b: (
        -b.priority,  # Higher priority first
        -b.duration,  # Larger blocks first (harder to place)
        b.block_type == 'study'  # Coursework before study
    ))

    # Step 3: Greedy placement - place each block in best available slot
    for block in all_blocks:
        best_slot = find_best_slot(block, available_slots)
        if best_slot:
            block.start_time = best_slot
            time_blocks.append(block)
            # Remove used time slots
            remove_used_slots(available_slots, best_slot, block.duration)
        else:
            # Couldn't place block - fragment it or flag as conflict
            fragmented_blocks = fragment_block(block, available_slots)
            time_blocks.extend(fragmented_blocks)
            for frag in fragmented_blocks:
                remove_used_slots(available_slots, frag.start_time, frag.duration)

    return time_blocks


def generate_available_slots(week_start: datetime) -> List[datetime]:
    """Generate available time slots for the week"""
    slots = []
    current = week_start

    for day in range(7):  # 7 days
        day_start = current + timedelta(days=day)
        # Available from 9 AM to 11 PM (14 hours)
        for hour in range(9, 23):  # 9 AM to 11 PM
            for minute in [0, 30]:  # 30-minute slots
                slot_time = day_start.replace(hour=hour, minute=minute)
                slots.append(slot_time)

    return slots


def create_coursework_blocks(coursework_list: List[Coursework], total_hours: float) -> List[TimeBlock]:
    """Create coursework blocks with smart sizing"""
    blocks = []

    if not coursework_list or total_hours <= 0:
        return blocks

    # Distribute total coursework hours among projects
    total_estimated = sum(cw.estimated_hours for cw in coursework_list)

    if total_estimated == 0:
        return blocks

    for coursework in coursework_list:
        # Proportional allocation
        allocated_hours = (coursework.estimated_hours / total_estimated) * total_hours

        # Create blocks of optimal size (2-4 hours for deep work)
        blocks.extend(create_blocks_for_task(
            coursework.name,
            allocated_hours,
            'coursework',
            priority=int(coursework.deadline_urgency * 5 + coursework.complexity),
            optimal_block_size=3.5  # Sweet spot for coursework
        ))

    return blocks


def create_study_blocks(study_allocation: Dict[str, float]) -> List[TimeBlock]:
    """Create study blocks with subject-appropriate sizing"""
    blocks = []

    for subject, hours in study_allocation.items():
        if hours > 0:
            # Study blocks should be smaller (0.5-2 hours for focus)
            blocks.extend(create_blocks_for_task(
                subject,
                hours,
                'study',
                priority=3,  # Medium priority
                optimal_block_size=1.5  # Sweet spot for study sessions
            ))

    return blocks


def create_blocks_for_task(
        task_name: str,
        total_hours: float,
        block_type: str,
        priority: int,
        optimal_block_size: float
) -> List[TimeBlock]:
    """Break task into optimally-sized blocks"""
    blocks = []
    remaining = total_hours

    while remaining > 0:
        if remaining >= optimal_block_size:
            block_size = optimal_block_size
        elif remaining >= optimal_block_size * 0.5:  # Don't create tiny blocks
            block_size = remaining
        else:
            # Merge small remainder with last block if exists
            if blocks:
                blocks[-1].duration += remaining
                break
            else:
                block_size = remaining

        blocks.append(TimeBlock(
            subject=task_name,
            start_time=datetime.min,  # Will be set by placement algorithm
            duration=block_size,
            block_type=block_type,
            priority=priority
        ))

        remaining -= block_size

    return blocks


def find_best_slot(block: TimeBlock, available_slots: List[datetime]) -> Optional[datetime]:
    """
    Find the best time slot for a block using greedy heuristics
    """
    if not available_slots:
        return None

    required_slots = int(block.duration * 2)  # Convert hours to 30-min slots

    best_slot = None
    best_score = -1

    for i in range(len(available_slots) - required_slots + 1):
        start_slot = available_slots[i]

        # Check if we have enough consecutive slots
        consecutive = True
        for j in range(required_slots):
            expected_time = start_slot + timedelta(minutes=j * 30)
            if j + i >= len(available_slots) or available_slots[i + j] != expected_time:
                consecutive = False
                break

        if not consecutive:
            continue

        # Score this slot based on multiple factors
        score = score_time_slot(start_slot, block)

        if score > best_score:
            best_score = score
            best_slot = start_slot

    return best_slot


def score_time_slot(start_time: datetime, block: TimeBlock) -> float:
    """
    Score a time slot based on suitability for the block type
    Higher score = better fit
    """
    hour = start_time.hour
    day_of_week = start_time.weekday()  # 0 = Monday
    score = 0

    # Time of day preferences
    if block.block_type == 'coursework':
        # Coursework benefits from longer uninterrupted periods
        # Prefer afternoon/evening when you can get into flow state
        if 14 <= hour <= 20:  # 2 PM to 8 PM
            score += 3
        elif 10 <= hour <= 14:  # Morning backup
            score += 2
    else:  # study
        # Study sessions can be more flexible, prefer morning for better focus
        if 9 <= hour <= 12:  # Morning
            score += 3
        elif 14 <= hour <= 17:  # Early afternoon
            score += 2
        elif 19 <= hour <= 21:  # Evening review
            score += 1.5

    # Day of week preferences - prioritize earlier days for high priority blocks
    if block.priority >= 7:  # Very high priority (urgent coursework)
        # Strong preference for early week
        if day_of_week <= 1:  # Monday or Tuesday
            score += 4
        elif day_of_week <= 3:  # Wednesday or Thursday
            score += 2
        elif day_of_week == 4:  # Friday
            score += 1
        # Weekend gets no bonus for urgent items
    else:
        # Normal priority - standard preferences
        if day_of_week < 5:  # Weekdays
            score += 1
        else:  # Weekends - good for longer coursework blocks
            if block.block_type == 'coursework' and block.duration >= 3:
                score += 2

    # Avoid very late hours
    if hour >= 22:
        score -= 2

    return score


def remove_used_slots(available_slots: List[datetime], start_time: datetime, duration: float):
    """Remove used time slots from available list"""
    if not available_slots or duration <= 0:
        return

    slots_to_remove = int(duration * 2)  # Convert to 30-min slots

    # Create a list of slots to remove
    slots_to_remove_list = []
    current_slot = start_time
    for _ in range(slots_to_remove):
        slots_to_remove_list.append(current_slot)
        current_slot = current_slot + timedelta(minutes=30)

    # Remove the slots
    for slot in slots_to_remove_list:
        if slot in available_slots:
            available_slots.remove(slot)


def fragment_block(block: TimeBlock, available_slots: List[datetime]) -> List[TimeBlock]:
    """
    Fragment a block that couldn't be placed as one piece
    Last resort - creates smaller blocks
    """
    fragments = []
    remaining_duration = block.duration
    min_fragment_size = 0.5  # 30 minutes minimum

    while remaining_duration >= min_fragment_size and available_slots:
        # Find largest available slot
        max_consecutive = 0
        best_start = None

        for slot in available_slots:
            consecutive = count_consecutive_slots(slot, available_slots)
            if consecutive > max_consecutive:
                max_consecutive = consecutive
                best_start = slot

        if max_consecutive == 0:
            break

        fragment_duration = min(remaining_duration, max_consecutive * 0.5)

        if fragment_duration >= min_fragment_size:
            fragments.append(TimeBlock(
                subject=f"{block.subject} (Part {len(fragments) + 1})",
                start_time=best_start,
                duration=fragment_duration,
                block_type=block.block_type,
                priority=block.priority - 1  # Lower priority for fragments
            ))
            remaining_duration -= fragment_duration
            remove_used_slots(available_slots, best_start, fragment_duration)

    return fragments


def count_consecutive_slots(start_slot: datetime, available_slots: List[datetime]) -> int:
    """Count consecutive 30-minute slots from start_slot"""
    count = 0
    current_slot = start_slot

    while current_slot in available_slots:
        count += 1
        current_slot += timedelta(minutes=30)

    return count


# Add to the main example section
if __name__ == "__main__":
    # Sample data - replace with your actual subjects and coursework
    sample_subjects = [
        Subject("Linear Algebra", 2.0, 4, 3.5, 0.7),
        Subject("Data Structures", 2.5, 5, 4.0, 0.6),
        Subject("Computer Networks", 1.5, 3, 2.5, 0.8),
        Subject("Software Engineering", 2.0, 4, 3.0, 0.9)
    ]

    sample_coursework = [
        Coursework("DS Assignment", 8.0, 0.9, 4.5),
        Coursework("Network Project", 12.0, 0.6, 3.5),
        Coursework("SE Report", 6.0, 0.8, 2.5)
    ]

    # Test different scenarios
    scenarios = [
        (40, OptimizationMode.BALANCED, "Normal week"),
        (30, OptimizationMode.HARM_REDUCTION, "Heavy coursework week"),
        (50, OptimizationMode.PERFECTION, "Light week"),
        (25, OptimizationMode.HARM_REDUCTION, "Crisis week")
    ]

    for hours, mode, description in scenarios:
        print(f"\n=== {description} ({hours} hours available, {mode.value}) ===")
        result = basic_schedule_optimizer(sample_subjects, sample_coursework, hours, mode)

        if result['feasible']:
            print("✓ Schedule is feasible")
            print(f"Study time: {sum(result['study_allocation'].values()):.1f}h")
            print(f"Coursework time: {result['coursework_allocation']:.1f}h")

            if 'time_blocks' in result:
                print(f"Generated {len(result['time_blocks'])} time blocks:")
                for i, block in enumerate(result['time_blocks'][:5]):  # Show first 5
                    if block.start_time != datetime.min:
                        day_name = block.start_time.strftime("%A")
                        time_str = block.start_time.strftime("%I:%M %p")
                        print(f"  {i + 1}. {block.subject} - {day_name} {time_str} ({block.duration:.1f}h)")
                    else:
                        print(f"  {i + 1}. {block.subject} - Not scheduled ({block.duration:.1f}h)")
                if len(result['time_blocks']) > 5:
                    print(f"  ... and {len(result['time_blocks']) - 5} more blocks")

            if 'warning' in result:
                print(f"⚠️ {result['warning']}")
        else:
            print("✗ Schedule not feasible")
            print(f"Deficit: {result['deficit']:.1f} hours")
            for rec in result['recommendations']:
                print(f"• {rec}")