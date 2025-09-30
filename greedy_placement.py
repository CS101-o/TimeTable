from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ================================
# DATA STRUCTURES
# ================================

class BlockType(Enum):
    STUDY = "study"
    COURSEWORK = "coursework"
    LECTURE = "lecture"
    WORKSHOP = "workshop"


class TimeOfDay(Enum):
    PROFESSIONAL = "professional"  # 9 AM - 6 PM
    EVENING = "evening"  # 6 PM - 11 PM


@dataclass
class LectureSchedule:
    subject_name: str
    day_of_week: int  # 0=Monday, 6=Sunday
    start_hour: int
    start_minute: int
    duration_hours: float


@dataclass
class TimeBlock:
    subject_name: str
    block_type: BlockType
    duration: float
    priority: int
    requires_professional_hours: bool = False
    preparation_for_lecture: Optional[LectureSchedule] = None


@dataclass
class PlacedBlock:
    subject_name: str
    block_type: BlockType
    day_of_week: int
    start_time: datetime
    duration: float
    time_of_day: TimeOfDay


@dataclass
class ScheduleConfig:
    week_start: datetime  # Should be Monday 9:00 AM
    professional_start: int = 9  # 9 AM
    professional_end: int = 18  # 6 PM
    evening_end: int = 23  # 11 PM


# ================================
# STEP 1: BREAK ALLOCATIONS INTO BLOCKS
# ================================

def generate_time_blocks(
        allocation_result: Dict[str, float],
        coursework_optimal_size: float = 3.5,
        study_optimal_size: float = 1.5
) -> List[TimeBlock]:
    """
    Break time allocations into schedulable blocks

    Args:
        allocation_result: Output from Phase 1 (e.g., {'linear_algebra': 2.0, ...})
        coursework_optimal_size: Optimal block size for coursework (hours)
        study_optimal_size: Optimal block size for study (hours)

    Returns:
        List of TimeBlock objects ready for placement
    """
    blocks = []

    for subject_name, total_hours in allocation_result.items():
        if subject_name == 'slack':
            continue  # Skip slack time

        # Determine if coursework or study based on name
        is_coursework = any(keyword in subject_name.lower()
                            for keyword in ['assignment', 'project', 'lab', 'report'])

        if is_coursework:
            blocks.extend(_split_into_blocks(
                subject_name=subject_name,
                total_hours=total_hours,
                block_type=BlockType.COURSEWORK,
                optimal_size=coursework_optimal_size,
                max_size=4.0,
                min_size=2.0,
                priority=5,  # Coursework is high priority
                requires_professional_hours=True
            ))
        else:
            blocks.extend(_split_into_blocks(
                subject_name=subject_name,
                total_hours=total_hours,
                block_type=BlockType.STUDY,
                optimal_size=study_optimal_size,
                max_size=2.0,
                min_size=0.5,
                priority=3,  # Study is medium priority
                requires_professional_hours=False
            ))

    return blocks


def _split_into_blocks(
        subject_name: str,
        total_hours: float,
        block_type: BlockType,
        optimal_size: float,
        max_size: float,
        min_size: float,
        priority: int,
        requires_professional_hours: bool
) -> List[TimeBlock]:
    """
    Split total hours into optimally-sized blocks

    Example: 14.1 hours → [4.0, 4.0, 3.5, 2.6]
    """
    blocks = []
    remaining = total_hours

    while remaining > 0:
        if remaining >= optimal_size:
            block_size = optimal_size
        elif remaining >= min_size:
            block_size = remaining
        else:
            # Merge small remainder with last block
            if blocks:
                blocks[-1] = TimeBlock(
                    subject_name=subject_name,
                    block_type=block_type,
                    duration=blocks[-1].duration + remaining,
                    priority=priority,
                    requires_professional_hours=requires_professional_hours
                )
                break
            else:
                block_size = remaining

        blocks.append(TimeBlock(
            subject_name=subject_name,
            block_type=block_type,
            duration=block_size,
            priority=priority,
            requires_professional_hours=requires_professional_hours
        ))

        remaining -= block_size

    return blocks


# ================================
# STEP 2: GENERATE AVAILABLE SLOTS
# ================================

def generate_available_slots(
        config: ScheduleConfig,
        lecture_schedule: List[LectureSchedule]
) -> Dict[int, List[Tuple[datetime, float]]]:
    """
    Generate available time slots for each day, excluding lectures

    Returns:
        Dictionary mapping day_of_week to list of (start_time, duration) tuples
    """
    available_slots = {}

    for day in range(7):  # Monday to Sunday
        day_start = config.week_start + timedelta(days=day)
        slots = []

        # Professional hours: 9 AM - 6 PM
        professional_start = day_start.replace(hour=config.professional_start, minute=0)
        professional_end = day_start.replace(hour=config.professional_end, minute=0)

        # Evening hours: 6 PM - 11 PM
        evening_start = day_start.replace(hour=config.professional_end, minute=0)
        evening_end = day_start.replace(hour=config.evening_end, minute=0)

        # Block out lecture times
        lectures_today = [lec for lec in lecture_schedule if lec.day_of_week == day]

        # Generate slots avoiding lectures
        current_time = professional_start
        while current_time < evening_end:
            # Check if this time conflicts with any lecture
            conflict = False
            for lecture in lectures_today:
                lecture_start = day_start.replace(hour=lecture.start_hour, minute=lecture.start_minute)
                lecture_end = lecture_start + timedelta(hours=lecture.duration_hours)

                if current_time < lecture_end and current_time >= lecture_start:
                    conflict = True
                    current_time = lecture_end
                    break

            if not conflict:
                # Find how much continuous time is available
                available_until = evening_end
                for lecture in lectures_today:
                    lecture_start = day_start.replace(hour=lecture.start_hour, minute=lecture.start_minute)
                    if lecture_start > current_time:
                        available_until = min(available_until, lecture_start)

                slot_duration = (available_until - current_time).total_seconds() / 3600
                if slot_duration >= 0.5:  # At least 30 minutes
                    slots.append((current_time, slot_duration))

                current_time = available_until

        available_slots[day] = slots

    return available_slots


# ================================
# STEP 3: GREEDY PLACEMENT
# ================================

def place_blocks_greedy(
        blocks: List[TimeBlock],
        available_slots: Dict[int, List[Tuple[datetime, float]]],
        lecture_schedule: List[LectureSchedule],
        config: ScheduleConfig
) -> List[PlacedBlock]:
    """
    Greedy algorithm to place blocks into available slots

    Priority order:
    1. High priority blocks first (coursework)
    2. Larger blocks first (harder to place)
    3. Blocks requiring professional hours
    4. Study sessions before lectures (1 day preparation)
    """
    placed_blocks = []
    remaining_slots = {day: slots.copy() for day, slots in available_slots.items()}

    # Sort blocks by priority
    sorted_blocks = sorted(blocks, key=lambda b: (
        -b.priority,  # Higher priority first
        -b.duration,  # Larger blocks first
        -b.requires_professional_hours  # Professional hours requirement
    ))

    for block in sorted_blocks:
        best_placement = _find_best_slot(block, remaining_slots, lecture_schedule, config)

        if best_placement:
            placed_blocks.append(best_placement)
            _mark_slot_used(remaining_slots, best_placement)
        else:
            print(f"Warning: Could not place block {block.subject_name} ({block.duration}h)")

    return placed_blocks


def _find_best_slot(
        block: TimeBlock,
        available_slots: Dict[int, List[Tuple[datetime, float]]],
        lecture_schedule: List[LectureSchedule],
        config: ScheduleConfig
) -> Optional[PlacedBlock]:
    """
    Find the best time slot for a block using scoring heuristics
    """
    best_slot = None
    best_score = -float('inf')

    for day, slots in available_slots.items():
        for slot_start, slot_duration in slots:
            if slot_duration >= block.duration:
                score = _score_slot(block, day, slot_start, lecture_schedule, config)

                if score > best_score:
                    best_score = score
                    best_slot = (day, slot_start)

    if best_slot:
        day, start_time = best_slot
        time_of_day = _get_time_of_day(start_time, config)

        return PlacedBlock(
            subject_name=block.subject_name,
            block_type=block.block_type,
            day_of_week=day,
            start_time=start_time,
            duration=block.duration,
            time_of_day=time_of_day
        )

    return None


def _score_slot(
        block: TimeBlock,
        day: int,
        start_time: datetime,
        lecture_schedule: List[LectureSchedule],
        config: ScheduleConfig
) -> float:
    """
    Score a time slot for a block (higher = better)

    Scoring factors:
    - Coursework in professional hours: +10
    - Study before related lecture: +20
    - Morning for study: +5
    - Afternoon/evening for coursework: +5
    """
    score = 0.0
    hour = start_time.hour

    # Professional hours preference for coursework
    if block.requires_professional_hours:
        if config.professional_start <= hour < config.professional_end:
            score += 10
        else:
            score -= 20  # Strong penalty for coursework outside professional hours

    # Study timing preferences
    if block.block_type == BlockType.STUDY:
        # Check if there's a related lecture the next day
        related_lectures = [lec for lec in lecture_schedule
                            if lec.subject_name.lower() in block.subject_name.lower()]

        for lecture in related_lectures:
            if lecture.day_of_week == (day + 1) % 7:  # Next day
                score += 20  # Strong bonus for studying before lecture
            elif lecture.day_of_week == day:  # Same day before lecture
                lecture_hour = lecture.start_hour
                if hour < lecture_hour - 2:  # At least 2 hours before
                    score += 10

        # Morning study bonus
        if 9 <= hour <= 12:
            score += 5

    # Coursework timing preferences
    if block.block_type == BlockType.COURSEWORK:
        # Afternoon/evening deep work
        if 14 <= hour <= 20:
            score += 5

    # Avoid very late hours
    if hour >= 21:
        score -= 3

    return score


def _get_time_of_day(start_time: datetime, config: ScheduleConfig) -> TimeOfDay:
    """Determine if time is in professional or evening hours"""
    hour = start_time.hour
    if config.professional_start <= hour < config.professional_end:
        return TimeOfDay.PROFESSIONAL
    else:
        return TimeOfDay.EVENING


def _mark_slot_used(
        available_slots: Dict[int, List[Tuple[datetime, float]]],
        placed_block: PlacedBlock
):
    """Remove or reduce the slot that was just used"""
    day = placed_block.day_of_week
    slots = available_slots[day]
    block_end = placed_block.start_time + timedelta(hours=placed_block.duration)

    new_slots = []
    for slot_start, slot_duration in slots:
        slot_end = slot_start + timedelta(hours=slot_duration)

        # Slot completely before placed block - keep it
        if slot_end <= placed_block.start_time:
            new_slots.append((slot_start, slot_duration))
        # Slot completely after placed block - keep it
        elif slot_start >= block_end:
            new_slots.append((slot_start, slot_duration))
        # Slot overlaps - split it
        else:
            # Before part
            if slot_start < placed_block.start_time:
                before_duration = (placed_block.start_time - slot_start).total_seconds() / 3600
                if before_duration >= 0.5:
                    new_slots.append((slot_start, before_duration))

            # After part
            if slot_end > block_end:
                after_duration = (slot_end - block_end).total_seconds() / 3600
                if after_duration >= 0.5:
                    new_slots.append((block_end, after_duration))

    available_slots[day] = new_slots


# ================================
# UTILITY FUNCTIONS
# ================================

def print_schedule(placed_blocks: List[PlacedBlock]):
    """Pretty print the generated schedule"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    print("\nGENERATED SCHEDULE")
    print("=" * 80)

    for day_idx in range(7):
        day_blocks = [b for b in placed_blocks if b.day_of_week == day_idx]
        if day_blocks:
            print(f"\n{days[day_idx]}:")
            day_blocks.sort(key=lambda b: b.start_time)
            for block in day_blocks:
                time_str = block.start_time.strftime("%H:%M")
                end_time = block.start_time + timedelta(hours=block.duration)
                end_str = end_time.strftime("%H:%M")
                print(
                    f"  {time_str}-{end_str} | {block.subject_name} | {block.block_type.value} | {block.duration:.1f}h")



