# your_schedule.py

from greedy_placement import LectureSchedule
from ILP import Subject, Coursework
from datetime import datetime

# Your actual lecture schedule
my_lectures = [
    # NLP - Tuesday 11:00-13:00 (2 hours)
    LectureSchedule(
        subject_name="NLP",
        day_of_week=1,  # Tuesday
        start_hour=11,
        start_minute=0,
        duration_hours=2.0
    ),

    # Chip Design - Wednesday 9:00-10:00 (1 hour lecture)
    LectureSchedule(
        subject_name="Chip Design",
        day_of_week=2,  # Wednesday
        start_hour=9,
        start_minute=0,
        duration_hours=1.0
    ),

    # Chip Design - Friday 9:00-10:00 (1 hour lecture)
    LectureSchedule(
        subject_name="Chip Design",
        day_of_week=4,  # Friday
        start_hour=9,
        start_minute=0,
        duration_hours=1.0
    ),

    # Chip Design Lab - Friday 10:00-11:00 (1 hour lab)
    LectureSchedule(
        subject_name="Chip Design",
        day_of_week=4,  # Friday
        start_hour=10,
        start_minute=0,
        duration_hours=1.0
    )
]

# Your subjects (adjust performance scores based on how you're doing)
my_subjects = [
    Subject(
        id=1,
        name="NLP",
        min_hours=2.0,
        priority=5,
        performance=0.70  # Adjust based on your actual performance
    ),
    Subject(
        id=2,
        name="Chip Design",
        min_hours=2.5,
        priority=5,
        performance=0.65  # Adjust based on your actual performance
    )
]

# Your coursework (adjust based on actual assignments)
my_coursework = [
    Coursework(
        id=1,
        name="NLP Assignment",
        estimated_hours=12.0,
        urgency=0.9,
        complexity=4.0
    ),
    Coursework(
        id=2,
        name="Chip Design Project",
        estimated_hours=15.0,
        urgency=0.8,
        complexity=4.5
    )
]