from integrated_scheduler import create_complete_schedule
from my_pesonel_schdule import my_lectures, my_subjects, my_coursework

if __name__ == "__main__":
    result = create_complete_schedule(
        subjects=my_subjects,
        coursework=my_coursework,
        lecture_schedule=my_lectures,
        total_hours=40.0,
        optimization_mode="balanced"
    )

    if result:
        print("\n" + "=" * 80)
        print("SCHEDULE GENERATION COMPLETE!")
        print("=" * 80)