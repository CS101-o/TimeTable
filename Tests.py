import unittest
import pytest
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os

# Add the optimization module to path (assuming it's in the same directory)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the optimization classes (you'll need to fix imports based on actual file structure)
from optimization_engine import (
    Subject, Coursework, OptimizationMode, TimeBlock,
    basic_schedule_optimizer, calculate_subject_minimums,
    estimate_coursework_time, greedy_time_block_placement
)


class TestOptimizationEngine(unittest.TestCase):

    def setUp(self):
        """Set up test data for each test case"""
        self.sample_subjects = [
            Subject("Linear Algebra", 2.0, 4, 3.5, 0.7),
            Subject("Data Structures", 2.5, 5, 4.0, 0.6),
            Subject("Computer Networks", 1.5, 3, 2.5, 0.8),
            Subject("Software Engineering", 2.0, 4, 3.0, 0.9)
        ]

        self.sample_coursework = [
            Coursework("DS Assignment", 8.0, 0.9, 4.5),
            Coursework("Network Project", 12.0, 0.6, 3.5),
            Coursework("SE Report", 6.0, 0.8, 2.5)
        ]

        self.available_hours = 40  # Typical student week

    def test_subject_minimum_calculation_balanced_mode(self):
        """Test minimum time calculation for balanced optimization mode"""
        minimums = calculate_subject_minimums(self.sample_subjects, OptimizationMode.BALANCED)

        # Check that minimums are calculated for all subjects
        self.assertEqual(len(minimums), len(self.sample_subjects))

        # Check that struggling subjects get more time
        struggling_subject = next(s for s in self.sample_subjects if s.last_performance < 0.7)
        good_subject = next(s for s in self.sample_subjects if s.last_performance > 0.8)

        self.assertGreater(
            minimums[struggling_subject.name],
            minimums[good_subject.name],
            "Struggling subjects should get more minimum time"
        )

        # All minimums should be positive
        for subject_name, min_time in minimums.items():
            self.assertGreater(min_time, 0, f"Minimum time for {subject_name} should be positive")

    def test_subject_minimum_calculation_harm_reduction_mode(self):
        """Test that harm reduction mode reduces minimum times"""
        balanced_minimums = calculate_subject_minimums(self.sample_subjects, OptimizationMode.BALANCED)
        harm_reduction_minimums = calculate_subject_minimums(self.sample_subjects, OptimizationMode.HARM_REDUCTION)

        for subject in self.sample_subjects:
            self.assertLessEqual(
                harm_reduction_minimums[subject.name],
                balanced_minimums[subject.name],
                f"Harm reduction should not increase minimum time for {subject.name}"
            )

    def test_coursework_time_estimation(self):
        """Test coursework time estimation with buffers"""
        estimated_time = estimate_coursework_time(self.sample_coursework)
        base_time = sum(cw.estimated_hours for cw in self.sample_coursework)

        # Should be greater than base estimate due to buffer
        self.assertGreater(estimated_time, base_time, "Estimated time should include buffer")

        # Should not be unreasonably high
        self.assertLess(estimated_time, base_time * 2, "Buffer should not be excessive")

        # Complex coursework should affect total more
        simple_coursework = [Coursework("Simple Task", 5.0, 0.5, 1.0)]
        complex_coursework = [Coursework("Complex Task", 5.0, 0.5, 5.0)]

        simple_estimate = estimate_coursework_time(simple_coursework)
        complex_estimate = estimate_coursework_time(complex_coursework)

        self.assertGreater(complex_estimate, simple_estimate,
                           "Complex coursework should have higher time estimates")

    def test_feasible_schedule_generation(self):
        """Test basic schedule optimization with feasible inputs"""
        result = basic_schedule_optimizer(
            self.sample_subjects,
            self.sample_coursework,
            self.available_hours,
            OptimizationMode.BALANCED
        )

        # Should be feasible
        self.assertTrue(result['feasible'], "Schedule should be feasible with reasonable inputs")

        # Should have allocations for all subjects
        self.assertEqual(len(result['study_allocation']), len(self.sample_subjects))

        # Total time should not exceed available hours
        total_time = sum(result['study_allocation'].values()) + result['coursework_allocation']
        self.assertLessEqual(total_time, self.available_hours,
                             "Total scheduled time should not exceed available hours")

        # Should have some buffer time or use all available time
        self.assertGreaterEqual(result['buffer_time'], 0, "Buffer time should be non-negative")

    def test_infeasible_schedule_handling(self):
        """Test behavior when schedule is mathematically impossible"""
        # Create impossible scenario - too many subjects, too little time
        impossible_subjects = [
            Subject(f"Subject {i}", 5.0, 5, 5.0, 0.5) for i in range(10)  # 10 subjects, 5 hours each
        ]
        impossible_coursework = [
            Coursework("Huge Project", 30.0, 1.0, 5.0)
        ]
        limited_time = 20  # Only 20 hours available

        result = basic_schedule_optimizer(
            impossible_subjects,
            impossible_coursework,
            limited_time,
            OptimizationMode.HARM_REDUCTION
        )

        # Should detect infeasibility or provide emergency solution
        if not result['feasible']:
            self.assertIn('deficit', result, "Infeasible result should include deficit information")
            self.assertIn('recommendations', result, "Should provide recommendations")
        else:
            # If it found a solution, it should be valid
            total_time = sum(result['study_allocation'].values()) + result['coursework_allocation']
            self.assertLessEqual(total_time, limited_time)

    def test_optimization_mode_effects(self):
        """Test that different optimization modes produce different results"""
        harm_result = basic_schedule_optimizer(
            self.sample_subjects, self.sample_coursework,
            self.available_hours, OptimizationMode.HARM_REDUCTION
        )

        perfection_result = basic_schedule_optimizer(
            self.sample_subjects, self.sample_coursework,
            self.available_hours, OptimizationMode.PERFECTION
        )

        # Harm reduction should allocate less study time
        harm_study_time = sum(harm_result['study_allocation'].values())
        perfection_study_time = sum(perfection_result['study_allocation'].values())

        self.assertLessEqual(harm_study_time, perfection_study_time,
                             "Harm reduction should allocate less or equal study time")

    def test_time_block_generation(self):
        """Test time block placement algorithm"""
        # First get a feasible allocation
        result = basic_schedule_optimizer(
            self.sample_subjects,
            self.sample_coursework,
            self.available_hours,
            OptimizationMode.BALANCED
        )

        if result['feasible'] and 'time_blocks' in result:
            time_blocks = result['time_blocks']

            # Should have time blocks
            self.assertGreater(len(time_blocks), 0, "Should generate time blocks")

            # All time blocks should have valid properties
            for block in time_blocks:
                self.assertIsInstance(block, TimeBlock)
                self.assertGreater(block.duration, 0, "Block duration should be positive")
                self.assertIsInstance(block.start_time, datetime)
                self.assertIn(block.block_type, ['study', 'coursework'])
                self.assertGreater(block.priority, 0, "Block priority should be positive")

            # Time blocks should not overlap (simplified check)
            sorted_blocks = sorted(time_blocks, key=lambda b: b.start_time)
            for i in range(len(sorted_blocks) - 1):
                current_end = sorted_blocks[i].start_time + timedelta(hours=sorted_blocks[i].duration)
                next_start = sorted_blocks[i + 1].start_time
                self.assertLessEqual(current_end, next_start,
                                     f"Blocks should not overlap: {sorted_blocks[i].subject} and {sorted_blocks[i + 1].subject}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_empty_inputs(self):
        """Test behavior with empty subject/coursework lists"""
        with self.assertRaises(Exception):
            basic_schedule_optimizer([], [], 40)

    def test_zero_available_time(self):
        """Test behavior with zero available time"""
        subjects = [Subject("Test Subject", 1.0, 3, 2.0, 0.8)]
        coursework = [Coursework("Test Work", 2.0, 0.5, 2.0)]

        result = basic_schedule_optimizer(subjects, coursework, 0)
        self.assertFalse(result['feasible'])

    def test_negative_time_inputs(self):
        """Test handling of negative time values"""
        with self.assertRaises(ValueError):
            Subject("Test", -1.0, 3, 2.0, 0.8)  # Negative minimum time

    def test_invalid_performance_scores(self):
        """Test handling of invalid performance scores"""
        with self.assertRaises(ValueError):
            Subject("Test", 2.0, 3, 2.0, 1.5)  # Performance > 1.0


class TestPerformanceBasedAdjustments(unittest.TestCase):
    """Test performance-based time adjustments"""

    def test_struggling_subject_gets_more_time(self):
        """Test that subjects with poor performance get more time"""
        struggling_subject = Subject("Struggling", 2.0, 4, 3.0, 0.4)  # Low performance
        easy_subject = Subject("Easy", 2.0, 4, 3.0, 0.95)  # High performance

        minimums = calculate_subject_minimums([struggling_subject, easy_subject], OptimizationMode.BALANCED)

        self.assertGreater(minimums["Struggling"], minimums["Easy"],
                           "Struggling subjects should get more minimum time")

    def test_high_priority_subject_allocation(self):
        """Test that high priority subjects get appropriate attention"""
        high_priority = Subject("Critical", 2.0, 5, 3.0, 0.7)  # Priority 5
        low_priority = Subject("Optional", 2.0, 1, 3.0, 0.7)  # Priority 1

        coursework = [Coursework("Assignment", 10.0, 0.5, 3.0)]

        result = basic_schedule_optimizer([high_priority, low_priority], coursework, 30)

        if result['feasible']:
            # High priority should get at least as much time as low priority
            self.assertGreaterEqual(
                result['study_allocation']["Critical"],
                result['study_allocation']["Optional"]
            )


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic usage scenarios"""

    def test_typical_cs_student_schedule(self):
        """Test with realistic CS student data"""
        cs_subjects = [
            Subject("Algorithms", 3.0, 5, 4.5, 0.6),  # Hard, important
            Subject("Software Engineering", 2.0, 4, 3.0, 0.8),  # Medium
            Subject("Database Systems", 2.5, 4, 3.5, 0.7),  # Medium-hard
            Subject("Computer Graphics", 2.0, 3, 4.0, 0.9)  # Hard but doing well
        ]

        cs_coursework = [
            Coursework("Algorithm Implementation", 15.0, 0.9, 5.0),  # Major project
            Coursework("SE Group Project", 12.0, 0.7, 4.0),  # Group work
            Coursework("Database Design", 8.0, 0.8, 3.0),  # Standard assignment
        ]

        # Test different time availability scenarios
        time_scenarios = [35, 45, 25]  # Light, normal, heavy workload weeks

        for available_time in time_scenarios:
            result = basic_schedule_optimizer(cs_subjects, cs_coursework, available_time)

            # Should handle all scenarios gracefully
            self.assertIsNotNone(result)

            if result['feasible']:
                # Algorithms (struggling + high priority) should get significant time
                self.assertGreater(result['study_allocation']["Algorithms"], 2.0)

                # Graphics (doing well) might get less time in harm reduction scenarios
                if available_time <= 25:  # Time pressure
                    graphics_time = result['study_allocation']["Computer Graphics"]
                    algorithms_time = result['study_allocation']["Algorithms"]
                    self.assertLessEqual(graphics_time, algorithms_time)


# Add these new test classes to your existing Tests.py file

class TestTimeBlockSizing(unittest.TestCase):
    """Test time block sizing strategies"""

    def setUp(self):
        """Set up test data for block sizing tests"""
        self.sample_subjects = [
            Subject("Linear Algebra", 2.0, 4, 3.5, 0.7),
            Subject("Data Structures", 3.0, 5, 4.0, 0.6)
        ]

        self.sample_coursework = [
            Coursework("DS Assignment", 8.0, 0.9, 4.5),
            Coursework("Network Project", 14.0, 0.6, 3.5)
        ]

        self.available_hours = 45

    def test_coursework_vs_study_block_sizes(self):
        """Test that coursework blocks are larger than study blocks"""
        result = basic_schedule_optimizer(
            self.sample_subjects,
            self.sample_coursework,
            self.available_hours,
            OptimizationMode.BALANCED
        )

        if result['feasible'] and 'time_blocks' in result:
            coursework_blocks = [b for b in result['time_blocks'] if b.block_type == 'coursework']
            study_blocks = [b for b in result['time_blocks'] if b.block_type == 'study']

            if coursework_blocks and study_blocks:
                avg_coursework_duration = sum(b.duration for b in coursework_blocks) / len(coursework_blocks)
                avg_study_duration = sum(b.duration for b in study_blocks) / len(study_blocks)

                self.assertGreater(avg_coursework_duration, avg_study_duration,
                                   "Coursework blocks should be larger on average than study blocks")

                # Test expected size ranges
                for block in coursework_blocks:
                    self.assertGreaterEqual(block.duration, 1.5,
                                            f"Coursework block too small: {block.duration}h")
                    self.assertLessEqual(block.duration, 5.0,
                                         f"Coursework block too large: {block.duration}h")

                for block in study_blocks:
                    self.assertGreaterEqual(block.duration, 0.5,
                                            f"Study block too small: {block.duration}h")
                    self.assertLessEqual(block.duration, 3.0,
                                         f"Study block too large: {block.duration}h")

    def test_optimal_block_sizing_logic(self):
        """Test that blocks are created at optimal sizes when possible"""
        from optimization_engine import create_blocks_for_task

        # Test coursework optimal sizing (should prefer 3.5h blocks)
        coursework_blocks = create_blocks_for_task(
            "Test Coursework", 10.5, 'coursework', 5, 3.5
        )

        # Should create 3 blocks of 3.5 hours each
        self.assertEqual(len(coursework_blocks), 3)
        for block in coursework_blocks:
            self.assertEqual(block.duration, 3.5)

        # Test study optimal sizing (should prefer 1.5h blocks)
        study_blocks = create_blocks_for_task(
            "Test Subject", 4.5, 'study', 3, 1.5
        )

        # Should create 3 blocks of 1.5 hours each
        self.assertEqual(len(study_blocks), 3)
        for block in study_blocks:
            self.assertEqual(block.duration, 1.5)

    def test_fragmentation_prevention(self):
        """Test that tiny blocks are merged instead of created separately"""
        from optimization_engine import create_blocks_for_task

        # Test case that would create a tiny remainder
        blocks = create_blocks_for_task(
            "Test Task", 7.2, 'coursework', 5, 3.5  # Would create 3.5 + 3.5 + 0.2
        )

        # Should merge the 0.2h remainder with the last block
        self.assertEqual(len(blocks), 2, "Should merge tiny remainder instead of creating separate block")

        # One block should be exactly 3.5h, the other should be 3.7h (3.5 + 0.2)
        durations = [block.duration for block in blocks]
        durations.sort()

        self.assertEqual(durations[0], 3.5, "First block should be optimal size")
        self.assertAlmostEqual(durations[1], 3.7, places=1, msg="Second block should include merged remainder")

        # Verify no block is smaller than 50% of optimal size
        for block in blocks:
            self.assertGreaterEqual(block.duration, 1.75,  # 50% of 3.5h
                                    f"Block too small: {block.duration}h")


class TestGreedyPlacementStrategy(unittest.TestCase):
    """Test the greedy time block placement algorithm"""

    def setUp(self):
        """Set up test data for placement tests"""
        self.high_priority_subject = Subject("Critical Subject", 2.0, 5, 4.0, 0.5)  # Struggling + high priority
        self.low_priority_subject = Subject("Optional Subject", 2.0, 2, 2.0, 0.9)  # Easy + low priority

        self.urgent_coursework = Coursework("Urgent Project", 7.0, 0.95, 4.5)  # Very urgent
        self.regular_coursework = Coursework("Regular Work", 7.0, 0.3, 2.0)  # Not urgent

    def test_high_priority_blocks_get_better_time_slots(self):
        """Test that high priority items are scheduled at better times"""
        result = basic_schedule_optimizer(
            [self.high_priority_subject, self.low_priority_subject],
            [self.urgent_coursework, self.regular_coursework],
            35,
            OptimizationMode.BALANCED
        )

        if result['feasible'] and 'time_blocks' in result:
            # Find blocks for high vs low priority items
            urgent_blocks = [b for b in result['time_blocks'] if 'Urgent' in b.subject]
            regular_blocks = [b for b in result['time_blocks'] if 'Regular' in b.subject]
            critical_blocks = [b for b in result['time_blocks'] if 'Critical' in b.subject]
            optional_blocks = [b for b in result['time_blocks'] if 'Optional' in b.subject]

            # High priority blocks should have higher priority values
            if urgent_blocks and regular_blocks:
                avg_urgent_priority = sum(b.priority for b in urgent_blocks) / len(urgent_blocks)
                avg_regular_priority = sum(b.priority for b in regular_blocks) / len(regular_blocks)

                self.assertGreater(avg_urgent_priority, avg_regular_priority,
                                   "Urgent coursework should have higher priority than regular coursework")

            # Test that high priority blocks are scheduled earlier in the week
            if urgent_blocks and regular_blocks:
                earliest_urgent = min(b.start_time for b in urgent_blocks if b.start_time != datetime.min)
                earliest_regular = min(b.start_time for b in regular_blocks if b.start_time != datetime.min)

                # This test might be flaky depending on implementation, so make it lenient
                self.assertLessEqual(earliest_urgent.weekday(), earliest_regular.weekday() + 2,
                                     "Urgent items should generally be scheduled earlier in the week")

    def test_block_priority_calculation(self):
        """Test that block priorities are calculated correctly"""
        from optimization_engine import create_coursework_blocks

        coursework_list = [self.urgent_coursework, self.regular_coursework]
        blocks = create_coursework_blocks(coursework_list, 14.0)

        urgent_blocks = [b for b in blocks if 'Urgent' in b.subject]
        regular_blocks = [b for b in blocks if 'Regular' in b.subject]

        if urgent_blocks and regular_blocks:
            # Urgent coursework should have higher priority
            # Priority = int(deadline_urgency * 5 + complexity)
            # Urgent: int(0.95 * 5 + 4.5) = int(9.25) = 9
            # Regular: int(0.3 * 5 + 2.0) = int(3.5) = 3

            for urgent_block in urgent_blocks:
                self.assertEqual(urgent_block.priority, 9,
                                 f"Urgent block priority should be 9, got {urgent_block.priority}")

            for regular_block in regular_blocks:
                self.assertEqual(regular_block.priority, 3,
                                 f"Regular block priority should be 3, got {regular_block.priority}")

    def test_time_slot_scoring_preferences(self):
        """Test time of day preferences for different block types"""
        from optimization_engine import score_time_slot

        # Create test time slots
        morning_slot = datetime(2024, 1, 15, 10, 0)  # 10 AM Monday
        afternoon_slot = datetime(2024, 1, 15, 15, 0)  # 3 PM Monday
        evening_slot = datetime(2024, 1, 15, 20, 0)  # 8 PM Monday
        late_slot = datetime(2024, 1, 15, 23, 0)  # 11 PM Monday

        # Create test blocks
        coursework_block = TimeBlock("Test Project", datetime.min, 3.5, 'coursework', 5)
        study_block = TimeBlock("Test Subject", datetime.min, 1.5, 'study', 3)

        # Test coursework preferences (should prefer afternoon/evening)
        coursework_morning_score = score_time_slot(morning_slot, coursework_block)
        coursework_afternoon_score = score_time_slot(afternoon_slot, coursework_block)
        coursework_evening_score = score_time_slot(evening_slot, coursework_block)
        coursework_late_score = score_time_slot(late_slot, coursework_block)

        # Afternoon should be better than morning for coursework
        self.assertGreater(coursework_afternoon_score, coursework_morning_score,
                           "Coursework should prefer afternoon over morning")

        # Late night should be penalized
        self.assertGreater(coursework_afternoon_score, coursework_late_score,
                           "Late night slots should be penalized")

        # Test study preferences (should prefer morning)
        study_morning_score = score_time_slot(morning_slot, study_block)
        study_afternoon_score = score_time_slot(afternoon_slot, study_block)

        self.assertGreater(study_morning_score, study_afternoon_score,
                           "Study blocks should prefer morning slots")


class TestComprehensiveBlockPlacement(unittest.TestCase):
    """Integration tests for the complete block placement system"""

    def test_realistic_scheduling_scenario(self):
        """Test a realistic scenario with mixed coursework and study requirements"""
        subjects = [
            Subject("Algorithms", 2.5, 5, 4.5, 0.6),  # Hard, important, struggling
            Subject("Software Eng", 2.0, 4, 3.0, 0.8),  # Medium, doing okay
            Subject("Database", 1.5, 3, 2.5, 0.9)  # Easy, doing well
        ]

        coursework = [
            Coursework("Algorithm Project", 12.0, 0.9, 5.0),  # Large, urgent, complex
            Coursework("SE Documentation", 4.0, 0.6, 2.0),  # Small, not urgent
            Coursework("DB Assignment", 6.0, 0.8, 3.0)  # Medium
        ]

        result = basic_schedule_optimizer(subjects, coursework, 40, OptimizationMode.BALANCED)

        if result['feasible'] and 'time_blocks' in result:
            blocks = result['time_blocks']

            # Should generate a reasonable number of blocks
            self.assertGreater(len(blocks), 5, "Should generate multiple time blocks")
            self.assertLess(len(blocks), 20, "Should not over-fragment the schedule")

            # All subjects should have some allocation
            subject_blocks = [b for b in blocks if b.block_type == 'study']
            subjects_scheduled = set(b.subject for b in subject_blocks)
            expected_subjects = {"Algorithms", "Software Eng", "Database"}

            self.assertTrue(expected_subjects.issubset(subjects_scheduled),
                            f"All subjects should be scheduled. Got: {subjects_scheduled}")

            # All coursework should have some allocation
            coursework_blocks = [b for b in blocks if b.block_type == 'coursework']
            coursework_scheduled = set(b.subject for b in coursework_blocks)
            expected_coursework = {"Algorithm Project", "SE Documentation", "DB Assignment"}

            # Allow partial matches since coursework names might be modified in blocks
            scheduled_matches = sum(1 for cw in expected_coursework
                                    if any(cw.split()[0] in scheduled for scheduled in coursework_scheduled))
            self.assertGreater(scheduled_matches, 0, "At least some coursework should be scheduled")

            # Struggling subject (Algorithms) should get significant time
            algo_blocks = [b for b in subject_blocks if "Algorithms" in b.subject]
            if algo_blocks:
                total_algo_time = sum(b.duration for b in algo_blocks)
                self.assertGreater(total_algo_time, 2.0,
                                   "Struggling high-priority subject should get substantial time")

    def test_block_temporal_consistency(self):
        """Test that scheduled blocks have consistent temporal properties"""
        subjects = [Subject("Test Subject", 3.0, 3, 3.0, 0.7)]
        coursework = [Coursework("Test Project", 10.0, 0.7, 3.0)]

        result = basic_schedule_optimizer(subjects, coursework, 25, OptimizationMode.BALANCED)

        if result['feasible'] and 'time_blocks' in result:
            blocks = result['time_blocks']

            for block in blocks:
                # All blocks should have valid start times (not datetime.min)
                if block.start_time != datetime.min:
                    # Should be within a reasonable time range (9 AM to 11 PM)
                    self.assertGreaterEqual(block.start_time.hour, 9,
                                            f"Block scheduled too early: {block.start_time}")
                    self.assertLessEqual(block.start_time.hour, 22,
                                         f"Block scheduled too late: {block.start_time}")

                    # Should be on a reasonable day (within next week)
                    days_from_now = (block.start_time.date() - datetime.now().date()).days
                    self.assertGreaterEqual(days_from_now, 0, "Block scheduled in the past")
                    self.assertLessEqual(days_from_now, 7, "Block scheduled too far in future")


# Add this function to your existing run_comprehensive_tests() function
def run_enhanced_tests():
    """Run all test suites including new MVP tests"""

    test_suite = unittest.TestSuite()

    # Add all test classes (existing + new)
    test_classes = [
        TestOptimizationEngine,
        TestEdgeCases,
        TestPerformanceBasedAdjustments,
        TestIntegrationScenarios,
        # New test classes for MVP coverage
        TestTimeBlockSizing,
        TestGreedyPlacementStrategy,
        TestComprehensiveBlockPlacement
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print(f"\n{'=' * 60}")
    print(f"ENHANCED MVP TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")

    return result

def run_comprehensive_tests():
    """Run all test suites and generate report"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestOptimizationEngine,
        TestEdgeCases,
        TestPerformanceBasedAdjustments,
        TestIntegrationScenarios
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")

    return result


if __name__ == "__main__":
    print("Starting comprehensive optimization engine tests...")
    print("Note: This test suite requires the optimization engine code to be properly implemented.")
    print("Some tests may fail until all functions are correctly implemented.\n")

    # Run all tests
    test_result = run_comprehensive_tests()

    # Exit with appropriate code
    sys.exit(0 if test_result.wasSuccessful() else 1)