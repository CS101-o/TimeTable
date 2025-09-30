# Adaptive Study Scheduler

## Project Overview

A mathematical optimization-based timetable engine that intelligently allocates study time and schedules coursework while respecting lecture commitments and learning science principles.

## Current Implementation Status

### Phase 1: Resource Allocation (ILP) ✅ Complete

**Core Algorithm:** Integer Linear Programming using PuLP solver

**What it does:**
- Calculates optimal time allocation for subjects and coursework
- Adjusts minimums based on student performance (struggling students get more time)
- Applies complexity buffers to coursework estimates
- Supports three optimization modes:
  - **Harm Reduction:** Minimize study time during heavy coursework weeks
  - **Balanced:** Balance study and coursework priorities
  - **Perfection:** Optimize for learning when time permits

**Mathematical Formulation:**
```
Decision Variables: x_i (study hours), y_j (coursework hours), s (slack time)

Objective (Balanced Mode): 
  min Σ(x_i/p_i) + Σ(u_j × y_j) + α × s

Constraints:
  Σx_i + Σy_j + s = H (time budget)
  x_i ≥ min_i × φ(perf_i) (performance-adjusted minimums)
  y_j ≥ est_j × β_j (complexity-adjusted estimates)
  All variables ≥ 0
```

**Key Functions:**
- `phi(performance)`: Performance adjustment (1.3x for struggling, 0.7x for excelling)
- `beta(complexity)`: Complexity buffer (1.0 to 1.2x multiplier)
- `solve_resource_allocation()`: Main optimization solver

**Files:**
- `ILP.py`: Production-ready elegant implementation
- `ILP2.py`: Educational hardcoded version for learning
- `TestILP.py`: Unit tests

### Phase 2: Greedy Time Block Placement ✅ Complete

**Core Algorithm:** Priority-based greedy placement with heuristic scoring

**What it does:**
- Breaks time allocations into schedulable blocks (3.5h for coursework, 1.5h for study)
- Generates available time slots excluding lectures/workshops
- Places blocks using intelligent scoring:
  - Study sessions scheduled day before lectures (preparation principle)
  - Coursework prioritized during professional hours (9 AM - 6 PM)
  - Morning preference for study, afternoon for coursework
  - Avoids very late hours

**Key Features:**
- Respects fixed lecture schedule
- Implements spaced repetition principle
- Professional vs evening hour distinction
- Conflict detection and resolution

**Files:**
- `greedy_placement.py`: Time block generation and placement logic
- `integrated_scheduler.py`: Phase 1 → Phase 2 pipeline
- `your_schedule.py`: User's personal lecture schedule and subjects
- `test_my_schedule.py`: Main execution script

## Project Structure

```
timeTable/
├── ILP.py                      # Phase 1: Elegant ILP implementation
├── ILP2.py                     # Phase 1: Learning version (hardcoded)
├── greedy_placement.py         # Phase 2: Greedy placement algorithm
├── integrated_scheduler.py     # Integration pipeline
├── your_schedule.py            # Personal schedule configuration
├── test_my_schedule.py         # Main runner
├── TestILP.py                  # Unit tests for Phase 1
└── README.md                   # This file
```

## Usage Example

```python
from integrated_scheduler import create_complete_schedule
from your_schedule import my_lectures, my_subjects, my_coursework

result = create_complete_schedule(
    subjects=my_subjects,
    coursework=my_coursework,
    lecture_schedule=my_lectures,
    total_hours=40.0,
    optimization_mode="balanced"
)
```

**Output:** Complete weekly schedule with time blocks placed in optimal slots

## Current Limitations

1. **No break time insertion** - Continuous blocks may be unrealistic
2. **No daily hour caps** - Can generate 12+ hour days
3. **Basic block sizing** - Fixed optimal sizes (3.5h, 1.5h)
4. **No weekend balancing** - Tends to front-load weekdays
5. **Simple scoring function** - Doesn't consider energy levels, meal times
6. **No feedback loop yet** - Can't learn from actual performance

## Future Development Roadmap

### Phase 3: Enhanced Constraints & Realism

**Priority: High**

- [ ] Add maximum daily study hours constraint (e.g., 8-10 hours)
- [ ] Insert mandatory break times (15 min per 2 hours)
- [ ] Add meal time blocking (lunch 12-1, dinner 6-7)
- [ ] Implement energy level modeling (morning vs evening productivity)
- [ ] Add minimum sleep requirements (block 11 PM - 9 AM)
- [ ] Weekend vs weekday load balancing

**Technical Approach:**
- Add hard constraints in Phase 1 for daily maximums
- Modify greedy scorer to penalize consecutive long blocks
- Insert break blocks automatically in Phase 2

### Phase 4: Feedback Integration & Learning

**Priority: High**

- [ ] Collect actual time spent vs estimated
- [ ] Track focus/productivity ratings per session
- [ ] Quiz/assessment result integration
- [ ] Automatic minimum time adjustment based on performance
- [ ] Time estimation improvement (learn buffer factors)
- [ ] Personalized complexity scoring

**Technical Approach:**
- Database schema for session tracking
- Feedback processing service
- Update `phi()` and `beta()` based on historical data
- Reinforcement learning for weight optimization (advanced)

### Phase 5: Advanced Optimization Algorithms

**Priority: Medium**

- [ ] Implement true Linear Programming with OR-Tools
- [ ] Add Constraint Programming solver (CP-SAT)
- [ ] Genetic Algorithm for multi-objective optimization
- [ ] Simulated Annealing for escaping local optima
- [ ] A/B testing framework to compare algorithms
- [ ] Machine Learning for time prediction

**Technical Approach:**
- Abstract solver interface allowing algorithm swapping
- Benchmark suite with standard test cases
- Performance metrics (solution quality, execution time, user satisfaction)

### Phase 6: Backend & Frontend Development

**Priority: Medium**

**Backend (Node.js/Express):**
- [ ] RESTful API endpoints
- [ ] PostgreSQL database integration
- [ ] User authentication (JWT)
- [ ] Python bridge for optimization engine
- [ ] WebSocket for real-time updates
- [ ] Background job processing (schedule regeneration)

**Frontend (React/TypeScript):**
- [ ] Interactive calendar with drag-and-drop
- [ ] Time tracking interface with timers
- [ ] Analytics dashboard (performance trends)
- [ ] Subject/coursework management
- [ ] Optimization mode selector
- [ ] Manual override capabilities

### Phase 7: Advanced Features

**Priority: Low**

- [ ] Multi-user collaboration (study groups)
- [ ] Calendar integration (Google Calendar, Outlook)
- [ ] LMS integration (Canvas, Blackboard)
- [ ] Mobile app (React Native)
- [ ] Voice interface integration
- [ ] Spaced repetition algorithm (Anki-style)
- [ ] Pomodoro timer integration
- [ ] Gamification (streaks, achievements)

### Phase 8: Research & Innovation

**Priority: Research**

- [ ] Deep learning for optimal scheduling
- [ ] Natural language processing for task extraction
- [ ] Computer vision for study habit tracking
- [ ] Gradient descent for weight optimization
- [ ] Causal inference for performance attribution
- [ ] Academic paper publication

## Technical Dependencies

**Current:**
- Python 3.9+
- PuLP (Linear Programming solver)
- dataclasses (Python built-in)

**Planned:**
- Node.js 18+
- PostgreSQL 14+
- Redis (caching)
- React 18+
- TypeScript 5+
- OR-Tools (advanced optimization)

## Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd timeTable

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pulp

# Configure your schedule
# Edit your_schedule.py with your lectures, subjects, and coursework

# Run the scheduler
python test_my_schedule.py
```

## Known Issues

1. Greedy placement sometimes generates odd decimal durations (e.g., 3.18h)
2. No validation for overlapping lectures
3. Slack time not explicitly scheduled (just calculated)
4. Performance scores must be manually set (no automatic tracking yet)

## Contributing

This is currently a personal learning project. Contributions welcome after Phase 4 completion.

## License

[To be determined]

## Acknowledgments

Built as a solution to the common student problem of coursework overwhelming weekly study commitments. Inspired by operations research, learning science, and personal scheduling struggles.

---

**Last Updated:** January 2025  
**Status:** Phase 2 Complete, Phase 3 Planning
