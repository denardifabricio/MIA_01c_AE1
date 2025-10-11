
from fastapi import FastAPI
from insurance_course_calendar import CourseGenerator
import logging
from model.course_request import CourseRequest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/course")
def create_course(request: CourseRequest):

    logger.info(f"Generator initialized with: subjects={request.subjects}, professors={request.professors}, cohorts={request.cohorts}, years={request.num_years}, semesters={request.semesters_per_year}")
    logger.info(f"semesters {request.semesters_per_year}, start_year {request.start_year}, max_classes_per_week {request.max_classes_per_week}, max_subjects_per_day_professor {request.max_subjects_per_day_professor}, max_subjects_per_day_cohort {request.max_subjects_per_day_cohort}")
    logger.info(f"week_days {request.week_days}, shifts {request.shifts}, blocked_slots {request.blocked_slots}")


    generator = CourseGenerator(
        subjects=request.subjects,
        prerequisites=request.prerequisites,
        professors=request.professors,
        cohorts=request.cohorts,
        professor_by_subject=request.professor_by_subject,
        num_years=request.num_years,
        semesters_per_year=request.semesters_per_year,
        start_year=request.start_year,
        max_classes_per_week=request.max_classes_per_week,
        max_subjects_per_day_professor=request.max_subjects_per_day_professor,
        max_subjects_per_day_cohort=request.max_subjects_per_day_cohort,
        week_days=request.week_days,
        shifts=request.shifts,
        blocked_slots=request.blocked_slots
    )

    cost, pos, convergence = generator.run_pso_optimizer()
    solution = generator.get_solution()
    return {
        "cost": cost,
        "solution": solution,
        "convergence": list(convergence)
    }

