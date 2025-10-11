
from pydantic import BaseModel


class CourseRequest(BaseModel):
    subjects: list = None
    prerequisites: list = None
    professors: list = None
    cohorts: list = None
    professor_by_subject: dict = None
    num_years: int = 2
    semesters_per_year: int = 2
    start_year: int = 2026
    max_classes_per_week: int = 2
    max_subjects_per_day_professor: int = 2
    max_subjects_per_day_cohort: int = 2
    week_days: list = None
    shifts: list = None
    blocked_slots: list = None