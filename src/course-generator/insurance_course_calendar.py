
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyswarms as ps
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseGenerator:
	def __init__(self,
		subjects=None,
		prerequisites=None,
		professors=None,
		cohorts=None,
		professor_by_subject=None,
		num_years=2,
		semesters_per_year=2,
		start_year=2026,
		max_classes_per_week=2,
		max_subjects_per_day_professor=2,
		max_subjects_per_day_cohort=2,
		week_days=None,
		shifts=None,
		blocked_slots=None,
		print_image_result=False,
		print_excel_result=True,
		logger=logger
	):
		self.subjects = subjects 
		self.prerequisites = prerequisites 
		self.professors = professors 
		self.cohorts = cohorts 
		self.professor_by_subject = professor_by_subject 
		self.num_years = num_years
		self.semesters_per_year = semesters_per_year
		self.start_year = start_year
		self.max_classes_per_week = max_classes_per_week
		self.max_subjects_per_day_professor = max_subjects_per_day_professor
		self.max_subjects_per_day_cohort = max_subjects_per_day_cohort
		self.week_days = week_days
		self.shifts = shifts 
		self.blocked_slots = blocked_slots
		self.print_image_result = print_image_result
		self.print_excel_result = print_excel_result
		self.logger = logger
		self.num_subjects = len(self.subjects)
		self.num_professors = len(self.professors)
		self.num_cohorts = len(self.cohorts)
		self.num_semesters = self.num_years * self.semesters_per_year
		self.dim = self.num_subjects * self.num_cohorts * 4
		self.max_iters = 200
		self.bounds = (np.zeros(self.dim), np.ones(self.dim) * max(self.num_semesters, len(self.week_days), len(self.shifts)))
		self.logger.info(f'Config: subjects={self.num_subjects}, professors={self.num_professors}, cohorts={self.num_cohorts}, years={self.num_years}, semesters={self.num_semesters}')
		self.colors = ['tab:blue','tab:orange','tab:green','tab:red']

	def decode_solution(self, x):
		self.logger.debug('Decoding solution vector')
		sol = []
		for i in range(self.num_subjects):
			for cohort_idx in range(self.num_cohorts):
				base = i*self.num_cohorts*4 + cohort_idx*4
				semester = int(round(x[base])) % self.num_semesters
				day = int(round(x[base+1])) % len(self.week_days)
				shift = int(round(x[base+2])) % len(self.shifts)
				prof = self.professor_by_subject[self.subjects[i]]
				cohort = self.cohorts[cohort_idx]
				sol.append((self.subjects[i], semester, prof, cohort, self.week_days[day], self.shifts[shift]))
		self.logger.debug(f'Decoded solution length: {len(sol)}')
		return sol






	def objective(self, x):
		self.logger.debug('Evaluating objective function')
		sol = self.decode_solution(x)
		penalty = 0
		if self.num_subjects % self.num_semesters == 0:
			expected_subjects = self.num_subjects // self.num_semesters
		else:
			expected_subjects = int(round(self.num_subjects / self.num_semesters))
		for cohort in self.cohorts:
			subjects_per_semester = [0]*self.num_semesters
			for _, semester, _, coh, _, _ in sol:
				if coh == cohort:
					subjects_per_semester[semester] += 1
			for semester_idx, count in enumerate(subjects_per_semester):
				if count != expected_subjects:
					penalty += 1e6 * abs(count - expected_subjects)
		for cohort in self.cohorts:
			semester_dict = {m: s for m, s, _, coh, _, _ in sol if coh == cohort}
			for subj, prereq in self.prerequisites:
				if semester_dict[subj] <= semester_dict[prereq]:
					penalty += 1000
		# Penalizar bloques bloqueados
		for _, _, _, _, day, shift in sol:
			if (day, shift) in self.blocked_slots:
				penalty += 5000
		# Penalizar exceso de materias por día por cohorte y cuatrimestre
		for cohort in self.cohorts:
			for semester in range(self.num_semesters):
				subjects_per_day = {d: 0 for d in self.week_days}
				for _, s, _, coh, day, _ in sol:
					if coh == cohort and s == semester:
						subjects_per_day[day] += 1
				for d in self.week_days:
					for prof in self.professors:
						subjects_prof_day = 0
						for _, s, p, coh, day, _ in sol:
							if coh == cohort and s == semester and day == d and p == prof:
								subjects_prof_day += 1
						if subjects_prof_day > self.max_subjects_per_day_professor:
							penalty += 50 * (subjects_prof_day - self.max_subjects_per_day_professor)
				attended_days = sum(1 for v in subjects_per_day.values() if v > 0)
				if attended_days > self.max_subjects_per_day_cohort:
					penalty += 200 * (attended_days - self.max_subjects_per_day_cohort)
		for prof in self.professors:
			for d in self.week_days:
				subjects_prof_day = 0
				for _, _, p, _, day, _ in sol:
					if p == prof and day == d:
						subjects_prof_day += 1
				if subjects_prof_day > self.max_subjects_per_day_professor:
					penalty += 100 * (subjects_prof_day - self.max_subjects_per_day_professor)
		for cohort in self.cohorts:
			for d in self.week_days:
				subjects_cohort_day = 0
				for _, _, _, coh, day, _ in sol:
					if coh == cohort and day == d:
						subjects_cohort_day += 1
				if subjects_cohort_day > self.max_subjects_per_day_cohort:
					penalty += 200 * (subjects_cohort_day - self.max_subjects_per_day_cohort)
		slots = set()
		for subj, semester, prof, cohort, day, shift in sol:
			key = (cohort, semester, day, shift)
			if key in slots:
				penalty += 100
			else:
				slots.add(key)
		days_count = {d: 0 for d in self.week_days}
		for _, _, _, _, day, _ in sol:
			days_count[day] += 1
		penalty += np.std(list(days_count.values())) * 100
		for d in self.week_days:
			shifts_used = set()
			for _, _, _, _, day, shift in sol:
				if day == d:
					shifts_used.add(shift)
			if len(shifts_used) < len(self.shifts):
				penalty += 200 * (len(self.shifts) - len(shifts_used))
		return penalty

	def objective_pyswarms(self, x):
		return np.array([self.objective(xi) for xi in x])

	def run_pso_optimizer(self):
		options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
		self.logger.info('Initializing PSO optimizer')
		optimizer = ps.single.GlobalBestPSO(
			n_particles=60,
			dimensions=self.dim,
			options=options,
			bounds=self.bounds
		)
		self.logger.info('Starting PSO optimization')
		cost, pos = optimizer.optimize(self.objective_pyswarms, iters=self.max_iters)
		self.logger.info(f'Optimization finished. Best cost: {cost}')
		self.convergencia = optimizer.cost_history
		self.best_x = pos
		return cost, pos, self.convergencia


	def get_solution(self):
		return self.decode_solution(self.best_x) if hasattr(self, 'best_x') else None


