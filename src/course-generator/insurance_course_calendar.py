
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
		pso_options=None,
		n_particles=60,
		max_iters=200,
		penalty_weights=None,
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
		
		# Parámetros del PSO
		self.pso_options = pso_options if pso_options is not None else {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
		self.n_particles = n_particles
		self.max_iters = max_iters
		
		# Pesos de penalización configurables
		default_weights = {
			'semester_distribution': 1e6,
			'prerequisites': 5000,
			'blocked_slots': 10000,
			'professor_overload': 100,
			'cohort_overload': 200,
			'slot_conflicts': 500,
			'day_balance': 50,
			'shift_usage': 100
		}
		self.penalty_weights = penalty_weights if penalty_weights is not None else default_weights
		
		self.num_subjects = len(self.subjects)
		self.num_professors = len(self.professors)
		self.num_cohorts = len(self.cohorts)
		self.num_semesters = self.num_years * self.semesters_per_year
		self.dim = self.num_subjects * self.num_cohorts * 4
		self.bounds = (np.zeros(self.dim), np.ones(self.dim) * max(self.num_semesters, len(self.week_days), len(self.shifts)))
		
		# Validar y procesar prerequisitos
		self._validate_prerequisites()
		self._build_prerequisite_graph()
		
		self.logger.info(f'Config: subjects={self.num_subjects}, professors={self.num_professors}, cohorts={self.num_cohorts}, years={self.num_years}, semesters={self.num_semesters}')
		self.colors = ['tab:blue','tab:orange','tab:green','tab:red']

	def _validate_prerequisites(self):
		"""Validar que todos los prerequisitos sean válidos."""
		if not self.prerequisites:
			self.prerequisites = []
			return
		
		subject_set = set(self.subjects)
		valid_prerequisites = []
		
		for prereq_pair in self.prerequisites:
			if len(prereq_pair) != 2:
				self.logger.warning(f"Prerequisito inválido (debe tener 2 elementos): {prereq_pair}")
				continue
			
			subject, prereq = prereq_pair
			if subject not in subject_set:
				self.logger.warning(f"Materia '{subject}' no encontrada en prerequisito: {prereq_pair}")
				continue
			if prereq not in subject_set:
				self.logger.warning(f"Prerequisito '{prereq}' no encontrado en prerequisito: {prereq_pair}")
				continue
			if subject == prereq:
				self.logger.warning(f"Una materia no puede ser prerequisito de sí misma: {prereq_pair}")
				continue
			
			valid_prerequisites.append((subject, prereq))
		
		self.prerequisites = valid_prerequisites
		self.logger.info(f"Prerequisitos válidos: {len(self.prerequisites)}")
	
	def _build_prerequisite_graph(self):
		"""Construir grafo de prerequisitos para detectar ciclos y calcular niveles."""
		self.prerequisite_map = {}  # materia -> lista de prerequisitos
		self.dependent_map = {}     # prerequisito -> lista de materias que lo requieren
		
		# Inicializar mapas
		for subject in self.subjects:
			self.prerequisite_map[subject] = []
			self.dependent_map[subject] = []
		
		# Llenar mapas
		for subject, prereq in self.prerequisites:
			self.prerequisite_map[subject].append(prereq)
			self.dependent_map[prereq].append(subject)
		
		# Detectar ciclos y calcular niveles mínimos
		self._detect_cycles()
		self._calculate_minimum_levels()
	
	def _detect_cycles(self):
		"""Detectar ciclos en el grafo de prerequisitos usando DFS."""
		visited = set()
		rec_stack = set()
		
		def dfs(subject):
			visited.add(subject)
			rec_stack.add(subject)
			
			for prereq in self.prerequisite_map[subject]:
				if prereq not in visited:
					if dfs(prereq):
						return True
				elif prereq in rec_stack:
					self.logger.error(f"Ciclo detectado en prerequisitos: {subject} -> {prereq}")
					return True
			
			rec_stack.remove(subject)
			return False
		
		for subject in self.subjects:
			if subject not in visited:
				if dfs(subject):
					raise ValueError("Se detectaron ciclos en los prerequisitos")
	
	def _calculate_minimum_levels(self):
		"""Calcular el nivel mínimo (cuatrimestre) para cada materia."""
		self.min_levels = {}
		
		def calculate_level(subject):
			if subject in self.min_levels:
				return self.min_levels[subject]
			
			if not self.prerequisite_map[subject]:
				# Sin prerequisitos, puede ir en el primer cuatrimestre
				self.min_levels[subject] = 0
				return 0
			
			# El nivel mínimo es 1 + el máximo nivel de sus prerequisitos
			max_prereq_level = max(calculate_level(prereq) for prereq in self.prerequisite_map[subject])
			self.min_levels[subject] = max_prereq_level + 1
			return self.min_levels[subject]
		
		for subject in self.subjects:
			calculate_level(subject)
		
		self.logger.info(f"Niveles mínimos calculados: {self.min_levels}")

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
		"""Función objetivo mejorada con mejor manejo de prerequisitos y eficiencia."""
		self.logger.debug('Evaluating objective function')
		sol = self.decode_solution(x)
		penalty = 0
		
		# Estructuras de datos para análisis eficiente
		cohort_semester_subjects = {}
		cohort_semester_schedule = {}
		professor_day_count = {}
		cohort_day_count = {}
		slot_usage = {}
		
		# Procesar solución una sola vez
		for subject, semester, professor, cohort, day, shift in sol:
			# Organizar por cohorte y cuatrimestre
			key = (cohort, semester)
			if key not in cohort_semester_subjects:
				cohort_semester_subjects[key] = []
				cohort_semester_schedule[key] = {}
			cohort_semester_subjects[key].append(subject)
			
			# Registro de horarios
			day_key = (cohort, semester, day)
			if day_key not in cohort_semester_schedule[key]:
				cohort_semester_schedule[key][day_key] = []
			cohort_semester_schedule[key][day_key].append((subject, professor, shift))
			
			# Contadores para profesores y cohortes
			prof_day_key = (professor, day)
			cohort_day_key = (cohort, day)
			professor_day_count[prof_day_key] = professor_day_count.get(prof_day_key, 0) + 1
			cohort_day_count[cohort_day_key] = cohort_day_count.get(cohort_day_key, 0) + 1
			
			# Uso de slots
			slot_key = (cohort, semester, day, shift)
			if slot_key in slot_usage:
				penalty += self.penalty_weights['slot_conflicts']
			else:
				slot_usage[slot_key] = True
			
			# Penalizar bloques bloqueados
			if (day, shift) in self.blocked_slots:
				penalty += self.penalty_weights['blocked_slots']
		
		# 1. Distribución de materias por cuatrimestre
		penalty += self._evaluate_semester_distribution(cohort_semester_subjects)
		
		# 2. Verificar prerequisitos (mejorado)
		penalty += self._evaluate_prerequisites(sol)
		
		# 3. Sobrecarga de profesores
		for count in professor_day_count.values():
			if count > self.max_subjects_per_day_professor:
				penalty += self.penalty_weights['professor_overload'] * (count - self.max_subjects_per_day_professor)
		
		# 4. Sobrecarga de cohortes
		for count in cohort_day_count.values():
			if count > self.max_subjects_per_day_cohort:
				penalty += self.penalty_weights['cohort_overload'] * (count - self.max_subjects_per_day_cohort)
		
		# 5. Balance de días
		penalty += self._evaluate_day_balance(sol)
		
		# 6. Uso de turnos
		penalty += self._evaluate_shift_usage(sol)
		
		return penalty
	
	def _evaluate_semester_distribution(self, cohort_semester_subjects):
		"""Evaluar distribución de materias por cuatrimestre (más flexible)."""
		penalty = 0
		
		for cohort in self.cohorts:
			subjects_per_semester = [0] * self.num_semesters
			
			# Contar materias por cuatrimestre
			for semester in range(self.num_semesters):
				key = (cohort, semester)
				if key in cohort_semester_subjects:
					subjects_per_semester[semester] = len(cohort_semester_subjects[key])
			
			# Calcular distribución ideal (más flexible)
			total_subjects = self.num_subjects
			ideal_per_semester = total_subjects / self.num_semesters
			min_per_semester = max(1, int(ideal_per_semester * 0.7))  # 70% del ideal como mínimo
			max_per_semester = int(ideal_per_semester * 1.3)  # 130% del ideal como máximo
			
			for count in subjects_per_semester:
				if count < min_per_semester:
					penalty += self.penalty_weights['semester_distribution'] * (min_per_semester - count)
				elif count > max_per_semester:
					penalty += self.penalty_weights['semester_distribution'] * (count - max_per_semester)
		
		return penalty
	
	def _evaluate_prerequisites(self, sol):
		"""Evaluación mejorada de prerequisitos."""
		penalty = 0
		
		for cohort in self.cohorts:
			# Crear mapeo de materia -> cuatrimestre para esta cohorte
			semester_dict = {}
			for subject, semester, _, coh, _, _ in sol:
				if coh == cohort:
					semester_dict[subject] = semester
			
			# Verificar prerequisitos
			for subject, prereq in self.prerequisites:
				if subject in semester_dict and prereq in semester_dict:
					subject_semester = semester_dict[subject]
					prereq_semester = semester_dict[prereq]
					
					# El prerequisito debe estar en un cuatrimestre ANTERIOR
					if subject_semester <= prereq_semester:
						# Penalización proporcional a la violación
						violation_severity = (prereq_semester - subject_semester + 1)
						penalty += self.penalty_weights['prerequisites'] * violation_severity
						
						self.logger.debug(f"Prerequisito violado en {cohort}: {subject} (sem {subject_semester+1}) requiere {prereq} (sem {prereq_semester+1})")
				
				elif subject in semester_dict and prereq not in semester_dict:
					# Materia sin prerequisito asignado
					penalty += self.penalty_weights['prerequisites'] * 0.5
				elif subject not in semester_dict and prereq in semester_dict:
					# Prerequisito sin materia asignada
					penalty += self.penalty_weights['prerequisites'] * 0.5
		
		return penalty
	
	def _evaluate_day_balance(self, sol):
		"""Evaluar balance en el uso de días."""
		penalty = 0
		
		days_count = {d: 0 for d in self.week_days}
		for _, _, _, _, day, _ in sol:
			days_count[day] += 1
		
		# Penalizar desbalance en uso de días
		avg_usage = sum(days_count.values()) / len(days_count)
		for count in days_count.values():
			deviation = abs(count - avg_usage)
			penalty += self.penalty_weights['day_balance'] * deviation
		
		return penalty
	
	def _evaluate_shift_usage(self, sol):
		"""Evaluar uso de turnos por día."""
		penalty = 0
		
		for day in self.week_days:
			shifts_used = set()
			for _, _, _, _, d, shift in sol:
				if d == day:
					shifts_used.add(shift)
			
			# Penalizar si no se usan todos los turnos disponibles
			unused_shifts = len(self.shifts) - len(shifts_used)
			penalty += self.penalty_weights['shift_usage'] * unused_shifts
		
		return penalty

	def analyze_solution_quality(self, solution=None):
		"""Analizar la calidad de la solución de manera detallada."""
		if solution is None:
			solution = self.get_solution()
		
		if not solution:
			return {"error": "No hay solución disponible"}
		
		analysis = {
			"total_assignments": len(solution),
			"prerequisite_violations": 0,
			"blocked_slot_violations": 0,
			"professor_overloads": 0,
			"cohort_overloads": 0,
			"slot_conflicts": 0,
			"semester_distribution": {},
			"day_usage": {},
			"shift_usage": {},
			"prerequisite_details": []
		}
		
		# Análisis por cohorte
		for cohort in self.cohorts:
			cohort_assignments = [a for a in solution if a[3] == cohort]
			semester_count = {}
			
			for subject, semester, professor, coh, day, shift in cohort_assignments:
				if semester not in semester_count:
					semester_count[semester] = 0
				semester_count[semester] += 1
			
			analysis["semester_distribution"][cohort] = semester_count
		
		# Análisis de prerequisitos
		for cohort in self.cohorts:
			semester_dict = {}
			for subject, semester, _, coh, _, _ in solution:
				if coh == cohort:
					semester_dict[subject] = semester
			
			for subject, prereq in self.prerequisites:
				if subject in semester_dict and prereq in semester_dict:
					subject_sem = semester_dict[subject]
					prereq_sem = semester_dict[prereq]
					
					if subject_sem <= prereq_sem:
						analysis["prerequisite_violations"] += 1
						analysis["prerequisite_details"].append({
							"cohort": cohort,
							"subject": subject,
							"subject_semester": subject_sem + 1,
							"prerequisite": prereq,
							"prerequisite_semester": prereq_sem + 1
						})
		
		# Análisis de otros aspectos
		slot_usage = set()
		day_count = {}
		shift_count = {}
		professor_day_count = {}
		cohort_day_count = {}
		
		for subject, semester, professor, cohort, day, shift in solution:
			# Conflictos de slots
			slot_key = (cohort, semester, day, shift)
			if slot_key in slot_usage:
				analysis["slot_conflicts"] += 1
			slot_usage.add(slot_key)
			
			# Uso de días y turnos
			day_count[day] = day_count.get(day, 0) + 1
			shift_count[shift] = shift_count.get(shift, 0) + 1
			
			# Sobrecarga de profesores y cohortes
			prof_day_key = (professor, day)
			cohort_day_key = (cohort, day)
			professor_day_count[prof_day_key] = professor_day_count.get(prof_day_key, 0) + 1
			cohort_day_count[cohort_day_key] = cohort_day_count.get(cohort_day_key, 0) + 1
			
			# Bloques bloqueados
			if (day, shift) in self.blocked_slots:
				analysis["blocked_slot_violations"] += 1
		
		# Contar sobrecargas
		for count in professor_day_count.values():
			if count > self.max_subjects_per_day_professor:
				analysis["professor_overloads"] += 1
		
		for count in cohort_day_count.values():
			if count > self.max_subjects_per_day_cohort:
				analysis["cohort_overloads"] += 1
		
		analysis["day_usage"] = day_count
		analysis["shift_usage"] = shift_count
		
		# Calcular score de calidad (0-100, donde 100 es perfecto)
		total_issues = (analysis["prerequisite_violations"] + 
						analysis["blocked_slot_violations"] + 
						analysis["professor_overloads"] + 
						analysis["cohort_overloads"] + 
						analysis["slot_conflicts"])
		
		max_possible_issues = len(self.prerequisites) * len(self.cohorts) + len(solution)
		quality_score = max(0, 100 - (total_issues / max_possible_issues * 100)) if max_possible_issues > 0 else 100
		analysis["quality_score"] = round(quality_score, 2)
		
		return analysis

	def objective_pyswarms(self, x):
		return np.array([self.objective(xi) for xi in x])

	def run_pso_optimizer(self):
		self.logger.info(f'Initializing PSO optimizer with options: {self.pso_options}')
		self.logger.info(f'Particles: {self.n_particles}, Max iterations: {self.max_iters}')
		
		optimizer = ps.single.GlobalBestPSO(
			n_particles=self.n_particles,
			dimensions=self.dim,
			options=self.pso_options,
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


