import streamlit as st
from datetime import datetime
import logging
import requests
import os
import pandas as pd
import json

from model.course_request import CourseRequest
import numpy as np

# Sidebar descriptivo
st.sidebar.title("Descripción")
st.sidebar.markdown(
    """
   
# Maestría en Inteligencia Artificial

## Trabajo práctico final

### Algoritmos Evolutivos I

**Autores:**  
Esp. Ing. Fabricio Denardi  
Esp. Ing. Bruno  Masoller   
Esp. Lic. Noelia Qualindi  

**Docente:** 
Esp. Ing. Miguel Augusto Azar  

Esta aplicación permite generar y visualizar el calendario académico para el armado de un curso utilizando algoritmos evolutivos.
    
En el panel principal puedes configurar los parámetros y generar el calendario optimizado para cada cohorte y cuatrimestre.


Ciudad de Buenos Aires, Argentina 2025
    """
)
# Cargar configuración de ejemplo desde JSON
with open(os.path.join(os.path.dirname(__file__), "config_example.json"), encoding="utf-8") as f:
    config_example = json.load(f)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_url = f'{os.environ.get("COURSE_API_URL", "http://localhost:8000")}/course'
logger.info(f"API URL: {api_url}")


logger.info("Iniciando la aplicación de calendario")

st.set_page_config(page_title="Calendario", layout="wide")

st.title("Calendario Interactivo")


# Formulario de configuración inicial
st.subheader("Configuración Inicial del Calendario del curso")


with st.form("config_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        subjects = st.text_area(
            "Materias (una por línea)",
            value="\n".join(config_example["subjects"])
        )
        professors = st.text_area(
            "Profesores (una por línea)",
            value="\n".join(config_example["professors"])
        )
        cohorts = st.text_area(
            "Cohortes (una por línea)",
            value="\n".join(config_example["cohorts"])
        )
        week_days = st.multiselect(
            "Días de la semana",
            options=["Lunes","Martes","Miércoles","Jueves","Viernes"],
            default=config_example["week_days"]
        )
        professor_by_subject_input = st.text_area(
            "Profesor por materia (formato: materia,profesor por línea)",
            value="\n".join([f"{m},{p}" for m, p in config_example["professor_by_subject"]])
        )
    with col2:
        prerequisites = st.text_area(
            "Correlativas (formato: materia,prerequisito por línea)",
            value="\n".join([f"{m},{p}" for m, p in config_example["prerequisites"]])
        )
        shifts = st.multiselect(
            "Turnos",
            options=["Mañana","Tarde"],
            default=config_example["shifts"]
        )
        blocked_slots = st.text_area(
            "Bloques bloqueados (formato: día,turno por línea)",
            value="\n".join([f"{d},{t}" for d, t in config_example["blocked_slots"]])
        )
    with col3:
        num_years = st.number_input(
            "Años", min_value=1, max_value=10, value=config_example["num_years"]
        )
        semesters_per_year = st.number_input(
            "Cuatrimestres por año", min_value=1, max_value=4, value=config_example["semesters_per_year"]
        )
        start_year = st.number_input(
            "Año de inicio", min_value=2020, max_value=2100, value=config_example["start_year"]
        )
        max_classes_per_week = st.number_input(
            "Máx. clases por semana", min_value=1, max_value=10, value=config_example["max_classes_per_week"]
        )
        max_subjects_per_day_professor = st.number_input(
            "Máx. materias por día por profesor", min_value=1, max_value=10, value=config_example["max_subjects_per_day_professor"]
        )
        max_subjects_per_day_cohort = st.number_input(
            "Máx. días por cuatrimestre por cohorte", min_value=1, max_value=10, value=config_example["max_subjects_per_day_cohort"]
        )
    submitted = st.form_submit_button("Generar Calendario")



if submitted:


    logger.info("Submit recibido. Procesando datos del formulario...")
    logger.info(f"Cohortes: {cohorts}")

    

    if submitted:
        with st.spinner("Generando calendario, por favor espere..."):
            logger.info("Submit recibido. Procesando datos del formulario...")
            logger.info(f"Cohortes: {cohorts}")
            logger.info(f"Materias: {subjects}")
            logger.info(f"Profesores: {professors}")
            logger.info(f"Correlativas: {prerequisites}")
            logger.info(f"Días de la semana: {week_days}")
            logger.info(f"Turnos: {shifts}")
            logger.info(f"Bloques bloqueados: {blocked_slots}")
            logger.info(f"Años: {num_years}, Cuatrimestres por año: {semesters_per_year}, Año de inicio: {start_year}")
            logger.info(f"Máx. clases por semana: {max_classes_per_week}, Máx. materias por día por profesor: {max_subjects_per_day_professor}, Máx. días por cuatrimestre por cohorte: {max_subjects_per_day_cohort}")

            cohort_list = [c.strip() for c in cohorts.splitlines() if c.strip()]
            subject_list = [s.strip() for s in subjects.splitlines() if s.strip()]
            prerequisite_list = [(a.strip(), b.strip()) for a, b in (line.split(',') for line in prerequisites.splitlines() if ',' in line)]
            professor_list = [p.strip() for p in professors.splitlines() if p.strip()]
            week_days_list = week_days
            shifts_list = shifts
            blocked_slots_list = [(d.strip(), t.strip()) for d, t in (line.split(',') for line in blocked_slots.splitlines() if ',' in line)]
            professor_by_subject = {}
            for line in professor_by_subject_input.splitlines():
                if ',' in line:
                    materia, profesor = line.split(',', 1)
                    professor_by_subject[materia.strip()] = profesor.strip()

            logger.info(f"cohort_list: {cohort_list}")
            logger.info(f"subject_list: {subject_list}")
            logger.info(f"prerequisite_list: {prerequisite_list}")
            logger.info(f"professor_list: {professor_list}")
            logger.info(f"week_days_list: {week_days_list}")
            logger.info(f"shifts_list: {shifts_list}")
            logger.info(f"blocked_slots_list: {blocked_slots_list}")

            course_request = CourseRequest(
                subjects=subject_list,
                prerequisites=prerequisite_list,
                professors=professor_list,
                cohorts=cohort_list,
                professor_by_subject=professor_by_subject,
                num_years=num_years,
                semesters_per_year=semesters_per_year,
                start_year=start_year,
                max_classes_per_week=max_classes_per_week,
                max_subjects_per_day_professor=max_subjects_per_day_professor,
                max_subjects_per_day_cohort=max_subjects_per_day_cohort,
                week_days=week_days_list,
                shifts=shifts_list,
                blocked_slots=blocked_slots_list
            )
            
            logger.info(f"Enviando datos a la API /course.: {course_request.model_dump()}")
            response = requests.post(api_url, json=course_request.model_dump())
            if response.status_code == 200:
                result = response.json()
                cost = result.get("cost")
                solution = result.get("solution")
                convergence = result.get("convergence")
                logger.info(f"Respuesta de la API recibida. Penalidad: {cost}")

                st.success(f"Optimización completada. Penalidad: {cost}")
            else:
                st.error(f"Error al obtener datos de la API: {response.status_code}")
                logger.error(f"Error al obtener datos de la API: {response.text}")

            # Organizar la solución por cohorte y cuatrimestre
            num_semesters = num_years * semesters_per_year
            cohort_semester_subjects = {cohort: {sem: [] for sem in range(num_semesters)} for cohort in cohort_list}
            for subj, semester, prof, cohort, day, shift in solution:
                cohort_semester_subjects[cohort][semester].append((subj, prof, day, shift))
            logger.info(f"cohort_semester_subjects: {cohort_semester_subjects}")

            import matplotlib.pyplot as plt

            tabs = st.tabs(["Inicio"] + cohort_list)
            with tabs[0]:
                st.subheader("Convergencia del Algoritmo")
                if convergence:
                    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                    ax_conv.set_title('Convergencia PSO (pyswarms)', fontsize=18, fontweight='bold')
                    ax_conv.set_xlabel('Iteración', fontsize=14)
                    ax_conv.set_ylabel('Penalidad', fontsize=14)
                    ax_conv.plot(np.array(convergence), marker='', color='blue', linewidth=2)
                    ax_conv.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig_conv)
                else:
                    st.info("No hay datos de convergencia disponibles.")
          
            for i, cohort in enumerate(cohort_list):
                with tabs[i + 1]:
                    semester_names = [f"Cuatrimestre {j+1}" for j in range(num_semesters)]
                    semester_tab = st.tabs(semester_names)
                    for sem_idx, sem_name in enumerate(semester_names):
                        with semester_tab[sem_idx]:
                            subjects_in_semester = cohort_semester_subjects[cohort][sem_idx]
                            logger.info(f"subjects_in_semester para {cohort} - {sem_name}: {subjects_in_semester}")
                            sem_events = []
                            for subject, prof, day, shift in subjects_in_semester:
                                # Ejemplo: cada materia inicia el primer lunes del cuatrimestre
                                start_date = datetime(start_year, 3 + 6 * (sem_idx % semesters_per_year), 1)
                                sem_events.append({
                                    "title": f"{subject} ({cohort}) - {prof} [{day}, {shift}]",
                                    "start": start_date.isoformat(),
                                    "end": (start_date.replace(hour=2)).isoformat(),
                                })
                            logger.info(f"sem_events para {cohort} - {sem_name}: {sem_events}")
                            st.subheader(f"Calendario de {cohort} - {sem_name}")
                            
                            
                            # Crear una grilla vacía
                            grid = pd.DataFrame('', index=shifts_list, columns=week_days_list)

                            # Llenar la grilla con las materias y profesores
                            for subject, prof, day, shift in subjects_in_semester:
                                if day in grid.columns and shift in grid.index:
                                    grid.at[shift, day] = f"{subject}\n{prof}"

                            st.dataframe(grid, height=200)