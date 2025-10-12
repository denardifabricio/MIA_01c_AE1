<img src="https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg" width="500" align="center">

# Algoritmos Evolutivos I - Trabajo Práctico Final

**Autores:**  
- Esp. Ing. Fabricio Denardi  
- Esp. Ing. Bruno Masoller   
- Esp. Lic. Noelia Qualindi  

**Docente:** 
Esp. Ing. Miguel Augusto Azar  
**Cohorte:**
MIA 01-2025

## Descripción del Proyecto

Este proyecto implementa un **algoritmo evolutivo** para resolver el problema de programación de calendarios académicos para un curso introductorio de seguros en una empresa aseguradora argentina. El sistema utiliza **Particle Swarm Optimization (PSO)** para optimizar la asignación de materias, cohortes y cronogramas, considerando múltiples restricciones y objetivos.

## Objetivo

Desarrollar una aplicación web que permita generar y visualizar calendarios académicos optimizados, utilizando algoritmos evolutivos para encontrar la mejor distribución de materias a lo largo del tiempo, respetando restricciones de recursos, capacidades y disponibilidad.

## Características Principales

- **Optimización con PSO**: Implementación de Particle Swarm Optimization para la optimización multiobjetivo
- **Interface Web**: Aplicación web interactiva desarrollada con Streamlit
- **API REST**: Backend con FastAPI para el procesamiento de optimizaciones
- **Visualización**: Generación automática de calendarios en formato visual y Excel
- **Dockerización**: Proyecto completamente containerizado para fácil despliegue

## Arquitectura del Sistema

El proyecto está dividido en dos componentes principales:

1. **Course Generator** (Backend): API REST que ejecuta el algoritmo de optimización
2. **Course Viewer** (Frontend): Interface web para configuración y visualización

## Estructura de Carpetas

```
MIA_01c_AE1/
├── README.MD                           # Este archivo
├── docker-compose.yml                  # Configuración de Docker Compose
├── report.log                          # Logs de ejecución
│
├── src/                                # Código fuente principal
│   ├── course-generator/               # Backend - API de optimización
│   │   ├── Dockerfile                  # Configuración Docker del backend
│   │   ├── main.py                     # API FastAPI principal
│   │   ├── insurance_course_calendar.py # Lógica del algoritmo PSO
│   │   ├── experiment_pso.py           # Experimentos y análisis
│   │   ├── optimize_penalties.py       # Optimización de penalidades
│   │   ├── test.py                     # Scripts de testing
│   │   ├── requirements.txt            # Dependencias Python
│   │   ├── config_example.json         # Configuración de ejemplo
│   │   ├── model/                      # Modelos de datos
│   │   └── report/                     # Reportes generados
│   │
│   └── course-viewer/                  # Frontend - Interface web
│       ├── Dockerfile                  # Configuración Docker del frontend
│       ├── main.py                     # Aplicación Streamlit principal
│       ├── requirements.txt            # Dependencias Python
│       ├── config_example.json         # Configuración de ejemplo
│       ├── model/                      # Modelos compartidos
│       ├── .streamlit/                 # Configuración Streamlit
│       └── images/                     # Recursos gráficos
│
├── lecture/                            # Material académico
│   └── ae1_ga_optim_aulas_02.ipynb    # Jupyter notebook con análisis
│
├── report/                             # Documentación y reportes
│   ├── MIA_c01_AE1.pdf                # Informe técnico principal
│   ├── Conclusiones.tex               # Conclusiones en LaTeX
│   └── ...                            # Otros archivos del reporte
│
└── logs/                              # Directorio de logs del sistema
```

## Instalación y Ejecución con Docker

### Prerrequisitos

- Docker
- Docker Compose

### Configuración

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd MIA_01c_AE1
```

2. **Configurar variables de entorno:**
Crear un archivo `.env` en la raíz del proyecto con:
```env
COURSE_GENERATOR_PORT=8000
COURSE_VIEWER_PORT=8501
```

### Ejecución

**Opción 1: Ejecutar con Docker Compose (Recomendado)**
```bash
docker compose up --build
```

**Opción 2: Ejecutar en segundo plano**
```bash
docker compose up -d --build
```

### Acceso a la Aplicación

Una vez ejecutado, la aplicación estará disponible en:

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API (FastAPI)**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs

### Detener la Aplicación

```bash
docker compose down
```

## Uso del Sistema

1. **Acceder a la interface web** en http://localhost:8501
2. **Configurar parámetros** del curso (número de cohortes, materias, restricciones)
3. **Ejecutar optimización** con el algoritmo PSO
4. **Visualizar resultados** en forma de calendarios y gráficos
5. **Descargar** calendarios en formato Excel o PNG

## Algoritmo de Optimización

El sistema utiliza **Particle Swarm Optimization (PSO)** con las siguientes características:

- **Función objetivo multicriterio** que minimiza conflictos de horarios
- **Restricciones** de capacidad, disponibilidad de aulas y recursos
- **Penalizaciones** por solapamientos y violaciones de restricciones
- **Convergencia adaptativa** con criterios de parada inteligentes

## Documentación Técnica

Para más detalles sobre la implementación, metodología y resultados, consultar el [informe técnico completo](report/MIA_c01_AE1.pdf).

## Desarrollo Local (Sin Docker)

Si prefiere ejecutar sin Docker:

1. **Backend:**
```bash
cd src/course-generator
pip install -r requirements.txt
python main.py
```

2. **Frontend:**
```bash
cd src/course-viewer
pip install -r requirements.txt
streamlit run main.py
```