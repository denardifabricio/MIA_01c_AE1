#!/usr/bin/env python3
"""
Script simple para probar y ajustar el algoritmo del CourseGenerator.
Ejecuta el generador con la configuración de ejemplo y muestra los resultados.
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from insurance_course_calendar import CourseGenerator

# Configurar logging simple
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Cargar configuración desde el archivo JSON."""
    with open("config_example.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_professor_assignments(config):
    """Preparar asignaciones de profesores."""
    professor_assignments = {}
    for subject, professor in config["professor_by_subject"]:
        if subject not in professor_assignments:
            professor_assignments[subject] = professor
    return professor_assignments

def get_mandatory_penalty_weights(config):
    """Retornar pesos que hacen obligatorias las restricciones principales."""
    # Obtener los pesos de penalización desde config si existen, sino usar valores por defecto
    weights = config.get("penalty_weights", {})
    return {
        'semester_distribution': weights.get('semester_distribution', 1e12),
        'prerequisites': weights.get('prerequisites', 1e11),
        'blocked_slots': weights.get('blocked_slots', 1000),
        'professor_overload': weights.get('professor_overload', 100),
        'cohort_overload': weights.get('cohort_overload', 200),
        'slot_conflicts': weights.get('slot_conflicts', 500),
        'day_balance': weights.get('day_balance', 50),
        'shift_usage': weights.get('shift_usage', 100)
    }

def print_solution_summary(solution, cost, convergence):
    """Imprimir un resumen de la solución."""
    print("\n" + "="*60)
    print("RESUMEN DE LA SOLUCIÓN")
    print("="*60)
    print(f"Costo final: {cost:.2f}")
    print(f"Iteraciones: {len(convergence)}")
    
    if len(convergence) > 1:
        print(f"Mejora: {convergence[0] - convergence[-1]:.2f}")
        print(f"Convergencia: {convergence[0]:.2f} → {convergence[-1]:.2f}")
    
    # Agrupar por cuatrimestre
    by_semester = {}
    for subject, semester, professor, cohort, day, shift in solution:
        if semester not in by_semester:
            by_semester[semester] = []
        by_semester[semester].append((subject, professor, cohort, day, shift))
    
    print(f"\nDistribución por cuatrimestre:")
    for semester in sorted(by_semester.keys()):
        print(f"\nCuatrimestre {semester + 1} ({len(by_semester[semester])} asignaciones):")
        for subject, professor, cohort, day, shift in sorted(by_semester[semester]):
            print(f"  • {subject:20} | {professor:10} | {cohort:10} | {day:10} | {shift}")

def plot_convergence(convergence, save_plot=True, show_plot=True):
    """Crear y mostrar el gráfico de convergencia."""
    if not convergence:
        print("⚠️  No hay datos de convergencia para graficar")
        return
    
    print(f"\n📈 Generando gráfico de convergencia ({len(convergence)} iteraciones)...")
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal de convergencia
    plt.subplot(2, 2, 1)
    plt.plot(convergence, 'b-', linewidth=2, alpha=0.8)
    plt.title('Convergencia del PSO', fontsize=14, fontweight='bold')
    plt.xlabel('Iteración')
    plt.ylabel('Costo')
    plt.grid(True, alpha=0.3)
    
    # Gráfico en escala logarítmica
    plt.subplot(2, 2, 2)
    plt.semilogy(convergence, 'r-', linewidth=2, alpha=0.8)
    plt.title('Convergencia (Escala Log)', fontsize=14, fontweight='bold')
    plt.xlabel('Iteración')
    plt.ylabel('Costo (log)')
    plt.grid(True, alpha=0.3)
    
    # Mejora relativa por iteración
    plt.subplot(2, 2, 3)
    if len(convergence) > 1:
        improvements = [-((convergence[i] - convergence[i-1]) / convergence[i-1] * 100) 
                       for i in range(1, len(convergence))]
        plt.plot(range(1, len(convergence)), improvements, 'g-', linewidth=2, alpha=0.8)
        plt.title('Mejora Relativa por Iteración', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración')
        plt.ylabel('Mejora (%)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Estadísticas de convergencia
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calcular estadísticas
    initial_cost = convergence[0]
    final_cost = convergence[-1]
    total_improvement = initial_cost - final_cost
    improvement_percent = (total_improvement / initial_cost * 100) if initial_cost > 0 else 0
    
    # Encontrar punto de convergencia (cambio < 1%)
    convergence_point = len(convergence)
    for i in range(1, len(convergence)):
        if abs(convergence[i] - convergence[i-1]) / convergence[i-1] < 0.01:
            convergence_point = i
            break
    
    stats_text = f"""
📊 ESTADÍSTICAS DE CONVERGENCIA

• Costo inicial: {initial_cost:.2f}
• Costo final: {final_cost:.2f}
• Mejora total: {total_improvement:.2f}
• Mejora porcentual: {improvement_percent:.1f}%
• Iteraciones totales: {len(convergence)}
• Convergencia en iter: {convergence_point}
• Mejor mejora: {max(improvements) if len(convergence) > 1 else 0:.2f}%
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = 'convergencia_pso.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado como: {plot_filename}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'total_improvement': total_improvement,
        'improvement_percent': improvement_percent,
        'convergence_point': convergence_point
    }

def print_detailed_analysis(solution, config):
    """Imprimir análisis detallado de la solución."""
    print("\n" + "="*60)
    print("ANÁLISIS DETALLADO")
    print("="*60)
    
    # Contar uso de profesores
    prof_count = {}
    for _, _, professor, _, _, _ in solution:
        prof_count[professor] = prof_count.get(professor, 0) + 1
    
    print("Uso de profesores:")
    for prof, count in sorted(prof_count.items()):
        print(f"  • {prof}: {count} asignaciones")
    
    # Contar uso de días
    day_count = {}
    for _, _, _, _, day, _ in solution:
        day_count[day] = day_count.get(day, 0) + 1
    
    print("\nUso de días:")
    for day, count in sorted(day_count.items()):
        print(f"  • {day}: {count} asignaciones")
    
    # Verificar prerequisitos
    print("\nVerificación de prerequisitos:")
    semester_by_subject = {}
    for subject, semester, _, cohort, _, _ in solution:
        key = (subject, cohort)
        if key not in semester_by_subject:
            semester_by_subject[key] = semester
    
    violations = 0
    for prereq_pair in config["prerequisites"]:
        subject, prereq = prereq_pair
        for cohort in config["cohorts"]:
            subj_sem = semester_by_subject.get((subject, cohort), -1)
            prereq_sem = semester_by_subject.get((prereq, cohort), -1)
            if subj_sem <= prereq_sem and subj_sem != -1 and prereq_sem != -1:
                print(f"  ❌ {subject} (sem {subj_sem+1}) antes que {prereq} (sem {prereq_sem+1}) para {cohort}")
                violations += 1
    
    if violations == 0:
        print("  ✅ Todos los prerequisitos se cumplen correctamente")
    else:
        print(f"  ❌ {violations} violaciones de prerequisitos encontradas")

def print_quality_analysis(generator, save_to_file=True):
    """Imprimir análisis de calidad de la solución y guardarlo en archivo."""
    print("\n" + "="*60)
    print("ANÁLISIS DE CALIDAD")
    print("="*60)
    
    analysis = generator.analyze_solution_quality()
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    # Crear el contenido del análisis
    analysis_content = []
    analysis_content.append("="*60)
    analysis_content.append("ANÁLISIS DE CALIDAD DE LA SOLUCIÓN")
    analysis_content.append("="*60)
    analysis_content.append("")
    
    # Score principal
    score_line = f"📊 Score de Calidad: {analysis['quality_score']}/100"
    print(score_line)
    analysis_content.append(score_line)
    analysis_content.append("")
    
    # Resumen de problemas
    problems_header = "📈 Resumen de Problemas:"
    print(problems_header)
    analysis_content.append(problems_header)
    
    problems = [
        f"  • Violaciones de prerequisitos: {analysis['prerequisite_violations']}",
        f"  • Bloques bloqueados usados: {analysis['blocked_slot_violations']}",
        f"  • Sobrecarga de profesores: {analysis['professor_overloads']}",
        f"  • Sobrecarga de cohortes: {analysis['cohort_overloads']}",
        f"  • Conflictos de horarios: {analysis['slot_conflicts']}"
    ]
    
    for problem in problems:
        print(problem)
        analysis_content.append(problem)
    
    analysis_content.append("")
    
    # Distribución por cuatrimestre
    distribution_header = "📅 Distribución por Cuatrimestre:"
    print(distribution_header)
    analysis_content.append(distribution_header)
    
    for cohort, distribution in analysis['semester_distribution'].items():
        sem_counts = [distribution.get(i, 0) for i in range(4)]
        dist_line = f"  • {cohort}: {sem_counts} materias"
        print(dist_line)
        analysis_content.append(dist_line)
    
    analysis_content.append("")
    
    # Uso de días y turnos
    day_usage_header = "📊 Uso de Días:"
    print(day_usage_header)
    analysis_content.append(day_usage_header)
    
    for day, count in analysis['day_usage'].items():
        day_line = f"  • {day}: {count} asignaciones"
        print(day_line)
        analysis_content.append(day_line)
    
    analysis_content.append("")
    
    shift_usage_header = "🕐 Uso de Turnos:"
    print(shift_usage_header)
    analysis_content.append(shift_usage_header)
    
    for shift, count in analysis['shift_usage'].items():
        shift_line = f"  • {shift}: {count} asignaciones"
        print(shift_line)
        analysis_content.append(shift_line)
    
    # Detalles de violaciones de prerequisitos
    if analysis['prerequisite_violations'] > 0:
        violations_header = "\n⚠️  Detalles de Violaciones de Prerequisitos:"
        print(violations_header)
        analysis_content.append("")
        analysis_content.append(violations_header)
        
        for detail in analysis['prerequisite_details']:
            violation_line = f"  • {detail['cohort']}: {detail['subject']} (C{detail['subject_semester']}) requiere {detail['prerequisite']} (C{detail['prerequisite_semester']})"
            print(violation_line)
            analysis_content.append(violation_line)
    else:
        success_line = "\n✅ No hay violaciones de prerequisitos"
        print(success_line)
        analysis_content.append("")
        analysis_content.append(success_line)
    
    # Guardar en archivo si se solicita
    if save_to_file:
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report/analisis_calidad_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(analysis_content))
                f.write("\n\n")
                f.write("="*60)
                f.write(f"\nArchivo generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                f.write("\n" + "="*60)
            
            print(f"\n💾 Análisis de calidad guardado en: {filename}")
            
        except Exception as e:
            print(f"\n⚠️  Error al guardar archivo: {e}")
    
    return analysis

def generate_complete_report(config, generator, cost, convergence, solution, analysis):
    """Generar un reporte completo de la ejecución."""
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report/reporte_completo_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Encabezado del reporte
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPLETO DE GENERACIÓN DE CALENDARIOS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Configuración utilizada
            f.write("CONFIGURACIÓN UTILIZADA:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Materias: {len(config['subjects'])}\n")
            f.write(f"• Profesores: {len(config['professors'])}\n")
            f.write(f"• Cohortes: {len(config['cohorts'])}\n")
            f.write(f"• Años: {config['num_years']}\n")
            f.write(f"• Cuatrimestres por año: {config['semesters_per_year']}\n")
            f.write(f"• Total cuatrimestres: {config['num_years'] * config['semesters_per_year']}\n")
            f.write(f"• Días de la semana: {len(config['week_days'])}\n")
            f.write(f"• Turnos: {len(config['shifts'])}\n")
            f.write(f"• Bloques bloqueados: {len(config.get('blocked_slots', []))}\n")
            
            # Parámetros del PSO
            f.write(f"\nPARÁMETROS DEL PSO:\n")
            f.write("-" * 40 + "\n")
            pso_options = config.get('pso_options', {})
            f.write(f"• c1 (cognitive): {pso_options.get('c1', 'default')}\n")
            f.write(f"• c2 (social): {pso_options.get('c2', 'default')}\n")
            f.write(f"• w (inertia): {pso_options.get('w', 'default')}\n")
            f.write(f"• Partículas: {config.get('n_particles', 60)}\n")
            f.write(f"• Iteraciones máximas: {config.get('max_iters', 200)}\n")
            
            # Pesos de penalización utilizados
            f.write(f"\nPESOS DE PENALIZACIÓN UTILIZADOS:\n")
            f.write("-" * 40 + "\n")
            penalty_weights = generator.penalty_weights if hasattr(generator, 'penalty_weights') else get_mandatory_penalty_weights(config)
            f.write(f"• Distribución de cuatrimestres: {penalty_weights.get('semester_distribution', 'N/A')}\n")
            f.write(f"• Prerequisitos: {penalty_weights.get('prerequisites', 'N/A')}\n")
            f.write(f"• Bloques bloqueados: {penalty_weights.get('blocked_slots', 'N/A')}\n")
            f.write(f"• Sobrecarga de profesores: {penalty_weights.get('professor_overload', 'N/A')}\n")
            f.write(f"• Sobrecarga de cohortes: {penalty_weights.get('cohort_overload', 'N/A')}\n")
            f.write(f"• Conflictos de horarios: {penalty_weights.get('slot_conflicts', 'N/A')}\n")
            f.write(f"• Balance de días: {penalty_weights.get('day_balance', 'N/A')}\n")
            f.write(f"• Uso de turnos: {penalty_weights.get('shift_usage', 'N/A')}\n")
            
            # Resultados de la optimización
            f.write(f"\nRESULTADOS DE LA OPTIMIZACIÓN:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Costo final: {cost:.2f}\n")
            f.write(f"• Iteraciones ejecutadas: {len(convergence)}\n")
            if len(convergence) > 1:
                f.write(f"• Costo inicial: {convergence[0]:.2f}\n")
                f.write(f"• Mejora total: {convergence[0] - cost:.2f}\n")
                f.write(f"• Mejora porcentual: {(convergence[0] - cost) / convergence[0] * 100:.1f}%\n")
            
            # Análisis de calidad
            f.write(f"\nANÁLISIS DE CALIDAD:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Score de calidad: {analysis['quality_score']}/100\n")
            f.write(f"• Violaciones de prerequisitos: {analysis['prerequisite_violations']}\n")
            f.write(f"• Bloques bloqueados usados: {analysis['blocked_slot_violations']}\n")
            f.write(f"• Sobrecarga de profesores: {analysis['professor_overloads']}\n")
            f.write(f"• Sobrecarga de cohortes: {analysis['cohort_overloads']}\n")
            f.write(f"• Conflictos de horarios: {analysis['slot_conflicts']}\n")
            
            # Distribución por cuatrimestre
            f.write(f"\nDISTRIBUCIÓN POR CUATRIMESTRE:\n")
            f.write("-" * 40 + "\n")
            for cohort, distribution in analysis['semester_distribution'].items():
                f.write(f"• {cohort}:\n")
                for sem in range(config['num_years'] * config['semesters_per_year']):
                    count = distribution.get(sem, 0)
                    f.write(f"  - Cuatrimestre {sem + 1}: {count} materias\n")
            
            # Listado completo de asignaciones
            f.write(f"\nLISTADO COMPLETO DE ASIGNACIONES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Materia':<25} {'Cuatrimestre':<12} {'Profesor':<15} {'Cohorte':<12} {'Día':<12} {'Turno':<10}\n")
            f.write("-" * 100 + "\n")
            
            # Ordenar por cuatrimestre, luego por cohorte
            sorted_solution = sorted(solution, key=lambda x: (x[1], x[3], x[0]))
            for subject, semester, professor, cohort, day, shift in sorted_solution:
                f.write(f"{subject:<25} {semester + 1:<12} {professor:<15} {cohort:<12} {day:<12} {shift:<10}\n")
            
            # Detalles de violaciones si existen
            if analysis['prerequisite_violations'] > 0:
                f.write(f"\nDETALLES DE VIOLACIONES DE PREREQUISITOS:\n")
                f.write("-" * 40 + "\n")
                for detail in analysis['prerequisite_details']:
                    f.write(f"• {detail['cohort']}: {detail['subject']} (C{detail['subject_semester']}) "
                           f"requiere {detail['prerequisite']} (C{detail['prerequisite_semester']})\n")
            
            # Estadísticas adicionales
            f.write(f"\nESTADÍSTICAS ADICIONALES:\n")
            f.write("-" * 40 + "\n")
            
            # Uso de profesores
            prof_count = {}
            for _, _, professor, _, _, _ in solution:
                prof_count[professor] = prof_count.get(professor, 0) + 1
            
            f.write("Asignaciones por profesor:\n")
            for prof, count in sorted(prof_count.items()):
                f.write(f"  • {prof}: {count} asignaciones\n")
            
            # Uso de días
            f.write("\nAsignaciones por día:\n")
            for day, count in analysis['day_usage'].items():
                f.write(f"  • {day}: {count} asignaciones\n")
            
            # Uso de turnos
            f.write("\nAsignaciones por turno:\n")
            for shift, count in analysis['shift_usage'].items():
                f.write(f"  • {shift}: {count} asignaciones\n")
            
            # Pie de página
            f.write("\n" + "=" * 80 + "\n")
            f.write("Fin del reporte\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 Reporte completo guardado en: {filename}")
        return filename
        
    except Exception as e:
        print(f"⚠️  Error al generar reporte completo: {e}")
        return None

def main():
    """Función principal."""
    print("🚀 Ejecutando prueba del CourseGenerator")
    print("="*60)
    
    try:
        # Cargar configuración
        print("📋 Cargando configuración...")
        config = load_config()
        
        print(f"   • Materias: {len(config['subjects'])}")
        print(f"   • Profesores: {len(config['professors'])}")
        print(f"   • Cohortes: {len(config['cohorts'])}")
        print(f"   • Cuatrimestres: {config['num_years'] * config['semesters_per_year']}")
        
        # Preparar datos
        professor_assignments = prepare_professor_assignments(config)
        blocked_slots = [tuple(slot) for slot in config["blocked_slots"]]
        
        # Crear generador con pesos obligatorios
        print("\n🔧 Inicializando generador...")
        print(f"   • Parámetros PSO: {config.get('pso_options', 'default')}")
        print(f"   • Partículas: {config.get('n_particles', 60)}")
        print(f"   • Iteraciones máximas: {config.get('max_iters', 200)}")
        print("   • ⚠️  RESTRICCIONES OBLIGATORIAS: Distribución de semestres y Prerequisitos")
        
        # Usar pesos obligatorios que hacen estas restricciones críticas
        mandatory_weights = get_mandatory_penalty_weights(config)
        
        generator = CourseGenerator(
            subjects=config["subjects"],
            prerequisites=config["prerequisites"],
            professors=config["professors"],
            cohorts=config["cohorts"],
            professor_by_subject=professor_assignments,
            num_years=config["num_years"],
            semesters_per_year=config["semesters_per_year"],
            start_year=config["start_year"],
            max_classes_per_week=config["max_classes_per_week"],
            max_subjects_per_day_professor=config["max_subjects_per_day_professor"],
            max_subjects_per_day_cohort=config["max_subjects_per_day_cohort"],
            week_days=config["week_days"],
            shifts=config["shifts"],
            blocked_slots=blocked_slots,
            pso_options=config.get("pso_options"),
            n_particles=config.get("n_particles", 60),
            max_iters=config.get("max_iters", 500),
            penalty_weights=mandatory_weights, 
            logger=logger
        )
        
        # Ejecutar optimización
        print("\n⚡ Ejecutando optimización PSO...")
        print("   (Esto puede tomar unos minutos...)")
        
        cost, position, convergence = generator.run_pso_optimizer()
        solution = generator.get_solution()
        
        # Mostrar resultados
        print_solution_summary(solution, cost, convergence)
        
        # Generar y mostrar gráfico de convergencia
        convergence_stats = plot_convergence(convergence, save_plot=True, show_plot=False)
        
        print_detailed_analysis(solution, config)
        analysis = print_quality_analysis(generator, save_to_file=True)
        
        # Generar reporte completo
        generate_complete_report(config, generator, cost, convergence, solution, analysis)
        
        print(f"\n✅ Optimización completada exitosamente!")
        print(f"💾 Los resultados se han guardado en los archivos de salida")
        print(f"📈 Gráfico de convergencia guardado como 'convergencia_pso.png'")
        print(f"📊 Análisis de calidad y reporte completo guardados en archivos .txt")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
