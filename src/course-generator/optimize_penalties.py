#!/usr/bin/env python3
"""
Script para optimizar pesos de penalización del algoritmo.
Permite encontrar la mejor combinación de pesos para mejorar la calidad de las soluciones.
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from insurance_course_calendar import CourseGenerator

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Cargar configuración base."""
    with open("config_example.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_professor_assignments(config):
    """Preparar asignaciones de profesores."""
    professor_assignments = {}
    for subject, professor in config["professor_by_subject"]:
        if subject not in professor_assignments:
            professor_assignments[subject] = professor
    return professor_assignments

def test_penalty_weights(config, penalty_weights, test_name, max_iters=50):
    """Probar una configuración específica de pesos de penalización."""
    print(f"\n🧪 Probando: {test_name}")
    
    # Preparar datos
    professor_assignments = prepare_professor_assignments(config)
    blocked_slots = [tuple(slot) for slot in config["blocked_slots"]]
    
    # Crear generador con pesos específicos
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
        print_image_result=False,
        print_excel_result=False,
        pso_options=config.get("pso_options"),
        n_particles=40,  # Menos partículas para pruebas rápidas
        max_iters=max_iters,
        penalty_weights=penalty_weights,
        logger=logger
    )
    
    try:
        # Ejecutar optimización
        cost, position, convergence = generator.run_pso_optimizer()
        solution = generator.get_solution()
        
        # Analizar calidad
        quality_analysis = generator.analyze_solution_quality(solution)
        
        return {
            'test_name': test_name,
            'penalty_weights': penalty_weights,
            'final_cost': cost,
            'quality_score': quality_analysis.get('quality_score', 0),
            'prerequisite_violations': quality_analysis.get('prerequisite_violations', 0),
            'blocked_slot_violations': quality_analysis.get('blocked_slot_violations', 0),
            'professor_overloads': quality_analysis.get('professor_overloads', 0),
            'cohort_overloads': quality_analysis.get('cohort_overloads', 0),
            'slot_conflicts': quality_analysis.get('slot_conflicts', 0),
            'convergence_history': convergence[-10:] if len(convergence) > 10 else convergence  # Últimas 10 iteraciones
        }
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def run_penalty_optimization():
    """Ejecutar optimización de pesos de penalización."""
    print("🎯 Optimizador de Pesos de Penalización")
    print("="*60)
    
    config = load_config()
    
    # Definir diferentes configuraciones de pesos para probar
    penalty_configurations = [
        {
            'name': 'Default',
            'weights': {
                'semester_distribution': 1000000,
                'prerequisites': 5000,
                'blocked_slots': 10000,
                'professor_overload': 100,
                'cohort_overload': 200,
                'slot_conflicts': 500,
                'day_balance': 50,
                'shift_usage': 100
            }
        },
        {
            'name': 'High Prerequisites',
            'weights': {
                'semester_distribution': 1000000,
                'prerequisites': 15000,  # Aumentar peso de prerequisitos
                'blocked_slots': 10000,
                'professor_overload': 100,
                'cohort_overload': 200,
                'slot_conflicts': 500,
                'day_balance': 50,
                'shift_usage': 100
            }
        },
        {
            'name': 'Balanced Distribution',
            'weights': {
                'semester_distribution': 500000,  # Reducir rigidez de distribución
                'prerequisites': 8000,
                'blocked_slots': 15000,
                'professor_overload': 200,
                'cohort_overload': 300,
                'slot_conflicts': 1000,
                'day_balance': 100,
                'shift_usage': 150
            }
        },
        {
            'name': 'Conflict Focus',
            'weights': {
                'semester_distribution': 1000000,
                'prerequisites': 10000,
                'blocked_slots': 20000,
                'professor_overload': 500,  # Aumentar penalización de sobrecarga
                'cohort_overload': 800,
                'slot_conflicts': 2000,
                'day_balance': 200,
                'shift_usage': 300
            }
        },
        {
            'name': 'Flexible Semester',
            'weights': {
                'semester_distribution': 200000,  # Muy flexible en distribución
                'prerequisites': 12000,
                'blocked_slots': 15000,
                'professor_overload': 300,
                'cohort_overload': 400,
                'slot_conflicts': 800,
                'day_balance': 150,
                'shift_usage': 200
            }
        }
    ]
    
    results = []
    
    # Probar cada configuración
    for config_test in penalty_configurations:
        result = test_penalty_weights(
            config=config,
            penalty_weights=config_test['weights'],
            test_name=config_test['name']
        )
        if result:
            results.append(result)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("📊 RESULTADOS DE OPTIMIZACIÓN DE PESOS")
    print("="*80)
    
    if results:
        # Ordenar por quality_score descendente
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"{'Configuración':<18} {'Calidad':<8} {'Costo':<12} {'PreReq':<7} {'Bloq':<5} {'ProfOL':<7} {'CohOL':<6} {'Confl':<6}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['test_name']:<18} "
                  f"{result['quality_score']:<8.1f} "
                  f"{result['final_cost']:<12.1f} "
                  f"{result['prerequisite_violations']:<7} "
                  f"{result['blocked_slot_violations']:<5} "
                  f"{result['professor_overloads']:<7} "
                  f"{result['cohort_overloads']:<6} "
                  f"{result['slot_conflicts']:<6}")
        
        # Mejor configuración
        best = results[0]
        print(f"\n🏆 Mejor configuración: {best['test_name']}")
        print(f"   • Calidad: {best['quality_score']:.1f}/100")
        print(f"   • Costo final: {best['final_cost']:.1f}")
        print(f"   • Violaciones de prerequisitos: {best['prerequisite_violations']}")
        
        print(f"\n🔧 Pesos recomendados:")
        for key, value in best['penalty_weights'].items():
            print(f"   • {key}: {value}")
        
        # Guardar resultados
        with open('penalty_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en 'penalty_optimization_results.json'")
        
        # Generar reporte de texto
        generate_penalty_report(results)
        
        # Crear gráfico comparativo
        create_comparison_plot(results)
        
        # Generar archivo de configuración optimizada
        generate_optimized_config(config, best['penalty_weights'])
    
    else:
        print("❌ No se pudieron completar las pruebas")

def create_comparison_plot(results):
    """Crear gráfico comparativo de resultados."""
    if not results:
        return
    
    print("\n📊 Generando gráfico comparativo...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [r['test_name'] for r in results]
    quality_scores = [r['quality_score'] for r in results]
    final_costs = [r['final_cost'] for r in results]
    prereq_violations = [r['prerequisite_violations'] for r in results]
    total_violations = [r['prerequisite_violations'] + r['blocked_slot_violations'] + 
                       r['professor_overloads'] + r['cohort_overloads'] + r['slot_conflicts'] 
                       for r in results]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    # 1. Quality Score
    ax1.bar(names, quality_scores, color=colors, alpha=0.8)
    ax1.set_title('Score de Calidad', fontweight='bold')
    ax1.set_ylabel('Calidad (0-100)')
    ax1.tick_params(axis='x', rotation=45)
    for i, score in enumerate(quality_scores):
        ax1.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Final Cost
    ax2.bar(names, final_costs, color=colors, alpha=0.8)
    ax2.set_title('Costo Final', fontweight='bold')
    ax2.set_ylabel('Costo')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Prerequisite Violations
    ax3.bar(names, prereq_violations, color=colors, alpha=0.8)
    ax3.set_title('Violaciones de Prerequisitos', fontweight='bold')
    ax3.set_ylabel('Número de Violaciones')
    ax3.tick_params(axis='x', rotation=45)
    for i, viol in enumerate(prereq_violations):
        if viol > 0:
            ax3.text(i, viol + 0.1, str(viol), ha='center', va='bottom', fontweight='bold')
    
    # 4. Total Violations
    ax4.bar(names, total_violations, color=colors, alpha=0.8)
    ax4.set_title('Total de Problemas', fontweight='bold')
    ax4.set_ylabel('Número Total de Problemas')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('penalty_optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado como: penalty_optimization_comparison.png")
    plt.close()

def generate_penalty_report(results):
    """Generar reporte de texto con los resultados de optimización de penalizaciones."""
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimizacion_penalizaciones_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Encabezado
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE OPTIMIZACIÓN DE PESOS DE PENALIZACIÓN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Resumen ejecutivo
            f.write("RESUMEN EJECUTIVO:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Configuraciones probadas: {len(results)}\n")
            if results:
                best = results[0]
                f.write(f"• Mejor configuración: {best['test_name']}\n")
                f.write(f"• Score de calidad máximo: {best['quality_score']:.1f}/100\n")
                f.write(f"• Costo final mínimo: {best['final_cost']:.1f}\n")
            f.write("\n")
            
            # Resultados detallados
            f.write("RESULTADOS DETALLADOS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Configuración':<18} {'Calidad':<8} {'Costo':<12} {'PreReq':<7} {'Bloq':<5} {'ProfOL':<7} {'CohOL':<6} {'Confl':<6}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                f.write(f"{result['test_name']:<18} "
                       f"{result['quality_score']:<8.1f} "
                       f"{result['final_cost']:<12.1f} "
                       f"{result['prerequisite_violations']:<7} "
                       f"{result['blocked_slot_violations']:<5} "
                       f"{result['professor_overloads']:<7} "
                       f"{result['cohort_overloads']:<6} "
                       f"{result['slot_conflicts']:<6}\n")
            
            # Análisis de la mejor configuración
            if results:
                best = results[0]
                f.write(f"\nANÁLISIS DE LA MEJOR CONFIGURACIÓN ({best['test_name']}):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score de calidad: {best['quality_score']:.1f}/100\n")
                f.write(f"Costo final: {best['final_cost']:.1f}\n")
                f.write(f"Violaciones de prerequisitos: {best['prerequisite_violations']}\n")
                f.write(f"Bloques bloqueados usados: {best['blocked_slot_violations']}\n")
                f.write(f"Sobrecarga de profesores: {best['professor_overloads']}\n")
                f.write(f"Sobrecarga de cohortes: {best['cohort_overloads']}\n")
                f.write(f"Conflictos de horarios: {best['slot_conflicts']}\n")
                
                f.write(f"\nPESOS RECOMENDADOS:\n")
                f.write("-" * 40 + "\n")
                for key, value in best['penalty_weights'].items():
                    f.write(f"• {key}: {value}\n")
            
            # Comparación entre configuraciones
            f.write(f"\nCOMPARACIÓN ENTRE CONFIGURACIONES:\n")
            f.write("-" * 40 + "\n")
            
            if len(results) >= 2:
                worst = results[-1]
                f.write(f"Mejor vs Peor configuración:\n")
                f.write(f"• Diferencia en calidad: {best['quality_score'] - worst['quality_score']:.1f} puntos\n")
                f.write(f"• Diferencia en costo: {worst['final_cost'] - best['final_cost']:.1f}\n")
                f.write(f"• Diferencia en violaciones prerequisitos: {worst['prerequisite_violations'] - best['prerequisite_violations']}\n")
            
            # Recomendaciones
            f.write(f"\nRECOMENDACIONES:\n")
            f.write("-" * 40 + "\n")
            
            if results:
                if best['quality_score'] >= 90:
                    f.write("✅ Excelente: La configuración optimal produce soluciones de alta calidad.\n")
                elif best['quality_score'] >= 70:
                    f.write("✓ Bueno: La configuración optimal produce soluciones aceptables.\n")
                else:
                    f.write("⚠️ Mejorable: Considere ajustar más los pesos o revisar las restricciones.\n")
                
                if best['prerequisite_violations'] == 0:
                    f.write("✅ Prerequisitos: Se respetan correctamente en la mejor configuración.\n")
                else:
                    f.write("⚠️ Prerequisitos: Aún hay violaciones. Considere aumentar el peso de 'prerequisites'.\n")
            
            # Pie de página
            f.write("\n" + "=" * 80 + "\n")
            f.write("Fin del reporte de optimización\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 Reporte de optimización guardado en: {filename}")
        return filename
        
    except Exception as e:
        print(f"⚠️  Error al generar reporte de optimización: {e}")
        return None

def generate_optimized_config(base_config, optimized_weights):
    """Generar archivo de configuración con pesos optimizados."""
    optimized_config = base_config.copy()
    optimized_config['penalty_weights'] = optimized_weights
    
    with open('config_optimized.json', 'w', encoding='utf-8') as f:
        json.dump(optimized_config, f, indent=2, ensure_ascii=False)
    
    print("📄 Configuración optimizada guardada como: config_optimized.json")

def main():
    """Función principal."""
    run_penalty_optimization()

if __name__ == "__main__":
    main()