#!/usr/bin/env python3
"""
Script para experimentar con diferentes parámetros del PSO.
Permite probar rápidamente diferentes configuraciones del algoritmo.
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

def run_experiment(config, pso_options, n_particles, max_iters, experiment_name):
    """Ejecutar un experimento con parámetros específicos."""
    print(f"\n🧪 Experimento: {experiment_name}")
    print(f"   PSO Options: {pso_options}")
    print(f"   Partículas: {n_particles}")
    print(f"   Iteraciones: {max_iters}")
    
    # Preparar datos
    professor_assignments = prepare_professor_assignments(config)
    blocked_slots = [tuple(slot) for slot in config["blocked_slots"]]
    
    # Crear generador
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
        print_excel_result=False,  # Desactivar para experimentos rápidos
        pso_options=pso_options,
        n_particles=n_particles,
        max_iters=max_iters,
        logger=logger
    )
    
    # Ejecutar optimización
    try:
        cost, position, convergence = generator.run_pso_optimizer()
        solution = generator.get_solution()
        
        # Calcular métricas
        initial_cost = convergence[0] if convergence else float('inf')
        final_cost = cost
        improvement = initial_cost - final_cost
        improvement_percent = (improvement / initial_cost * 100) if initial_cost > 0 else 0
        
        return {
            'experiment_name': experiment_name,
            'pso_options': pso_options,
            'n_particles': n_particles,
            'max_iters': max_iters,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'convergence_iterations': len(convergence),
            'solution_length': len(solution) if solution else 0,
            'convergence_history': convergence  # Agregar historial completo
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def plot_experiments_comparison(results):
    """Crear gráfico comparativo de los experimentos."""
    if not results:
        return
    
    print("\n📊 Generando gráfico comparativo de experimentos...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colores para cada experimento
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    # 1. Convergencia de todos los experimentos
    ax1.set_title('Convergencia por Experimento', fontweight='bold')
    for i, result in enumerate(results):
        if 'convergence_history' in result and result['convergence_history']:
            ax1.plot(result['convergence_history'], 
                    label=result['experiment_name'], 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Costo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Costo final por experimento
    ax2.set_title('Costo Final por Experimento', fontweight='bold')
    names = [r['experiment_name'] for r in results]
    final_costs = [r['final_cost'] for r in results]
    bars = ax2.bar(names, final_costs, color=colors, alpha=0.7)
    ax2.set_ylabel('Costo Final')
    ax2.tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar, cost in zip(bars, final_costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_costs)*0.01,
                f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Mejora porcentual
    ax3.set_title('Mejora Porcentual', fontweight='bold')
    improvements = [r['improvement_percent'] for r in results]
    bars = ax3.bar(names, improvements, color=colors, alpha=0.7)
    ax3.set_ylabel('Mejora (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar, improvement in zip(bars, improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvements)*0.01,
                f'{improvement:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Eficiencia (mejora por iteración)
    ax4.set_title('Eficiencia (Mejora/Iteración)', fontweight='bold')
    efficiency = [r['improvement_percent'] / r['max_iters'] for r in results]
    bars = ax4.bar(names, efficiency, color=colors, alpha=0.7)
    ax4.set_ylabel('Mejora % por Iteración')
    ax4.tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar, eff in zip(bars, efficiency):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico comparativo guardado como: experiment_comparison.png")
    plt.close()

def main():
    """Función principal para experimentos."""
    print("🔬 Experimentador de Parámetros PSO")
    print("="*60)
    
    # Cargar configuración base
    config = load_config()
    
    # Definir experimentos
    experiments = [
        # Experimento 1: Configuración por defecto
        {
            'name': 'Default',
            'pso_options': {'c1': 1.5, 'c2': 1.5, 'w': 0.7},
            'n_particles': 60,
            'max_iters': 100
        },
        
        # Experimento 2: Exploración alta
        {
            'name': 'Alta Exploración',
            'pso_options': {'c1': 2.0, 'c2': 1.0, 'w': 0.9},
            'n_particles': 60,
            'max_iters': 100
        },
        
        # Experimento 3: Explotación alta
        {
            'name': 'Alta Explotación',
            'pso_options': {'c1': 1.0, 'c2': 2.0, 'w': 0.4},
            'n_particles': 60,
            'max_iters': 100
        },
        
        # Experimento 4: Más partículas
        {
            'name': 'Más Partículas',
            'pso_options': {'c1': 1.5, 'c2': 1.5, 'w': 0.7},
            'n_particles': 100,
            'max_iters': 100
        },
        
        # Experimento 5: Menos partículas, más iteraciones
        {
            'name': 'Menos Partículas + Iteraciones',
            'pso_options': {'c1': 1.5, 'c2': 1.5, 'w': 0.7},
            'n_particles': 30,
            'max_iters': 150
        },
        
        # Experimento 6: Configuración balanceada
        {
            'name': 'Balanceado',
            'pso_options': {'c1': 1.8, 'c2': 1.8, 'w': 0.6},
            'n_particles': 80,
            'max_iters': 120
        }
    ]
    
    results = []
    
    # Ejecutar experimentos
    for exp in experiments:
        result = run_experiment(
            config=config,
            pso_options=exp['pso_options'],
            n_particles=exp['n_particles'],
            max_iters=exp['max_iters'],
            experiment_name=exp['name']
        )
        if result:
            results.append(result)
    
    # Mostrar resultados comparativos
    print("\n" + "="*80)
    print("📊 RESULTADOS COMPARATIVOS")
    print("="*80)
    
    if results:
        # Ordenar por costo final
        results.sort(key=lambda x: x['final_cost'])
        
        print(f"{'Experimento':<20} {'Costo Final':<12} {'Mejora %':<10} {'Partículas':<11} {'Iters':<6}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['experiment_name']:<20} "
                  f"{result['final_cost']:<12.2f} "
                  f"{result['improvement_percent']:<10.1f} "
                  f"{result['n_particles']:<11} "
                  f"{result['max_iters']:<6}")
        
        # Mejor resultado
        best = results[0]
        print(f"\n🏆 Mejor configuración: {best['experiment_name']}")
        print(f"   • Parámetros PSO: {best['pso_options']}")
        print(f"   • Partículas: {best['n_particles']}")
        print(f"   • Iteraciones: {best['max_iters']}")
        print(f"   • Costo final: {best['final_cost']:.2f}")
        print(f"   • Mejora: {best['improvement_percent']:.1f}%")
        
        # Guardar resultados
        with open('experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en 'experiment_results.json'")
        
        # Generar gráfico comparativo
        plot_experiments_comparison(results)
    
    else:
        print("❌ No se pudieron completar los experimentos")

if __name__ == "__main__":
    main()