#!/usr/bin/env python3
"""
Script rápido para probar que todo funciona correctamente
"""

from run_analysis import load_data, make_train_test_split, make_pipeline, make_grid_search, train_estimator, check_estimator
from sklearn.linear_model import LogisticRegression

def quick_test():
    print("=== Prueba Rápida del Sistema ===")
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    x, y = load_data()
    print(f"   ✓ Datos cargados: {x.shape[0]} muestras, {x.shape[1]} características")
    
    # 2. Crear pipeline simple
    print("2. Creando pipeline...")
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    print("   ✓ Pipeline creado")
    
    # 3. Entrenar con parámetros simples
    print("3. Entrenando modelo...")
    param_grid = {
        "selectkbest__k": [5, 6, 7],
        "estimator__C": [0.1, 1, 10],
    }
    
    estimator = make_grid_search(pipeline, param_grid, cv=3)
    train_estimator(estimator)
    print("   ✓ Modelo entrenado")
    
    # 4. Evaluar
    print("4. Evaluando modelo...")
    check_estimator()
    
    print("\n=== Prueba completada exitosamente ===")

if __name__ == "__main__":
    quick_test()
