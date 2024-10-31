import numpy as np
from math import comb


# Calcula el producto tensorial de una lista de matrices
def producto_tensorial(matrices: list[np.ndarray]) -> np.ndarray:
    resultados = matrices[0]
    for matriz in matrices[1:]:
        resultados = np.kron(resultados, matriz)
    return resultados


# Crea la matriz Hadamard para n qubits (es la matriz que representa la compuerta Hadamard)
def crear_matriz_hadamard(n: int) -> np.ndarray:
    matriz_hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])  # Matriz Hadamard básica
    matrices = [matriz_hadamard] * n
    return producto_tensorial(matrices)


# Crea la matriz Uf de la funcion oráculo f:{0,1}^n -> {0,1}
def crear_matriz_oraculo(n: int, f: int) -> np.ndarray:
    # Tamaño de la matriz: 2^(n+1)
    size = 2 ** (n + 1)

    # dtype=np.complex128 permite almacenar números complejos
    Uf = np.zeros((size, size), dtype=np.complex128)

    for x in range(2 ** n):
        # f >> x mueve los bits de f a la derecha x posiciones, & 1 se queda solo con el último bit
        fx = (f >> x) & 1
        for y in [0, 1]:
            # Calcula la posición en la matriz
            state = x * 2 + y
            # Calcula la fase: será +1 si fx*y es par, -1 si es impar
            phase = (-1) ** (fx * y)
            # Asigna la fase en la diagonal de la matriz
            Uf[state][state] = phase

    return Uf


# Algoritmo Deutsch-Jozsa, Retorna 0 para funciones constantes, 1 para otras funciones
def deutsch_jozsa(n: int, f: int) -> int:
    # Estado inicial |0...0⟩|1⟩
    estado_inicial = np.zeros(2 ** (n + 1), dtype=np.complex128)
    estado_inicial[1] = 1

    # Aplicamos H⊗(n+1), es decir la transformación Hadamard a todos los n+1 qubits
    hadamard_a_cada_qubit = crear_matriz_hadamard(n + 1)  # H⊗(n+1)
    estado = hadamard_a_cada_qubit @ estado_inicial

    # Aplicamos oráculo Uf
    Uf = crear_matriz_oraculo(n, f)
    estado = Uf @ estado

    # Aplicamos H⊗n ⊗ I, es decir Hadamard a los primeros n qubits y no hacer nada al último qubit auxiliar
    hadamard_qubit_de_entrada = crear_matriz_hadamard(n)  # H⊗n
    matriz_identidad_para_qubit = np.eye(2)  # I
    trasnformada_final = producto_tensorial([hadamard_qubit_de_entrada, matriz_identidad_para_qubit])  # H⊗n ⊗ I
    estado_final = trasnformada_final @ estado

    # Medir: Si solo hay amplitud en |0...0⟩, la función es constante
    # Si hay amplitud en cualquier otro estado, la función no es constante
    prob_zero = np.abs(estado_final[0]) ** 2 + np.abs(estado_final[1]) ** 2

    # usamos un umbral para manejar errores numéricos
    if prob_zero > 0.9:
        return 0
    return 1


def es_balanceada(f: int, n: int) -> bool:
    # Determina si una función es balanceada
    cantidad_de_unos = bin(f).count('1')
    if cantidad_de_unos == 2 ** (n - 1):
        return True
    return False


def es_constante(f: int, n: int) -> bool:
    # Determina si una función es constante
    if f == 0 or f == (2 ** 2 ** n - 1):
        return True
    return False


def formato_binario(num: int, n: int) -> str:
    # Formatea un número en binario con exactamente 2^n dígitos
    return format(num, f'0{2 ** n}b')


def main():
    n = int(input("Ingrese el tamaño del problema (n): "))

    print("\nFunción | Es balanceada | Es constante | Número de bits en 1 | Resultado")
    print("-" * 72)

    for f in range(2 ** (2 ** n)):
        resultado = deutsch_jozsa(n, f)
        balanceada = es_balanceada(f, n)
        constante = es_constante(f, n)
        # Cuenta el número de unos en la representación binaria
        cantidad_de_unos = bin(f).count('1')

        print(
            f"{formato_binario(f, n)} | {str(balanceada):12} | {str(constante):11} | {str(cantidad_de_unos):16} | {resultado}")

    cantidad_de_funciones = 2 ** (2 ** n)
    funciones_balanceadas = comb(2 ** n, 2 ** (n - 1))
    funciones_constantes = 2

    print(f"\nEstadísticas:")
    print(f"Total de funciones: {cantidad_de_funciones}")
    print(f"Funciones balanceadas: {funciones_balanceadas}")
    print(f"Funciones constantes: {funciones_constantes}")
    print(f"Otras funciones: {cantidad_de_funciones - funciones_balanceadas - funciones_constantes}")


if __name__ == "__main__":
    main()
