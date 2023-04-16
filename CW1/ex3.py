import numpy as np

# задаём коэффициент затухания бета
b = 0.8
# задаём точность
accuracy = 0.0001
# задаём количество итераций на случай, если до заданной точности так и не дойдём
count_of_iterations = 100
# задаём глубину
n = 4
# считаем
m = pow(2, n) - 1
# создаём пустую матрицу размера m
mass = [0] * m
for i in range(m):
    mass[i] = [0] * m
# заполняем матрицу значениями и переводим в формат для работы с numpy
mass[0][0] = 1
for i in range(m):
    for j in range(m):
        if (j == 2 * i + 1 or j == 2 * i + 2):
            mass[i][j] = 1
        # раскомментировать для генерации матрицы для обработки dead ends
        # if (i >= pow(2, n - 1) - 1 and j != i):
        #     mass[i][j] = 1
for i in range(m):
    print(mass[i])
A = np.array(mass)

# создаём единичный вектор, у которого размерность равна количеству вершин
e = np.zeros(shape=m)

# заполняем единичный вектор значениями по формуле (1 - b) / m,
for i in range(m):
    e[i] = (1 - b) / m
# вычисляем матрицу вероятностей перехода
M = np.zeros_like(A, dtype=np.float64)
for i in range(A.shape[0]):
    row_sum = np.sum(A[i])
    if row_sum > 0:
        M[i] = A[i] / row_sum

# транспонируем матрицу и сразу умножаем на b, чтобы не делать этого
# в каждой итерации
M = M.T * b

# задаем векторы начального распределения вероятностей
v = np.ones(A.shape[0], dtype=np.float64) / A.shape[0]

# считаем по формуле
for i in range(count_of_iterations):
    v_next = e + np.dot(M, v)
    # цикл остановится, если будет достигнута заданная точность
    if np.abs(v_next - v).sum() < accuracy:
        print("Цикл был остановлен на итерации ", i + 1)
        break
    v = v_next

print("PageRank: ", v)
print("Сумма PageRank'ов: ", sum(v))
