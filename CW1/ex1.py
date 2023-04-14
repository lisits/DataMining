import numpy as np

# задаём коэффициент затухания бета
b = 0.8
# задаём точность
accuracy = 0.0001
# задаём количество итераций на случай, если до заданной точности так и не дойдём
count_of_iterations = 10
# задаем матрицу смежности в привычном виде
A = np.array([[1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])
# задаём количество вершин
n = 3
# создаём единичный вектор, у которого размерность равна количеству вершин
e = np.zeros(shape=n)
# заполняем единичный вектор значениями по формуле (1 - b) / n,
for i in range(n):
    e[i] = (1 - b) / n
# print("e ", e)

# вычисляем матрицу вероятностей перехода
M = np.zeros_like(A, dtype=np.float64)
for i in range(A.shape[0]):
    row_sum = np.sum(A[i])
    if row_sum > 0:
        M[i] = A[i] / row_sum

# транспонируем матрицу и сразу умножаем на b, чтобы не делать этого
# в каждой итерации
M = M.T * b

# задаем векторв начального распределения вероятностей
v = np.ones(A.shape[0], dtype=np.float64) / A.shape[0]

# считаем по формуле
for i in range(count_of_iterations):
    v_next = e + np.dot(M, v)
    # цикл остановится, если будет достигнута заданная точность
    if np.abs(v_next - v).sum() < accuracy:
        break
    v = v_next

print("PageRank: ", v)
print("Сумма PageRank'ов: ", sum(v))

