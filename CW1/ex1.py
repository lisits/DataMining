import numpy as np

# задаем матрицу смежности в привычном виде
A = np.array([[1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

# вычисляем матрицу вероятностей перехода
P = np.zeros_like(A, dtype=np.float64)
for i in range(A.shape[0]):
    row_sum = np.sum(A[i])
    if row_sum > 0:
        P[i] = A[i] / row_sum

# транспонируем матрицу
P = P.T

# задаем начальное распределение вероятностей
r = np.ones(A.shape[0], dtype=np.float64) / A.shape[0]
# создаём копию, чтоюы потом посчитать PageRank по двум формулам
r1 = r.copy()

# задаём коэффициент затухания
d = 0.8
# задаём точность
accuracy = 0.0001
# задаём количество итераций на случай, если до заданной точности так и не дойдём
count_of_iterations = 1000

#считаем по формуле без (1 - d)
for i in range(count_of_iterations):
    r_next = d * np.dot(P, r)
    # цикл остановится, если будет достигнута заданная точность
    if np.abs(r_next - r).sum() < accuracy:
        print("i without (1 - d): ", i)
        break
    r = r_next

#считаем по формуле с (1 - d)
for i in range(count_of_iterations):
    r1_next = (1 - d) + d * np.dot(P, r1)
    # цикл остановится, если будет достигнута заданная точность
    if np.abs(r1_next - r1).sum() < accuracy:
        print("i with (1 - d): ", i)
        break
    r1 = r1_next

print("PageRank with (1 - d):\n", r1)
print("PageRank without (1 - d):\n", r)