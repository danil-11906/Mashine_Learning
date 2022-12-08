# Диафантовые уравнения
# Пример: 3x + 2y + z = 10
import random

# Функция для формирования расределения вероятностей
def create_distribution(results):
    reverses = [1 / res for res in results]
    sum_r = sum(reverses)
    distribution = [reverses[0] / sum_r]
    for i in range(1, len(results)):
        distribution.append(distribution[i-1] + reverses[i] / sum_r)
    return distribution


# N - Длинна уравнения, кол-во коэфицинтов
# K - Коэфицент для формирования популяции. Кол-во популяций равно N * K
# KK - Кол-во порожденных покалений
N = 6
K = 6
KK = 20

# coefs - [k1, k2, …, kn] - коэффициенты уравнения, kn - свободный член
COEFS = [random.randint(1, 20) for i in range(N -1)]
COEFS.append(random.randint(100, 200))


# max_vars - [mv_1, mv_2, …, mv_n-1] - максимально возможные коэффициенты при переменных
# max_vars[I] - вычисляется как k_n - sum(k_1, k_2, …, k_i-1, k_i+1, …., k_n-1)
max_vars = [COEFS[N - 1] - sum(COEFS[:N - 1]) + COEFS[i] for i in range(N - 1)]

# Count = n * K, где K - коэффицент, вводимой программой
count = N * K
population = [
    [random.randint(1, max_vars[i]) for i in range(N - 1)]
    for j in range(count)
]

# Прогоняем формирования популяции через KK покалений
for _ in range(KK):
    # Прогоняем удаление элементов популяции N - 1
    for i in range(N-1):
        # Считаем насколько отличается популяция от ответа для каждого элемента популяции
        results = [COEFS[N - 1] - sum([COEFS[i] * population[k][i] for i in range(N - 1)]) for k in range(count)]
        # Делаем распеределение для разниц с ответом
        distributioins = create_distribution(results)
        # Имитируем русскую рулетку
        rand = random.random()
        for j in range(len(distributioins)):
            if (distributioins[j] >= rand):
                break
        # Убираем из поплуции выпавшие элементы
        population.pop(j)
        results.pop(j)
        count -= 1

    # Потом для новой популяции пересраиваются вероятности.
    # И считается распределение для новой популяции. Так повторяется n-1 раз
    # n - кол-во коэффициентов исходного уравнения
    # Популяция, которая осталась скрещивается:
    # Если кол-во в популяции четное, составляем рандомные пары. Если нет, то один элемент останется.
    # Создаем новую популяцию на основе предыдущей.
    # Продолжаем делать так N раз
    pairs = []
    for i in range(count // 2):
        pairs.append([
            population.pop(random.randint(0, len(population) - 1)),
            population.pop(random.randint(0, len(population) - 1))
        ])
    if (count % 2 == 1):
        pairs.append([population[len(population) - 1], []])

    new_population = []
    for i in range(len(pairs)):
        if (len(pairs[i][1]) == 0):
            new_population.append(pairs[i][0])
        else:
            new_population.append(pairs[i][0][:len(pairs[i][0]) // 2] + pairs[i][1][len(pairs[i][0]) // 2:])
            new_population.append(pairs[i][1][:len(pairs[i][0]) // 2] + pairs[i][0][len(pairs[i][0]) // 2:])
    population = new_population

    # Если в попупуляции осталась одна особь, ее берем как идеальную
    if len(results) == 1:
        break

# Выводим коэффиценты и популяцию
print(COEFS)
print(population[0])