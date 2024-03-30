# coding:UTF-8
# @Time: 2023/4/19 20:29
# @Author: Lulu Cao
# @File: NDsort.py
# @Software: PyCharm



def nondominated_sorting_selection(population, n_parents):
    """
    非支配排序选择函数，适应度值越小越好
    :param population:是一个列表，表示种群中的所有个体
    :param n_parents: 是一个整数，表示要选择的父代个体数量
    :return:
    """
    # 计算每个个体的支配个数和被支配个体集合
    dominates = [set() for _ in population] # 列表中的每个元素是一个空集合
    dominated_by = [0] * len(population) # 列表中的每个元素是0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # 如果元素 i 的所有 fitness_ 值都小于等于元素 j 的对应值，并且至少有一个值小于元素 j 的对应值，那么元素 i 就支配元素 j
            if all(population[i].fitness_ <= population[j].fitness_) and any(
                    population[i].fitness_ < population[j].fitness_):
                dominates[i].add(j)
                dominated_by[j] += 1
            elif all(population[j].fitness_ <= population[i].fitness_) and any(
                    population[j].fitness_ < population[i].fitness_):
                dominates[j].add(i)
                dominated_by[i] += 1

    # 计算非支配排序
    # 首先创建一个空列表 fronts，然后找到所有被支配次数为0的个体，将它们放入 current_front 列表中
    fronts = []
    current_front = [i for i in range(len(population)) if dominated_by[i] == 0]

    # 当 current_front 不为空时，循环执行以下操作：
    # 创建一个空列表 next_front，遍历 current_front 中的每个元素 i，对于每个被 i 支配的元素 j，将其被支配次数减1，
    # 如果减1后其被支配次数为0，则将其加入 next_front 列表中。
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        fronts.append(current_front)
        current_front = next_front

    # 在从非支配排序的结果中选择前 n_parents 个个体作为父代
    parents = []
    for front in fronts:
        if len(parents) + len(front) <= n_parents:
            parents.extend(front)
        else:
            remaining = n_parents - len(parents)
            front.sort(key=lambda i: population[i].fitness_[1])  # 取决于哪个适应度值最重要
            parents.extend(front[:remaining])
            break

    return [population[i] for i in parents]
