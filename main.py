# 导入deap和其他需要的库
from deap import base, creator, gp, tools
import operator
import random
from local_optimize import *
from evaluate import *
def dataset():
    # 准备数据
    # 生成一个符合y = x**2 + x + 1关系的数据集
    rng = np.random.RandomState(0)
    X_train = rng.uniform(-10, 5, 5)
    y = list(X_train**2 + 3*X_train + 1)
    return X_train,y

def define_gp(X_train,y):
    # 定义一个保护除法函数，避免除零错误
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    # 定义一个新的终端符号类型：临时常数
    class TemporaryConstant(gp.Terminal):
        def __init__(self, name):
            super(TemporaryConstant, self).__init__(name)
            self.value = random.uniform(-1, 1)

        def format(self):
            return str(self.value)
    pset = gp.PrimitiveSet("MAIN", 2) # 创建了一个名为 "MAIN" 的原语集，其中包含一个变量，名称默认为 "ARG0"
    pset.addPrimitive(operator.add, 2)
    #pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    #pset.addPrimitive(protectedDiv, 2)
    #pset.addPrimitive(operator.neg, 1)
    #pset.addPrimitive(math.cos, 1)
    #pset.addPrimitive(math.sin, 1)
    # 临时常数是一种特殊类型的常数，它的值在每次创建新个体时都会重新生成
    # 添加了一个名称为 "RAND" 的临时常数。它的值由一个匿名函数生成，该函数使用 random.uniform(-2, 2) 来生成 [-2, 2] 范围内的随机数
    pset.addEphemeralConstant("RAND", lambda: random.uniform(-2, 2))

    pset.renameArguments(ARG0='x')
    # pset.renameArguments(ARG1='t')

    # 定义一个适应度类，继承自base.Fitness，并指定weights属性为负数，表示越小越好
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))

    # 定义一个个体类，继承自gp.PrimitiveTree，并添加fitness属性和params属性，并添加local_optimize方法

    creator.create("Individual", gp.PrimitiveTree,
                   fitness=creator.FitnessMin,
                   params=None,
                   local_optimize=local_optimize)

    # 创建一个基类对象，并注册相关的属性和方法
    toolbox = base.Toolbox()

    # 注册表达式生成器，使用ramped half-and-half方法，并指定最大深度为4
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

    # 注册个体生成器，使用表达式生成器，并将结果转换为个体类
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 注册种群生成器，使用个体生成器，并指定种群大小为1000
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 它还注册了一个编译函数，用于将树表达式转换为可调用函数
    toolbox.register("compile", gp.compile, pset=pset)


    # 注册适应度评估函数，使用均方误差作为评估指标，并传入数据集作为参数


    toolbox.register("evaluate", evalSymbReg,X_train=X_train,y=y,toolbox=toolbox)

    # 注册选择算子
    toolbox.register("select", tools.selNSGA2)  # 使用NSGA-II选择算子

    # 注册交叉算子，使用one point交叉，并指定交叉概率为0.5
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 注册变异算子，使用subtree变异，并指定
    # 注册变异算子，使用subtree变异，并指定变异概率为0.1
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # 使用完全生成法作为变异算子的基础
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # 使用均匀变异算子

    # toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    return toolbox,pset




def evolution_gp(toolbox,pset,X_train,y):

    # 创建一个随机数生成器，并指定随机种子为0
    random.seed(0)

    # 创建一个种群，并初始化每个个体的参数和适应度值
    pop = toolbox.population(n=1000)
    for ind in pop:
        # 编译个体的表达式为一个可执行的函数，并赋值给expr属性
        ind.expr = gp.compile(ind, pset)
        # 提取个体中的常数系数
        ind.params = [node.value for node in ind if isinstance(node, gp.RAND)]
        # 评估个体的适应度值
        ind.fitness.values = toolbox.evaluate(ind)

    # 定义进化的参数，比如进化代数、交叉概率、变异概率等
    ngen = 20
    cxpb = 0.5
    mutpb = 0.1

    # 进行进化，每一代都进行选择、交叉、变异和评估，并记录统计信息
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    best_inds = []  # 记录每一代的最优个体
    hof = tools.HallOfFame(1)

    for g in range(ngen):
        # 选择下一代的个体
        offspring = toolbox.select(pop, len(pop))
        # 克隆每个个体，避免修改原始种群
        offspring = list(map(toolbox.clone, offspring))

        # 对选出的个体进行交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 对每个新生成的个体进行局部优化和评估
        for ind in offspring:
            ind.expr = gp.compile(ind, pset)
            if not ind.fitness.valid:
                # 调用局部优化方法
                ind.local_optimize(X_train, y,pset)
                # 评估个体的适应度值
                ind.fitness.values = toolbox.evaluate(ind)

        # 更新种群
        pop[:] = offspring

        # 记录统计信息并打印输出
        record = mstats.compile(pop)
        best_individual = tools.selBest(pop, k=1)[0]

        hof.update(pop)
        best_ind = hof.items[0]
        best_inds.append(best_ind)
        print(f"Generation {g}: {best_individual}")


def main():
    X_train,y_train = dataset()
    toolbox,pset = define_gp(X_train,y_train)
    evolution_gp(toolbox,pset,X_train,y_train)

if __name__ == "__main__":
    main()