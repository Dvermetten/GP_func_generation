

import operator
import numpy as np
from functools import partial
from .pset import create_pset
from .symb_regression import symb_regr
from deap import algorithms, base, creator, tools, gp




#%%
class GP_func_generator:
    def __init__(self,
                 doe_x,
                 target_vector,
                 dist_metric: str = 'euclidean',
                 bs_repeat: int = 2,
                 problem_label: str = '',
                 filepath_save: str = '',
                 tree_size: tuple = (3,5),
                 population: int = 100,
                 cxpb: float = 0.5, 
                 mutpb: float = 0.1,
                 ngen: int = 10,
                 nhof: int = 1,
                 verbose: bool = True
                 ):
        # optimization
        self.doe_x = doe_x
        self.target_vector = target_vector
        self.dist_metric: str = dist_metric
        self.bs_repeat: int = bs_repeat
        self.problem_label: str = problem_label
        self.filepath_save: str = filepath_save
        self.tree_size: tuple = tree_size
        self.population: int = population
        self.cxpb: float = cxpb
        self.mutpb: float = mutpb
        self.ngen: int = ngen
        self.nhof: int = nhof
        self.verbose: bool = verbose
        self.fopt = np.inf
        self.fbest = None
    
    #%%
    def evalSymbReg(self, individual, points):
        f_ = partial(symb_regr, self.pset, self.target_vector, self.dist_metric, self.bs_repeat)
        fitness, fbest = f_(individual, points)
        if (fitness < self.fopt):
            self.fopt = fitness
            self.fbest = fbest
        print(f'fitness: {fitness}; fopt: {self.fopt}')
        return fitness,
    
    #%%    
    def __call__(self):
        self.pset = create_pset(self.doe_x)()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.tree_size[0], max_=self.tree_size[1])
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        self.toolbox.register("evaluate", self.evalSymbReg, points=self.doe_x)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        
        pop = self.toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.nhof)
    
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
    
        pop, logger = algorithms.eaSimple(pop, self.toolbox, self.cxpb, self.mutpb, self.ngen, stats=mstats,
                                          halloffame=hof, verbose=self.verbose)
        if (self.verbose):
            print('[GPFG] Optimization for symbolic regression done.')
        return hof, pop, logger
# END CLASS