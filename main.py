import random
import math

import numpy as np

from argparse import ArgumentParser
from math import ceil, log2

from dimod import Binary, Integer, ConstrainedQuadraticModel, ExactCQMSolver, cqm_to_bqm
from dwave.samplers import TabuSampler, SimulatedAnnealingSampler
from dwave.system import DWaveSampler, LeapHybridSampler, LeapHybridCQMSampler
from dwave.system.composites import EmbeddingComposite

def eval_u2(binary_vars):
    """Convert U2 bit array to signed integer."""
    n = len(binary_vars)
    sign_bit = -binary_vars[0] * (2 ** (n - 1))
    magnitude = sum(binary_vars[b] * (2 ** (n - b - 1)) for b in range(1, n))
    return sign_bit + magnitude

class SchedulingProblem:
    def __init__(self, processing_times, due_dates, tardiness_weights, earliness_weights=[], args=None):
        """Initialize the model with processing times, due dates, weights, and penalties."""
        self.n = len(processing_times)
        self.p = processing_times
        self.d = due_dates
        self.w = tardiness_weights
        self.u = earliness_weights
        self.cqm = ConstrainedQuadraticModel()
        self.no_bqm = args.no_bqm if args is not None else False
        self.improved = args.improved if args is not None else False
        self.eiui = args.eiui if args is not None else False
        self.solver = args.solver if args is not None else 'simulator'
        self.num_reads = args.num_reads if args is not None else 1000

        # Define decision variables
        self.max_time = sum(self.p) * 1.5
        self.M = 2 * self.max_time
        if self.improved:
            #print("[debug] Using improved model")
            self.L = {i: [Binary(f'L_{i}_{b}') for b in range(ceil(log2(self.max_time)) + 1)] for i in range(self.n)}
        else:
            self.S = {i: Integer(f'S_{i}', lower_bound=0, upper_bound=self.max_time) for i in range(self.n)}  # Job start times
            self.T = {i: Integer(f'T_{i}', lower_bound=0, upper_bound=self.max_time) for i in range(self.n)}  # Tardiness values
            if self.eiui:
                self.E = {i: Integer(f'E_{i}', lower_bound=0, upper_bound=self.max_time) for i in range(self.n)}  # Earliness values
            
        self.x = {(i, j): Binary(f'x_{i}_{j}') for i in range(self.n) for j in range(i+1,self.n)}  # Order variables

    def add_STE_constraints(self):
        for i in range(self.n):
            self.cqm.add_constraint(
                    self.S[i] + self.p[i] - self.d[i] - self.T[i] <= 0, label=f'constraint1_{i}')
            if self.eiui:
                self.cqm.add_constraint(
                        -self.S[i] - self.p[i] + self.d[i] - self.E[i] <= 0,
                        label=f'constraint2_{i}'
                )

        for i in range(self.n):
            for j in range(i+1,self.n):
                self.cqm.add_constraint(
                    self.S[i] + self.p[i] - self.S[j] - self.M + self.M * self.x[i, j] <= 0,
                    label=f'constraint2_{i}_{j}'
                )
                self.cqm.add_constraint(
                    self.S[j] + self.p[j] - self.S[i] - self.M * self.x[i, j] <= 0,
                    label=f'constraint3_{i}_{j}'
                )

    def add_L_constraints(self):
#        assert not self.eiui, "TODO: improved eiui"
        for i in range(self.n):
            L_i = eval_u2(self.L[i])


            if self.solver != 'hybrid':
                constraint_bound_1 = self.max_time - self.p[i] + self.d[i]
    
                slack_1 = Integer(f'slack1_{i}', lower_bound=0, upper_bound=constraint_bound_1)

                self.cqm.add_constraint(
                    -L_i + self.p[i] - self.d[i] + slack_1 == 0,
                    label=f'constraint1_{i}'
                )
            else:
                self.cqm.add_constraint(
                    -L_i + self.p[i] - self.d[i] <= 0,
                    label=f'constraint1_{i}'
                )

        for i in range(self.n):
            for j in range(i):
                L_i = eval_u2(self.L[i])
                L_j = eval_u2(self.L[j])

                if self.solver != 'hybrid':
                    constraint_bound_2 = self.max_time - self.d[i] + self.d[j] - self.p[j] + self.max_time
                    constraint_bound_3 = self.max_time + self.d[i] - self.d[j] - self.p[i] + self.max_time

                    slack_2 = Integer(f'slack2_{i}_{j}', lower_bound=0, upper_bound=constraint_bound_2)
                    slack_3 = Integer(f'slack3_{i}_{j}', lower_bound=0, upper_bound=constraint_bound_3)

                    self.cqm.add_constraint(
                        L_i - L_j + self.d[i] - self.d[j] + self.p[j] - self.max_time + self.max_time * self.x[j, i] + slack_2 == 0,
                        label=f'constraint2_{i}_{j}'
                    )
                    self.cqm.add_constraint(
                        L_j - L_i - self.d[i] + self.d[j] + self.p[i] - self.max_time * self.x[j, i] + slack_3 == 0,
                        label=f'constraint3_{i}_{j}'
                    )
                else:
                    self.cqm.add_constraint(
                        L_i - L_j + self.d[i] - self.d[j] + self.p[j] - self.max_time + self.max_time * self.x[j, i] <= 0,
                        label=f'constraint2_{i}_{j}'
                    )
                    self.cqm.add_constraint(
                        L_j - L_i - self.d[i] + self.d[j] + self.p[i] - self.max_time * self.x[j, i] <= 0,
                        label=f'constraint3_{i}_{j}'
                    )

    def add_constraints(self):
        """Add problem constraints to the model."""
        if self.improved:
            self.add_L_constraints()
        else:
            self.add_STE_constraints()


    def define_objective(self):
        """Define the objective function: minimize weighted tardiness."""
        if self.improved:
            self.cqm.set_objective(
                    sum((self.w[i] * (1 - self.L[i][0]) * eval_u2(self.L[i])) for i in range(self.n))
            )
        else:
            if self.eiui:
                self.cqm.set_objective(
                        sum((self.w[i] * self.T[i] + self.u[i] * self.E[i]) for i in range(self.n))
                )
            else:
                self.cqm.set_objective(sum(self.w[i] * self.T[i] for i in range(self.n)))

    def solve_cqm(self):
        #print('Using LeapHybrid (CQM)')
        #sampler = ExactCQMSolver()
        sampler = LeapHybridCQMSampler()
        result = sampler.sample_cqm(self.cqm)

        var_counts = {
            'vars': len(self.cqm.variables),
            'slacks': 0,
            'total': len(self.cqm.variables)
        }

        decoded_solution = result.first.sample
        if self.improved:
            #
            schedule = {}
            width = ceil(log2(self.max_time))
            for i in range(self.n):
                L = 0
                for b in range(width):
                    bit = decoded_solution[f'L_{i}_{b}']
                    L += bit*(2**b)
                bit = decoded_solution[f'L_{i}_{width}']
                L = -(2**(width))*bit + L
                schedule[i] = L
        else:
            schedule = {i: decoded_solution[f'S_{i}'] for i in range(self.n)}

        feasible_sampleset = result.filter(lambda d: d.is_feasible)
        num_feasible = len(feasible_sampleset)
        if num_feasible > 0:
            decoded_solution = feasible_sampleset.first.sample
        else:
            decoded_solution = result.first.sample

        if self.improved:
            #
            schedule2 = {}
            width = ceil(log2(self.max_time))
            for i in range(self.n):
                L = 0
                for b in range(width):
                    bit = decoded_solution[f'L_{i}_{b}']
                    L += bit*(2**b)
                bit = decoded_solution[f'L_{i}_{width}']
                L = -(2**(width))*bit + L
                schedule2[i] = L
        else:
            schedule2 = {i: decoded_solution[f'S_{i}'] for i in range(self.n)}

        return schedule, schedule2, var_counts

    def solve_bqm(self):
        """Convert CQM to BQM and solve using Simulated Annealing."""
        bqm, invert_map = cqm_to_bqm(self.cqm)
        bqm.normalize()

        # Count slack variables (strings starting with 'slack_')
        slack_count = 0#sum(1 for var in bqm.variables if isinstance(var, str) and var.startswith('slack'))
        for var in bqm.variables:
            if isinstance(var, tuple):
                var = var[0]

            if var.startswith('slack'):
                slack_count += 1
        bqm_var_counts = {
            'vars': len(bqm.variables) - slack_count,
            'slacks': slack_count,
            'total': len(bqm.variables)
        }

        if self.solver == 'simulator':
            #print('Using simulator')
            sampler = SimulatedAnnealingSampler()
        elif self.solver == 'quantum':
            #print('Using QPU')
            sampler = EmbeddingComposite(DWaveSampler())
        result = sampler.sample(bqm, num_reads=self.num_reads, embedding_parameters=dict(timeout=60))
        best_solution = result.first.sample

        best_solution = { k: np.int16(v) for k, v in best_solution.items() }

        # Extract integer values from binary variables
        decoded_solution = invert_map(best_solution)

        if self.improved:
            #
            schedule = {}
            width = ceil(log2(self.max_time))
            for i in range(self.n):
                L = 0
                for b in range(width):
                    bit = decoded_solution[f'L_{i}_{b}']
                    L = L | (bit << b)
                bit = decoded_solution[f'L_{i}_{width}']
                L = -(2**(width))*bit + L
                schedule[i] = L
            return schedule, schedule, bqm_var_counts
        else:
            return {i: decoded_solution[f'S_{i}'] for i in range(self.n)}, {i: decoded_solution[f'S_{i}'] for i in range(self.n)}, bqm_var_counts

    def solve(self):
        if self.solver == 'hybrid':
            return self.solve_cqm()
        else:
            return self.solve_bqm()

class RandomNumberGenerator:
    def __init__(self, seedVaule=None):
        self.__seed=seedVaule
    def nextInt(self, low, high):
        m = 2147483647
        a = 16807
        b = 127773
        c = 2836
        k = int(self.__seed / b)
        self.__seed = a * (self.__seed % b) - k * c
        if self.__seed < 0:
            self.__seed = self.__seed + m
        value_0_1 = self.__seed
        value_0_1 =  value_0_1/m
        return low + int(math.floor(value_0_1 * (high - low + 1)))
    def nextFloat(self, low, high):
        low*=100000
        high*=100000
        val = self.nextInt(low,high)/100000.0
        return val

class Generator:
    def __init__(self, seed, tf, rdd, py_rng=False):
        self.seed = seed
        self.TF = tf
        self.RDD = rdd
        self.py_rng = py_rng

    def py_rng_generate(self, n):
        random.seed(self.seed)

        P = 0
        p = [ random.randint(1, 100) for _ in range(n) ]
        P = sum(p)
        w = [ random.randint(1, 10) for _ in range(n) ]
        lower = math.floor(P * (1 - self.TF - self.RDD / 2))
        upper = math.floor(P * (1 - self.TF + self.RDD / 2))
        d = [ random.randint(lower, upper) for _ in range(n) ]
        u = [ random.randint(1, 10) for _ in range(n) ]
        return p, w, d, u

    def rng_generate(self, n):
        rng = RandomNumberGenerator(self.seed)

        P = 0
        p = [ rng.nextInt(1, 100) for _ in range(n) ]
        P = sum(p)
        w = [ rng.nextInt(1, 10) for _ in range(n) ]
        lower = math.floor(P * (1 - self.TF - self.RDD / 2))
        upper = math.floor(P * (1 - self.TF + self.RDD / 2))
        d = [ rng.nextInt(lower, upper) for _ in range(n) ]
        u = [ rng.nextInt(1, 10) for _ in range(n) ]
        return p, w, d, u
    
    def generate(self, n):
        if self.py_rng:
            return self.py_rng_generate(n)
        else:
            return self.rng_generate(n)

class Solution:
    def __init__(self, n, start_times, p, d, w, u, fixed_up):
        self.n = n
        self.start_times = start_times
        self.p = p
        self.d = d
        self.w = w
        self.u = u
        self.fixed_up = fixed_up

    def cost(self):
        eiui = self.u is not None
        cost = 0
        for i in range(self.n):
            start_time = int(self.start_times[i])
            completion_time = start_time + self.p[i]
            tardiness = max(0, completion_time - self.d[i])
            cost += tardiness * self.w[i]
            if eiui:
                earliness = max(0, self.d[i] - completion_time)
                cost += earliness * self.u[i]
        return cost

    def print_table(self):
        eiui = self.u is not None
        width = 57 if eiui else 46
        print("-" * width)
        if eiui:
            print(f"{'Job':<5} {'Start Time':<12} {'Completion Time':<17} {'Tardiness':<10} {'Earliness':<10}")
        else:
            print(f"{'Job':<5} {'Start Time':<12} {'Completion Time':<17} {'Tardiness':<10}")
        print("-" * width)
        cost = 0
        for i in range(self.n):
            start_time = int(self.start_times[i])
            completion_time = start_time + self.p[i]
            tardiness = max(0, completion_time - self.d[i])
            cost += tardiness * self.w[i]
            if eiui:
                earliness = max(0, self.d[i] - completion_time)
                #cost += earliness * self.u[i]
                print(f"{i:<5} {start_time:<12} {completion_time:<17} {tardiness:<10} {earliness:<10}")
            else:
                print(f"{i:<5} {start_time:<12} {completion_time:<17} {tardiness:<10}")
        print("-" * width)
        print(f"Cost: {cost}")

    def print(self):
        pass

    def repairWiTi(self, n, S, p, w, d):
        indexes = [i for i in range(n)]
        order = sorted(indexes, key=lambda i: S[i])

        time = 0
        newS = []
        newC = []
        goal = 0
        for i in range(n):
            newS.append(time)
            time += p[order[i]]
        return newS

    def repairWiTiuiEi(self, n, S, p, w, u, d):
        indexes = [i for i in range(n)]
        order = sorted(indexes, key=lambda i: S[i])

        newS = [S[order[i]] for i in range(n)]

        for i in range(n-1, 0, -1):
            if newS[i-1] + p[order[i-1]] > newS[i]:
                newS[i-1] = newS[i] - p[order[i-1]]

        for i in range(n-1):
            if newS[i] + p[order[i]] > newS[i+1]:
                newS[i+1] = newS[i] + p[order[i]]

        #goOn = True
        #leftToRight = True
        #while goOn:
        #    goOn = False
        #    if leftToRight:
        #        for i in range(n - 1):
        #            if newS[i] + p[order[i]] > newS[i + 1]:
        #                print(f'LtR')
        #                print(f'    i={i}, newS', newS, ', order', order, ', p', p)
        #                newS[i] = newS[i + 1] - p[order[i]]
        #                goOn = True
        #                break
        #    else:
        #        for i in range(n - 1, 0, -1):
        #            if newS[i] < newS[i - 1] + p[order[i - 1]]:
        #                print(f'RtL')
        #                print(f'    i={i}, newS', newS, ', order', order, ', p', p)
        #                newS[i] = newS[i - 1] + p[order[i - 1]]
        #                goOn = True
        #                break
        #    leftToRight = not leftToRight

        if newS[0] < 0:
            newS[0] = 0
        for i in range(n - 1):
            if newS[i] + p[order[i]] > newS[i + 1]:
                newS[i + 1] = newS[i] + p[order[i]]
        return newS

    def fixup(self):
        if self.u is not None:
            newS = self.repairWiTiuiEi(self.n, self.start_times, self.p, self.w, self.u, self.d)
        else:
            newS = self.repairWiTi(self.n, self.start_times, self.p, self.w, self.d)
        return Solution(self.n, newS, self.p, self.d, self.w, self.u, False)

    @staticmethod
    def from_S(n, S, p, d, w, u=None):
        return Solution(n, S, p, d, w, u, False)

    @staticmethod
    def from_L(n, L, p, d, w, u=None):
        start_times = [ L[i] + d[i] - p[i] for i in range(len(L)) ]
        return Solution(n, start_times, p, d, w, u, False)

if __name__ == "__main__":
    show_log=False
    parser = ArgumentParser()
    parser.add_argument("--no-bqm", action="store_true", help="Do not convert the problem to BQM before running")
    parser.add_argument("--use-L", action="store_true", dest="improved", help="Use L instead of S, T, E")
    parser.add_argument("--uiEi", action="store_true", dest="eiui", help="Solve wiTi+uiEi")
    parser.add_argument('--seed', type=int, default=42, help="Seed for the generator")
    parser.add_argument('--size', type=int, default=4, help="Number of jobs")
    parser.add_argument('--tf', type=float, default=0.2, help="TF generator parameter")
    parser.add_argument('--rdd', type=float, default=0.2, help="RDD generator parameter")
    parser.add_argument('--no-fixup', action='store_false', dest='fixup', help="Don't fixup results")
    parser.add_argument('--solver', choices=['simulator', 'hybrid', 'quantum'], default='simulator')
    parser.add_argument('--num-reads', type=int, default=2000)

    args = parser.parse_args()

    gen = Generator(args.seed, args.tf, args.rdd)

    p, w, d, u = gen.generate(args.size)

    # Initialize and solve
    problem = SchedulingProblem(p, d, w, u, args=args)
    problem.add_constraints()
    problem.define_objective()
    solution, solution2, counts = problem.solve()

    #print("Instance:")
    #print("p:", p)
    #print("d:", d)
    #print("w:", w)
    #print("")

    #if solution:
        #print("\nOptimal Schedule:")
    if args.improved:
        soln = Solution.from_L(problem.n, solution, p, d, w, u if args.eiui else None)
    else:
        soln = Solution.from_S(problem.n, solution, p, d, w, u if args.eiui else None)
    soln = soln.fixup()
    cost = soln.cost()

    if args.improved:
        soln = Solution.from_L(problem.n, solution2, p, d, w, u if args.eiui else None)
    else:
        soln = Solution.from_S(problem.n, solution2, p, d, w, u if args.eiui else None)
    soln = soln.fixup()
    cost2 = soln.cost()

    print("p =", p)
    print("w =", w)
    print("d =", d)
    print("u =", u)

    # n, seed, TF, RDD, w/u, noL/L, Q/H, cost, vars, slack, total
    print(','.join(map(str, [
        args.size,
        args.seed,
        args.tf,
        args.rdd,
        'uiEi' if args.eiui else 'wiTi',
        'L' if args.improved else 'noL',
        args.solver[0],
        cost if cost < cost2 else cost2,
        counts['vars'],
        counts['slacks'],
        counts['total']
    ])))
    #else:
    #    print("No solution found.")
