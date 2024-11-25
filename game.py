
import os
import numpy as np
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK

OBJ_EPSILON = 1e-12

class Game(object):

    def __init__(self, config, env, random_seed=1000):
        self.num_links = env.num_links
        self.num_pairs = env.num_pairs
        self.num_nodes = env.num_nodes
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.shortest_paths_node = env.shortest_paths_node  # paths with node info
        self.shortest_paths_link = env.shortest_paths_link  # paths with link info
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.get_ecmp_next_hops()

        # for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
        self.load_multiplier = {}


#ecmp计算方法
    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        if next_hops_cnt > 1:
            print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads
    def eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

#lp 解决方案
    def lp_optimal_routing_mlu(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")

        # 对应公式 （4e）
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        r = LpVariable(name="congestion_ratio")
        # 对应公式 （4d ）
        for pr in self.lp_pairs:
            model += (
            lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum(
                [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1,
            "flow_conservation_constr1_%d" % pr)

        for pr in self.lp_pairs:
            model += (
            lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum(
                [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
            "flow_conservation_constr2_%d" % pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum(
                        [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              "flow_conservation_constr3_%d_%d" % (pr, n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr] * ratio[pr, e[0], e[1]] for pr in self.lp_pairs]),
                      "link_load_constr%d" % ei)
            model += (link_load[ei] <= self.link_capacities[ei] * r, "congestion_ratio_constr%d" % ei)

        model += r + OBJ_EPSILON * lpSum([link_load[e] for e in self.links])

        model.solve(solver=GLPK(msg=False))
        #model.solve(solver=GUROBI(msg=False))

        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        pathk = {}
        for k in ratio:
            solution[k] = ratio[k].value()
            if ratio[k].value() > 0.0:
                pathk[k] = ratio[k].value()



        print('*******************')
        print(solution)
        print(pathk)

        return obj_r, solution

    def eval_lp_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand * solution[i, e[0], e[1]]

        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_max_utilization, delay


    #强化学习选择SR中间节点方案
    def mid_node_traffic_distribution(self, tm_idx, mid_nodes):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, mid_nodes[pair_idx])
                self.ecmp_next_hop_distribution(link_loads, demand, mid_nodes[pair_idx], d)

        return link_loads
    def optimal_routing_mlu_mid_node(self, tm_idx, mid_nodes, eval_delay=False):
        eval_link_loads = self.mid_node_traffic_distribution(tm_idx, mid_nodes)

        sr_pair_solution = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            mid = mid_nodes[i]
            for k in range(self.num_pairs):
                sr_pair_solution[k] = s, d, mid
            #background link load

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        obj_r = eval_max_utilization
        solution = sr_pair_solution

        return obj_r, delay, solution


    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, History]
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx-h])
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h] / tm_max_element        #[Valid_tms, Node, Node, History]
                else:
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h]                         #[Valid_tms, Node, Node, History]

class SRv6TE_Game(Game):
    def __init__(self, config, env, random_seed=1000):
        super(SRv6TE_Game, self).__init__(config, env, random_seed)

        self.project_name = config.project_name
        self.action_dim = env.num_pairs * env.num_nodes

        self.tm_history = 1
        self.tm_indexes = np.arange(self.tm_history - 1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        if config.method == 'pure_policy':
            self.baseline = {}
        self.generate_inputs(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)

    def get_state(self, tm_idx):
        idx_offset = self.tm_history - 1
        return self.normalized_traffic_matrices[tm_idx - idx_offset]
    def reward(self, tm_idx, actions):
        mlu, _, _ = self.optimal_routing_mlu_mid_node(tm_idx, actions)

        reward = 1 / mlu

        return reward

    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]

        # print(reward, (total_v/cnt))

        return reward - (total_v / cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    def evaluate(self, tm_idx, ecmp=True, eval_delay=False):

        #ecmp solve traffic engineering problem
        line = str(tm_idx)
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx, eval_delay=eval_delay)
            line += ', ' + str(ecmp_mlu) + ', ' + str(ecmp_delay) + ', '

        # LP method
        _, solution = self.lp_optimal_routing_mlu(tm_idx)
        lp_mlu, lp_delay = self.eval_lp_optimal_routing_mlu(tm_idx, solution, eval_delay=eval_delay)
        line += str(lp_mlu) + ', ' + str(lp_delay) + ', '

        print(line[:])