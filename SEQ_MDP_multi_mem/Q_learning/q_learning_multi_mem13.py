import numpy as np
import pickle
from collections import defaultdict
import random
import os
import itertools
from scipy.stats import binom

class QuantumNetworkEnv:
    def __init__(self, p, ps, C):
        self.N = len(C)
        self.p = p
        self.ps = ps
        self.C = C
        self.memory_allocation = [{} for _ in range(self.N)]
        for i in range(self.N):
            if i == 0:
                self.memory_allocation[i][1] = self.C[i]
            elif i == self.N - 1:
                self.memory_allocation[i][self.N - 2] = self.C[i]
            else:
                w_left = 1 / self.p[i - 1]
                w_right = 1 / self.p[i]
                total = w_left + w_right
                m_left = int(np.floor(self.C[i] * w_left / total))
                m_right = self.C[i] - m_left
                if self.C[i] >= 2:
                    if m_left == 0:
                        m_left = 1
                        m_right = self.C[i] - 1
                    elif m_right == 0:
                        m_right = 1
                        m_left = self.C[i] - 1
                self.memory_allocation[i][i - 1] = m_left
                self.memory_allocation[i][i + 1] = m_right

    def get_action_space(self, state):
        swap_options = []
        for i in range(1, self.N - 1):
            left_links = [state[j, i] for j in range(i) if not np.isinf(state[j, i])]
            right_links = [state[i, j] for j in range(i + 1, self.N) if not np.isinf(state[i, j])]
            if left_links and right_links:
                total_left = sum(left_links)
                total_right = sum(right_links)
                max_pairs = int(min(total_left, total_right))
                total_links = self.get_total_links(state, i)
                assert max_pairs <= self.C[i], f"Node {i} action exceeds capacity: {max_pairs} > {self.C[i]}"
                assert total_left >= max_pairs, f"Node {i} insufficient left links: {total_left} < {max_pairs}"
                assert total_right >= max_pairs, f"Node {i} insufficient right links: {total_right} < {max_pairs}"
                if max_pairs > 0:
                    for pairs in range(1, max_pairs + 1):
                        swap_options.append((i, pairs))
        
        action_space = [[]]
        for node, pairs in swap_options:
            action_space.append([(node, pairs)])
        
        node_to_pairs = defaultdict(list)
        for node, pairs in swap_options:
            node_to_pairs[node].append(pairs)
        
        nodes = sorted(node_to_pairs.keys())
        for r in range(2, len(nodes) + 1):
            for node_combo in itertools.combinations(nodes, r):
                pair_combinations = []
                for node in node_combo:
                    pair_combinations.append([(node, p) for p in node_to_pairs[node]])
                for combo in itertools.product(*pair_combinations):
                    action = list(combo)
                    action_space.append(action)
        
        return action_space
    
    def get_total_links(self, state, node):
        return sum(np.where(np.isinf(state[node, :]), 0, state[node, :]))
    
    def check_capacity(self, state):
        for i in range(self.N):
            total_links = self.get_total_links(state, i)
            if total_links > self.C[i] + 1e-10:
                print(f"__DEBUG__ assertion triggered for state:{state}")
            assert total_links <= self.C[i] + 1e-10, f"Node {i} exceeds capacity: {total_links} > {self.C[i]}"
    
    def split_action_into_layers(self, action):
        node_pairs = {node: pairs for node, pairs in action}
        max_layers = max(pairs for _, pairs in action) if action else 1
        layers = []
        
        for layer_idx in range(max_layers):
            current_layer = []
            for node, total_pairs in node_pairs.items():
                if layer_idx < total_pairs:
                    current_layer.append(node)
            if current_layer:
                layers.append(sorted(current_layer))
        return layers
    
    def attempt_swap(self, state, action):
        new_state = state.copy()
        layers = self.split_action_into_layers(action)
        
        for layer in layers:
            swap_results = []
            current_nodes = layer
            phase_idx = 0
            
            while current_nodes:
                phase_nodes = current_nodes[phase_idx::2]
                phase_changes = {}
                
                for node in phase_nodes:
                    left_connected = np.where(new_state[:node, node] != np.inf)[0]
                    right_connected = np.where(new_state[node, node+1:] != np.inf)[0]
                    
                    if len(left_connected) > 0 and len(right_connected) > 0:
                        node_pairs = sum(pairs for n, pairs in action if n == node) or 1
                        for _ in range(node_pairs):
                            max_distance = -1
                            best_pair = None
                            for left_node in left_connected:
                                for right_node in right_connected + node + 1:
                                    current_left = new_state[left_node, node]
                                    current_right = new_state[node, right_node]
                                    if current_left >= 1 and current_right >= 1:
                                        distance = right_node - left_node
                                        if distance > max_distance:
                                            max_distance = distance
                                            best_pair = (left_node, right_node)
                            
                            if best_pair:
                                left_node, right_node = best_pair
                                current_left = new_state[left_node, node]
                                current_right = new_state[node, right_node]
                                successes = 1 if np.random.random() < self.ps else 0
                                links_consumed = {
                                    (left_node, node): current_left - 1 if current_left > 1 else np.inf,
                                    (node, left_node): current_left - 1 if current_left > 1 else np.inf,
                                    (node, right_node): current_right - 1 if current_right > 1 else np.inf,
                                    (right_node, node): current_right - 1 if current_right > 1 else np.inf
                                }
                                links_created = {}
                                if successes > 0:
                                    left_total = self.get_total_links(new_state, left_node) - 1
                                    right_total = self.get_total_links(new_state, right_node) - 1
                                    current_new = new_state[left_node, right_node]
                                    new_links = 1 if np.isinf(current_new) else current_new + 1
                                    if (left_total + 1 <= self.C[left_node] and 
                                        right_total + 1 <= self.C[right_node]):
                                        links_created[(left_node, right_node)] = new_links
                                        links_created[(right_node, left_node)] = new_links
                                
                                swap_results.append({
                                    'node': node,
                                    'left_node': left_node,
                                    'right_node': right_node,
                                    'successes': successes,
                                    'links_consumed': links_consumed,
                                    'links_created': links_created
                                })
                                phase_changes.update(links_consumed)
                                if successes > 0:
                                    phase_changes.update(links_created)
                                new_state[left_node, node] = links_consumed[(left_node, node)]
                                new_state[node, left_node] = links_consumed[(node, left_node)]
                                new_state[node, right_node] = links_consumed[(node, right_node)]
                                new_state[right_node, node] = links_consumed[(right_node, node)]
                                if successes > 0:
                                    new_state[left_node, right_node] = links_created.get((left_node, right_node), new_state[left_node, right_node])
                                    new_state[right_node, left_node] = links_created.get((right_node, left_node), new_state[right_node, left_node])
                                left_connected = np.where(new_state[:node, node] != np.inf)[0]
                                right_connected = np.where(new_state[node, node+1:] != np.inf)[0]
                
                for (i, j), value in phase_changes.items():
                    if (i, j) not in [(left_node, node), (node, left_node), (node, right_node), (right_node, node)]:
                        new_state[i, j] = value
                
                current_nodes = [n for n in current_nodes if n not in phase_nodes]
                phase_idx = 0 if current_nodes else phase_idx + 1
            
            chain_success = all(result['successes'] > 0 for result in swap_results)
            new_state = state.copy()
            final_changes = {}
            
            for result in swap_results:
                final_changes.update(result['links_consumed'])
                if chain_success:
                    final_changes.update(result['links_created'])
            
            for (i, j), value in final_changes.items():
                new_state[i, j] = value
            
            self.check_capacity(new_state)
            state = new_state.copy()
        
        return new_state
    
    def attempt_elementary_links(self, state):
        new_state = state.copy()
        
        for i in range(self.N - 1):
            m_right = self.memory_allocation[i].get(i + 1, 0)
            m_left = self.memory_allocation[i + 1].get(i, 0)
            max_attempts = min(m_right, m_left)
            current_links = 0 if np.isinf(new_state[i, i + 1]) else int(new_state[i, i + 1])
            available_attempts = max_attempts - current_links
            if available_attempts <= 0:
                continue
            total_i = self.get_total_links(new_state, i)
            total_i_plus_1 = self.get_total_links(new_state, i + 1)
            available_i = self.C[i] - total_i
            available_i_plus_1 = self.C[i + 1] - total_i_plus_1
            max_possible = min(max(available_i, available_i_plus_1), available_attempts)
            if max_possible > 0:
                successes = sum(1 for _ in range(int(max_possible)) if np.random.random() < self.p[i])
                if successes > 0:
                    max_successes = min(self.C[i] - total_i, self.C[i + 1] - total_i_plus_1)
                    successes = min(successes, max_successes)
                    if successes > 0:
                        new_state[i, i + 1] = current_links + successes if current_links > 0 else successes
                        new_state[i + 1, i] = new_state[i, i + 1]
        
        self.check_capacity(new_state)
        return new_state
    
    def step(self, state, action):
        new_state = state.copy()
        action_space = self.get_action_space(state)
        
        # Check for dead-end state: action_space is [[]] and not initial state
        if len(action_space) == 1 and not np.array_equal(state, np.full((self.N, self.N), np.inf)):
            new_state = np.full((self.N, self.N), np.inf)
            # Preserve end-to-end links
            new_state[0, self.N-1] = state[0, self.N-1]
            new_state[self.N-1, 0] = state[self.N-1, 0]
            reward = -5  # Penalty for dead-end state
            print(f"__DEBUG__ Dead-end state detected, resetting all links except e2e")
        else:
            if action:
                new_state = self.attempt_swap(state, action)
            reward = -1  # Default reward, updated after elementary links if needed
        
        # Always attempt elementary links
        new_state = self.attempt_elementary_links(new_state)
        end_to_end_links = 0 if np.isinf(new_state[0, self.N-1]) else int(new_state[0, self.N-1])
        if end_to_end_links > 0:
            reward = 100 * end_to_end_links
        
        next_action_space = self.get_action_space(new_state)
        return new_state, next_action_space, reward, end_to_end_links
    
    def observe(self, state, action):
        possibilities = []
        
        if not action:
            outcomes_per_pair = []
            for i in range(self.N - 1):
                m_right = self.memory_allocation[i].get(i + 1, 0)
                m_left = self.memory_allocation[i + 1].get(i, 0)
                max_attempts = min(m_right, m_left)
                current_links = 0 if np.isinf(state[i, i + 1]) else int(state[i, i + 1])
                available_attempts = max_attempts - current_links
                if available_attempts <= 0:
                    outcomes_per_pair.append([(0, 1.0)])
                    continue
                total_i = self.get_total_links(state, i)
                total_i_plus_1 = self.get_total_links(state, i + 1)
                available_i = self.C[i] - total_i
                available_i_plus_1 = self.C[i + 1] - total_i_plus_1
                max_possible = min(max(available_i, available_i_plus_1), available_attempts)
                if max_possible <= 0:
                    outcomes_per_pair.append([(0, 1.0)])
                    continue
                outcomes = []
                for k in range(int(max_possible) + 1):
                    prob = binom.pmf(k, int(max_possible), self.p[i])
                    outcomes.append((k, prob))
                outcomes_per_pair.append(outcomes)
            
            for outcome_combo in itertools.product(*outcomes_per_pair):
                new_state = state.copy()
                prob = 1.0
                for i, (successes, pair_prob) in enumerate(outcome_combo):
                    if successes > 0:
                        total_i = self.get_total_links(new_state, i)
                        total_i_plus_1 = self.get_total_links(new_state, i + 1)
                        max_successes = min(self.C[i] - total_i, self.C[i + 1] - total_i_plus_1)
                        successes = min(successes, max_successes)
                        if successes > 0:
                            new_state[i, i + 1] = (state[i, i + 1] + successes if not np.isinf(state[i, i + 1])
                                                else successes)
                            new_state[i + 1, i] = new_state[i, i + 1]
                    prob *= pair_prob
                self.check_capacity(new_state)
                action_space = self.get_action_space(new_state)
                end_to_end_links = 0 if np.isinf(new_state[0, self.N-1]) else int(new_state[0, self.N-1])
                reward = 100 * end_to_end_links if end_to_end_links > 0 else -1
                possibilities.append((new_state, action_space, reward, prob))
        else:
            layers = self.split_action_into_layers(action)
            total_swaps = sum(pairs for node, pairs in action) * len(layers)
            outcome_combinations = list(itertools.product([0, 1], repeat=total_swaps))
            
            for outcome in outcome_combinations:
                new_state = state.copy()
                prob = 1.0
                layer_outcome_idx = 0
                
                for layer in layers:
                    swap_results = []
                    current_nodes = layer
                    phase_idx = 0
                    layer_prob = 1.0
                    
                    while current_nodes:
                        phase_nodes = current_nodes[phase_idx::2]
                        phase_changes = {}
                        
                        for node in phase_nodes:
                            left_connected = np.where(new_state[:node, node] != np.inf)[0]
                            right_connected = np.where(new_state[node, node+1:] != np.inf)[0]
                            
                            if len(left_connected) > 0 and len(right_connected) > 0:
                                node_pairs = sum(pairs for n, pairs in action if n == node) or 1
                                for _ in range(node_pairs):
                                    max_distance = -1
                                    best_pair = None
                                    for left_node in left_connected:
                                        for right_node in right_connected + node + 1:
                                            current_left = new_state[left_node, node]
                                            current_right = new_state[node, right_node]
                                            if current_left >= 1 and current_right >= 1:
                                                distance = right_node - left_node
                                                if distance > max_distance:
                                                    max_distance = distance
                                                    best_pair = (left_node, right_node)
                                    
                                    if best_pair:
                                        left_node, right_node = best_pair
                                        current_left = new_state[left_node, node]
                                        current_right = new_state[node, right_node]
                                        successes = outcome[layer_outcome_idx]
                                        print(f"__DEBUG__ +++++++++++++ len of outcome:{len(outcome)}, layer_outcome_idx:{layer_outcome_idx}")
                                        layer_outcome_idx += 1
                                        swap_prob = self.ps if successes else (1 - self.ps)
                                        layer_prob *= swap_prob
                                        
                                        links_consumed = {
                                            (left_node, node): current_left - 1 if current_left > 1 else np.inf,
                                            (node, left_node): current_left - 1 if current_left > 1 else np.inf,
                                            (node, right_node): current_right - 1 if current_right > 1 else np.inf,
                                            (right_node, node): current_right - 1 if current_right > 1 else np.inf
                                        }
                                        links_created = {}
                                        if successes > 0:
                                            left_total = self.get_total_links(new_state, left_node) - 1
                                            right_total = self.get_total_links(new_state, right_node) - 1
                                            current_new = new_state[left_node, right_node]
                                            new_links = 1 if np.isinf(current_new) else current_new + 1
                                            if (left_total + 1 <= self.C[left_node] and 
                                                right_total + 1 <= self.C[right_node]):
                                                links_created[(left_node, right_node)] = new_links
                                                links_created[(right_node, left_node)] = new_links
                                        
                                        swap_results.append({
                                            'node': node,
                                            'left_node': left_node,
                                            'right_node': right_node,
                                            'successes': successes,
                                            'links_consumed': links_consumed,
                                            'links_created': links_created
                                        })
                                        phase_changes.update(links_consumed)
                                        if successes > 0:
                                            phase_changes.update(links_created)
                                        new_state[left_node, node] = links_consumed[(left_node, node)]
                                        new_state[node, left_node] = links_consumed[(node, left_node)]
                                        new_state[node, right_node] = links_consumed[(node, right_node)]
                                        new_state[right_node, node] = links_consumed[(right_node, node)]
                                        if successes > 0:
                                            new_state[left_node, right_node] = links_created.get((left_node, right_node), new_state[left_node, right_node])
                                            new_state[right_node, left_node] = links_created.get((right_node, left_node), new_state[right_node, left_node])
                                        left_connected = np.where(new_state[:node, node] != np.inf)[0]
                                        right_connected = np.where(new_state[node, node+1:] != np.inf)[0]
                        
                        for (i, j), value in phase_changes.items():
                            if (i, j) not in [(left_node, node), (node, left_node), (node, right_node), (right_node, node)]:
                                new_state[i, j] = value
                        
                        current_nodes = [n for n in current_nodes if n not in phase_nodes]
                        phase_idx = 0 if current_nodes else phase_idx + 1
                    
                    chain_success = all(result['successes'] > 0 for result in swap_results)
                    new_state = state.copy()
                    final_changes = {}
                    
                    for result in swap_results:
                        final_changes.update(result['links_consumed'])
                        if chain_success:
                            final_changes.update(result['links_created'])
                    
                    for (i, j), value in final_changes.items():
                        new_state[i, j] = value
                    
                    self.check_capacity(new_state)
                    state = new_state.copy()
                    prob *= layer_prob
                
                new_state = self.attempt_elementary_links(new_state)
                action_space = self.get_action_space(new_state)
                end_to_end_links = 0 if np.isinf(new_state[0, self.N-1]) else int(new_state[0, self.N-1])
                reward = 100 * end_to_end_links if end_to_end_links > 0 else -1
                possibilities.append((new_state, action_space, reward, prob))
        
        return possibilities

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=0.5, epsilon_min=0.1, episodes=100):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.q_table = defaultdict(lambda: defaultdict(float))
        initial_state = np.full((env.N, env.N), np.inf)
        self.q_table[tuple(map(tuple, initial_state))][tuple()] = 0.0
    
    def get_action(self, state, action_space, epsilon):
        state_tuple = tuple(map(tuple, state))
        if np.random.random() < epsilon:
            return random.choice(action_space) if action_space else []
        else:
            q_values = self.q_table[state_tuple]
            if not q_values:
                return random.choice(action_space) if action_space else []
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state, next_action_space):
        state_tuple = tuple(map(tuple, state))
        action_tuple = tuple((node, pairs) for node, pairs in action)
        next_state_tuple = tuple(map(tuple, next_state))
        
        future_q = 0
        if next_action_space:
            future_q = max([self.q_table[next_state_tuple].get(tuple((n, p) for n, p in a), 0.0) 
                            for a in next_action_space])
        
        current_q = self.q_table[state_tuple][action_tuple]
        self.q_table[state_tuple][action_tuple] = (
            current_q + self.alpha * (reward + self.gamma * future_q - current_q)
        )
    
    def observe(self, state, action):
        possibilities = self.env.observe(state, action)
        state_tuple = tuple(map(tuple, state))
        
        for next_state, action_space, _, _ in possibilities:
            next_state_tuple = tuple(map(tuple, next_state))
            if next_state_tuple not in self.q_table:
                for a in action_space:
                    a_tuple = tuple((node, pairs) for node, pairs in a)
                    self.q_table[next_state_tuple][a_tuple] = 0.0

def train_agent(env, episodes, alpha, gamma):
    agent = QLearningAgent(env, alpha, gamma, episodes=episodes)
    episode_rewards = []
    
    e = min(env.C[0], env.C[env.N - 1])
    target_e2e = max(1, int(0.8 * e))
    max_steps = 100 * e
    max_failed_attempts = 20 * e
    extra_steps_limit = 10 * e
    
    for episode in range(episodes):
        epsilon = max(agent.epsilon_min, agent.epsilon_start * (1 - episode / episodes))
        print(f" ############__DEBUG__#################################################Episode number:{episode}, epsilon:{epsilon:.3f}################################################")
        state = np.full((env.N, env.N), np.inf)
        action_space = env.get_action_space(state)
        total_reward = 0
        steps = 0
        failed_attempts = 0
        current_e2e = 0
        extra_steps = 0
        target_reached = False
        
        while True:
            print(f": ############__DEBUG__########Step number{steps}###################################{episode}#################")
            print(f"__DEBUG__ The state for this step:{state}")
            
            action = agent.get_action(state, action_space, epsilon)
            print(f"__DEBUG__ The action for this step__________________________:{action} OUT OF ---action_space:{action_space}\n\n")
            agent.observe(state, action)
            next_state, next_action_space, reward, end_to_end_links = env.step(state, action)
            agent.update(state, action, reward, next_state, next_action_space)
            
            state = next_state
            action_space = next_action_space
            total_reward += reward
            steps += 1
            
            if end_to_end_links > current_e2e:
                failed_attempts = 0
                current_e2e = end_to_end_links
            else:
                failed_attempts += 1
            
            if not target_reached and current_e2e >= target_e2e:
                target_reached = True
                print(f"Episode {episode}: Reached target e2e links ({current_e2e}/{target_e2e}) at step {steps}")
            
            if target_reached:
                extra_steps += 1
            
            if current_e2e >= e:
                print(f"Episode {episode} terminated: Achieved max e2e links ({current_e2e}/{e}) at step {steps}")
                break
            if target_reached and extra_steps >= extra_steps_limit:
                print(f"Episode {episode} terminated: Extra steps limit reached ({extra_steps}/{extra_steps_limit}) after target e2e ({current_e2e}/{target_e2e})")
                break
            if failed_attempts >= max_failed_attempts:
                print(f"Episode {episode} terminated: Max failed attempts reached ({failed_attempts}/{max_failed_attempts})")
                break
            if steps >= max_steps:
                print(f"Episode {episode} terminated: Max steps reached ({steps}/{max_steps}")
                break
        
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode} completed - Total Reward: {total_reward}, e2e Links: {current_e2e}, Extra Steps: {extra_steps}")
    
    policy = []
    for state_tuple in agent.q_table:
        state = np.array(state_tuple)
        action_space = env.get_action_space(state)
        q_values = [agent.q_table[state_tuple].get(tuple((n, p) for n, p in a), 0.0) for a in action_space]
        if q_values:
            best_action_idx = np.argmax(q_values)
            one_hot = [0] * len(action_space)
            one_hot[best_action_idx] = 1
            policy.append({
                'state': state,
                'action_space': action_space,
                'policy': one_hot
            })
    
    return policy, episode_rewards

def main():
    N = 4
    p = [0.7, 0.7, 0.7]
    ps = 0.5
    C = [4,4,4,4]
    episodes = 100
    alpha = 0.05
    gamma = 0.95
    
    env = QuantumNetworkEnv(p, ps, C)
    os.makedirs("training_data", exist_ok=True)
    p_str = "_".join(map(str, p))
    c_str = "_".join(map(str, C))
    filename = f"training_data/policy_N{N}_p{p_str}_ps{ps}_C{c_str}_a{alpha}_g{gamma}_e0.5-0.1"
    policy_file = f"{filename}.pkl"
    rewards_file = f"{filename}_rewards.pkl"
    
    policy, episode_rewards = train_agent(env, episodes, alpha, gamma)
    
    with open(policy_file, 'wb') as f:
        pickle.dump(policy, f)
    with open(rewards_file, 'wb') as f:
        pickle.dump(episode_rewards, f)
    
    print(f"Training completed.")
    print(f"Policy saved as: {policy_file}")
    print(f"Rewards saved as: {rewards_file}")

if __name__ == "__main__":
    main()