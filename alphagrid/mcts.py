import math
import numpy as np
from grid2op.Observation import CompleteObservation


class Node:
    """
    MCTS node for Grid2Op topology optimization.

    Each node stores a full grid2op observation object — this is required
    because obs.simulate(action) needs the exact obs to run power flow.

    Contrast with game MCTS where state is a cheap array copy:
    here each node holds ~KB of grid state (line loads, topology vectors,
    generator/load states, forecasts, cooldowns, etc).
    """

    def __init__(self, obs, reward=0.0, done=False, parent=None,
                 action_index=-1, prior=0.0):
        self.obs = obs                  # grid2op CompleteObservation (or None if lazy)
        self.reward = reward
        self.done = done
        self.parent = parent
        self.action_index = action_index
        self.prior = prior

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        # grid-specific: tracks safe-state skipping
        self.steps_skipped = 0
        self.is_recovery = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def depth(self):
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct):
        """Select child with highest UCB (PUCT variant)."""
        best_child = None
        best_ucb = -float("inf")

        for child in self.children:
            ucb = self._puct(child, c_puct)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def _puct(self, child, c_puct):
        q = child.q_value
        exploration = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return q + exploration

    def expand(self, action_priors: dict):
        """
        Create child nodes for actions with non-zero prior.

        action_priors: {action_index: prior_probability}

        Children are created with obs=None (lazy). The actual
        obs.simulate() call happens in simulate_if_needed() when
        a child is first selected — avoids wasting power flow
        solves on branches we never visit.
        """
        for action_idx, prior in action_priors.items():
            child = Node(
                obs=None,
                parent=self,
                action_index=action_idx,
                prior=prior,
            )
            self.children.append(child)

    def simulate_if_needed(self, env):
        """
        Lazily run obs.simulate() from parent's observation.

        This is the expensive step — a full AC power flow solve.
        Returns True if simulation succeeded.
        """
        if self.obs is not None:
            return True

        action = env.reduced_actions[self.action_index]
        try:
            sim_obs, sim_reward, sim_done, sim_info = self.parent.obs.simulate(action)
        except Exception:
            self.done = True
            self.obs = self.parent.obs
            self.reward = 0.0
            return False

        if sim_info.get("is_illegal", False) or sim_info.get("is_ambiguous", False):
            self.done = True
            self.obs = self.parent.obs
            self.reward = 0.0
            return False

        self.obs = sim_obs
        self.done = sim_done
        self.reward = self._compute_reward(sim_obs, sim_done)
        return True

    def _compute_reward(self, obs, done):
        if done:
            return 0.0
        rho = obs.rho
        rho_max = rho.max()
        n_offline = int(np.sum(~obs.line_status))
        if rho_max <= 1.0:
            u = max(rho_max - 0.5, 0.0)
        else:
            u = float(np.sum(rho[rho > 1.0] - 0.5))
        return float(np.exp(-u - 0.5 * n_offline))

    def backpropagate(self, value, gamma):
        """Propagate discounted value up to root."""
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(gamma * value + self.reward, gamma)

    def max_reachable_steps(self):
        """Max survivable steps from this node downward."""
        if not self.children:
            return self.steps_skipped
        visited = [c for c in self.children if c.visit_count > 0]
        if not visited:
            return self.steps_skipped
        return self.steps_skipped + max(c.max_reachable_steps() for c in visited)

    def count_recovery_nodes(self):
        count = 1 if self.is_recovery else 0
        for child in self.children:
            count += child.count_recovery_nodes()
        return count


class MCTS:
    """
    Monte Carlo Tree Search for Grid2Op topology optimization.

    Adapted from standard AlphaZero MCTS with these grid-specific changes:
    - obs.simulate() replaces game.get_next_state()
    - Lazy child simulation (power flow is expensive)
    - Safe-state skipping through stable grid states
    - Heuristic value function (no learned value net needed)
    - Early stopping when enough recovery nodes found
    """

    def __init__(self, env, policy_fn, config):
        """
        Args:
            env: GridEnv wrapper with reduced_actions and simulate()
            policy_fn: callable(obs) -> (action_priors: dict, value: float)
                       action_priors = {action_idx: probability}
            config: dict with MCTS hyperparameters
        """
        self.env = env
        self.policy_fn = policy_fn
        self.config = {
            "num_simulations": 200,
            "c_puct": 1.41,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "gamma": 0.99,
            "temperature": 1.0,
            "thermal_limit": 0.98,
            "safe_skip_steps": 10,
            "recovery_threshold": 6,
            "use_heuristic_value": True,
        }
        self.config.update(config)

    def search(self, obs):
        """
        Run MCTS from the given grid observation.

        Returns:
            action_probs: np.array of shape [action_size] — visit count distribution
            root: the root Node (for visualization / inspection)
            stats: dict with search statistics
        """
        root = Node(obs=obs, reward=0.0, done=False)
        root.visit_count = 1

        # get policy prior for root + add dirichlet noise
        action_priors, root_value = self.policy_fn(obs)
        action_priors = self._add_noise(action_priors)
        root.expand(action_priors)

        stats = {"simulations": 0, "recovery_nodes": 0, "early_stopped": False, "max_depth": 0}

        for _ in range(self.config["num_simulations"]):

            # === SELECTION: walk down tree by PUCT ===
            node = root
            while node.is_expanded():
                node = node.select_child(self.config["c_puct"])

            # === EXPANSION: simulate this node's action ===
            valid = node.simulate_if_needed(self.env)

            if not valid or node.done:
                node.backpropagate(0.0, self.config["gamma"])
                stats["simulations"] += 1
                continue

            # safe-state skip
            self._safe_state_skip(node)

            # === EVALUATION: get value for this leaf ===
            if self.config["use_heuristic_value"]:
                value = self._heuristic_value(node.obs, node.done)
            else:
                _, value = self.policy_fn(node.obs)

            # expand leaf for future iterations
            if not node.done:
                child_priors, _ = self.policy_fn(node.obs)
                node.expand(child_priors)

            # === BACKPROPAGATION ===
            node.backpropagate(value, self.config["gamma"])

            stats["simulations"] += 1
            stats["max_depth"] = max(stats["max_depth"], node.depth)

            # early stopping
            recovery_count = root.count_recovery_nodes()
            stats["recovery_nodes"] = recovery_count
            if recovery_count >= self.config["recovery_threshold"]:
                stats["early_stopped"] = True
                break

        # build action probability distribution from visit counts
        action_size = len(self.env.reduced_actions)
        action_probs = np.zeros(action_size, dtype=np.float32)
        for child in root.children:
            if child.visit_count > 0:
                action_probs[child.action_index] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs, root, stats

    def _safe_state_skip(self, node):
        """
        Fast-forward through safe grid states via do-nothing simulation.

        Note: obs.simulate() requires forecast data. Simulated observations
        from deeper tree nodes may not carry forecasts, so we catch
        NoForecastAvailable and stop skipping.
        """
        if node.obs.rho.max() >= self.config["thermal_limit"]:
            return

        current_obs = node.obs
        skipped = 0
        do_nothing = self.env.reduced_actions[0]  # action 0 = do-nothing

        while skipped < self.config["safe_skip_steps"]:
            try:
                sim_obs, _, sim_done, _ = current_obs.simulate(do_nothing)
            except Exception:
                break
            skipped += 1
            if sim_done:
                node.done = True
                break
            current_obs = sim_obs
            if current_obs.rho.max() >= self.config["thermal_limit"]:
                break

        node.obs = current_obs
        node.steps_skipped = skipped
        if skipped >= self.config["safe_skip_steps"]:
            node.is_recovery = True

    def _heuristic_value(self, obs, done):
        if done:
            return 0.0
        rho_max = obs.rho.max()
        n_offline = int(np.sum(~obs.line_status))
        if rho_max <= 1.0:
            u = max(rho_max - 0.5, 0.0)
        else:
            u = float(np.sum(obs.rho[obs.rho > 1.0] - 0.5))
        r = float(np.exp(-u - 0.5 * n_offline))
        gamma = self.config["gamma"]
        return r / (1.0 - gamma)

    def _add_noise(self, action_priors):
        """Add Dirichlet noise to root priors for exploration."""
        actions = list(action_priors.keys())
        priors = np.array([action_priors[a] for a in actions])
        noise = np.random.dirichlet([self.config["dirichlet_alpha"]] * len(actions))
        eps = self.config["dirichlet_epsilon"]
        noisy_priors = (1 - eps) * priors + eps * noise
        return {a: float(p) for a, p in zip(actions, noisy_priors)}
