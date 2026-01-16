"""MDP (Markov Decision Process) for Inventory Optimization."""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from scipy import sparse

logger = logging.getLogger(__name__)


class MDPOptimizer:
    """
    Markov Decision Process optimizer for inventory and pricing decisions.

    Maximizes expected rewards considering:
    - Inventory holding costs
    - Stock-out penalties
    - Order costs
    - Sales revenue
    """

    def __init__(
        self,
        inventory_bins: List[int] = None,
        demand_states: List[str] = None,
        order_quantities: List[int] = None,
        rewards: Dict[str, float] = None,
        discount_factor: float = 0.95,
        solver: str = "value_iteration",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize MDP optimizer.

        Args:
            inventory_bins: Inventory level bins (e.g., [0, 10, 25, 50, 100, 500])
            demand_states: Demand state labels
            order_quantities: Possible order quantities
            rewards: Dictionary with reward structure
            discount_factor: Discount factor (gamma)
            solver: 'value_iteration' or 'policy_iteration'
            max_iterations: Max iterations for convergence
            tolerance: Convergence tolerance
        """
        self.inventory_bins = inventory_bins or [0, 10, 25, 50, 100, 500]
        self.demand_states = demand_states or ["low", "medium", "high"]
        self.order_quantities = order_quantities or [0, 50, 100, 200, 500]
        self.discount_factor = discount_factor
        self.solver_method = solver
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Reward structure
        default_rewards = {
            "stock_out_penalty": -100,
            "holding_cost_per_unit": -0.5,
            "order_cost_per_unit": -0.1,
            "sales_reward_per_unit": 5.0,
        }
        self.rewards = {**default_rewards, **(rewards or {})}

        # State and action spaces
        self.num_inventory_states = len(self.inventory_bins)
        self.num_demand_states = len(self.demand_states)
        self.num_actions = len(self.order_quantities)
        self.num_states = self.num_inventory_states * self.num_demand_states

        # Value function and policy
        self.value_function = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

        # Transition probabilities (to be estimated from data)
        self.transition_matrix = None
        self.demand_probabilities = None

        logger.info(
            f"Initialized MDP with {self.num_states} states, "
            f"{self.num_actions} actions"
        )

    def fit_transition_probabilities(self, demand_history: np.ndarray):
        """
        Estimate transition probabilities from historical demand.

        Args:
            demand_history: Array of historical demand values
        """
        logger.info("Fitting transition probabilities...")

        # Discretize demand into states
        demand_discrete = self._discretize_demand(demand_history)

        # Estimate demand state distribution
        unique, counts = np.unique(demand_discrete, return_counts=True)
        self.demand_probabilities = np.zeros(self.num_demand_states)
        self.demand_probabilities[unique] = counts / len(demand_discrete)

        # Initialize transition matrix (simplified)
        # P(next_inventory | current_inventory, action, demand)
        self.transition_matrix = self._estimate_transition_matrix(demand_history)

        logger.info("Transition probabilities fitted")

    def solve(self):
        """Solve MDP using specified method."""
        if self.solver_method == "value_iteration":
            self._value_iteration()
        elif self.solver_method == "policy_iteration":
            self._policy_iteration()
        else:
            raise ValueError(f"Unknown solver: {self.solver_method}")

        logger.info(f"MDP solved using {self.solver_method}")

    def _value_iteration(self):
        """Solve MDP using value iteration algorithm."""
        logger.info(f"Starting value iteration (max_iters={self.max_iterations})...")

        for iteration in range(self.max_iterations):
            old_value = self.value_function.copy()

            # Bellman update for each state
            for state in range(self.num_states):
                inv_state, demand_state = self._decode_state(state)

                # Try all actions
                action_values = []
                for action in range(self.num_actions):
                    # Expected immediate reward
                    immediate_reward = self._compute_reward(inv_state, action, demand_state)

                    # Expected future value
                    future_value = self._compute_expected_future_value(
                        state, action, inv_state, demand_state
                    )

                    action_value = immediate_reward + self.discount_factor * future_value
                    action_values.append(action_value)

                # Update value function and policy
                best_action = np.argmax(action_values)
                self.value_function[state] = action_values[best_action]
                self.policy[state] = best_action

            # Check convergence
            max_change = np.max(np.abs(self.value_function - old_value))
            if max_change < self.tolerance:
                logger.info(f"Converged at iteration {iteration+1}")
                break

            if (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration+1}, max_change={max_change:.6f}")

    def _policy_iteration(self):
        """Solve MDP using policy iteration algorithm."""
        logger.info("Starting policy iteration...")

        for iteration in range(self.max_iterations):
            # Policy evaluation
            self._evaluate_policy()

            # Policy improvement
            policy_changed = self._improve_policy()

            if not policy_changed:
                logger.info(f"Policy converged at iteration {iteration+1}")
                break

            if (iteration + 1) % 10 == 0:
                logger.info(f"Policy iteration {iteration+1}")

    def _evaluate_policy(self):
        """Evaluate current policy (solve system of linear equations)."""
        # For each state, solve: V(s) = R(s,π(s)) + γ * Σ P(s'|s,π(s)) * V(s')
        for state in range(self.num_states):
            inv_state, demand_state = self._decode_state(state)
            action = self.policy[state]

            immediate_reward = self._compute_reward(inv_state, action, demand_state)
            future_value = self._compute_expected_future_value(
                state, action, inv_state, demand_state
            )

            self.value_function[state] = immediate_reward + self.discount_factor * future_value

    def _improve_policy(self) -> bool:
        """Improve policy, return whether it changed."""
        policy_changed = False

        for state in range(self.num_states):
            inv_state, demand_state = self._decode_state(state)
            old_action = self.policy[state]

            # Find best action
            best_value = -np.inf
            best_action = 0

            for action in range(self.num_actions):
                immediate_reward = self._compute_reward(inv_state, action, demand_state)
                future_value = self._compute_expected_future_value(
                    state, action, inv_state, demand_state
                )
                action_value = immediate_reward + self.discount_factor * future_value

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            self.policy[state] = best_action
            if best_action != old_action:
                policy_changed = True

        return policy_changed

    def get_optimal_order(
        self, current_inventory: int, predicted_demand_state: str
    ) -> int:
        """
        Get optimal order quantity for current state.

        Args:
            current_inventory: Current inventory level
            predicted_demand_state: Predicted demand state

        Returns:
            Optimal order quantity
        """
        if self.transition_matrix is None:
            raise ValueError("MDP not solved yet. Call fit_transition_probabilities and solve first.")

        # Discretize inventory
        inv_bin = np.searchsorted(self.inventory_bins, current_inventory, side="right") - 1
        inv_bin = np.clip(inv_bin, 0, self.num_inventory_states - 1)

        # Get demand state index
        demand_idx = self.demand_states.index(predicted_demand_state)

        # Get state and optimal action
        state = self._encode_state(inv_bin, demand_idx)
        action = self.policy[state]

        return self.order_quantities[action]

    def get_optimal_policy(self) -> Dict[str, int]:
        """Get complete optimal policy."""
        policy_dict = {}

        for state in range(self.num_states):
            inv_bin, demand_state = self._decode_state(state)
            action = self.policy[state]
            inv_level = self.inventory_bins[inv_bin]
            demand_label = self.demand_states[demand_state]

            policy_dict[f"inv_{inv_level}_demand_{demand_label}"] = self.order_quantities[action]

        return policy_dict

    # ========== Helper Methods ==========

    def _compute_reward(self, inventory: int, action: int, demand_state: int) -> float:
        """Compute immediate reward for state-action pair."""
        order_qty = self.order_quantities[action]

        # Order cost
        reward = -order_qty * self.rewards["order_cost_per_unit"]

        # Inventory holding cost
        new_inventory = inventory + order_qty
        reward += -new_inventory * self.rewards["holding_cost_per_unit"]

        # Sales revenue (based on demand state)
        if demand_state == 0:  # Low
            sales = 0.3 * new_inventory
        elif demand_state == 1:  # Medium
            sales = 0.6 * new_inventory
        else:  # High
            sales = min(1.0, new_inventory) * new_inventory

        reward += min(sales, new_inventory) * self.rewards["sales_reward_per_unit"]

        # Stock-out penalty
        if new_inventory == 0:
            reward += self.rewards["stock_out_penalty"]

        return reward

    def _compute_expected_future_value(
        self, state: int, action: int, inv_state: int, demand_state: int
    ) -> float:
        """Compute expected future value."""
        if self.demand_probabilities is None:
            return 0.0

        expected_value = 0.0

        for next_demand_state in range(self.num_demand_states):
            prob = self.demand_probabilities[next_demand_state]

            # Transition to next inventory state (simplified)
            next_inv_state = self._compute_next_inventory_state(inv_state, action, next_demand_state)
            next_state = self._encode_state(next_inv_state, next_demand_state)

            expected_value += prob * self.value_function[next_state]

        return expected_value

    def _compute_next_inventory_state(
        self, inv_state: int, action: int, demand_state: int
    ) -> int:
        """Compute next inventory state given current state and action."""
        current_inv = self.inventory_bins[inv_state]
        order_qty = self.order_quantities[action]
        new_inv = current_inv + order_qty

        # Demand reduces inventory
        if demand_state == 0:
            demand = int(0.3 * new_inv)
        elif demand_state == 1:
            demand = int(0.6 * new_inv)
        else:
            demand = min(new_inv, int(1.2 * new_inv))

        final_inv = max(0, new_inv - demand)

        # Discretize to bin
        next_inv_state = np.searchsorted(self.inventory_bins, final_inv, side="right") - 1
        return np.clip(next_inv_state, 0, self.num_inventory_states - 1)

    def _discretize_demand(self, demand: np.ndarray) -> np.ndarray:
        """Discretize demand into states."""
        low_threshold = np.percentile(demand, 33)
        high_threshold = np.percentile(demand, 67)

        discrete = np.zeros_like(demand)
        discrete[demand <= low_threshold] = 0
        discrete[(demand > low_threshold) & (demand <= high_threshold)] = 1
        discrete[demand > high_threshold] = 2

        return discrete.astype(int)

    def _estimate_transition_matrix(self, demand_history: np.ndarray) -> np.ndarray:
        """Estimate state transition matrix from data."""
        # Simplified: return identity matrix
        return np.eye(self.num_states)

    def _encode_state(self, inv_state: int, demand_state: int) -> int:
        """Encode (inventory_state, demand_state) into single state index."""
        return inv_state * self.num_demand_states + demand_state

    def _decode_state(self, state: int) -> Tuple[int, int]:
        """Decode state index into (inventory_state, demand_state)."""
        inv_state = state // self.num_demand_states
        demand_state = state % self.num_demand_states
        return inv_state, demand_state
