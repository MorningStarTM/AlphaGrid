"""
Visual test for MCTS on Grid2Op.

Runs MCTS with a random policy on l2rpn_case14_sandbox,
then renders the search tree as a graph using networkx + matplotlib.

Node colors:
  - Blue:   safe state (rho_max < 0.98)
  - Orange: critical state (rho_max >= 0.98)
  - Red:    terminal / blackout
  - Green:  recovery node (safe + skipped >= 10 steps)

Edge labels show: action index
Node labels show: visit count, Q-value, rho_max
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import grid2op
from grid2op.Action import PlayableAction

from alphagrid.mcts import MCTS, Node


# ─── Minimal env wrapper (just what MCTS needs) ───

class SimpleGridEnv:
    def __init__(self, env_name="l2rpn_case14_sandbox", max_actions=50):
        self.env = grid2op.make(env_name, action_class=PlayableAction)
        do_nothing = self.env.action_space({})
        all_topos = self.env.action_space.get_all_unitary_topologies_set(
            self.env.action_space
        )
        self.reduced_actions = [do_nothing] + all_topos[:max_actions - 1]
        self.action_size = len(self.reduced_actions)

    def reset(self):
        return self.env.reset()

    def step(self, action_idx):
        return self.env.step(self.reduced_actions[action_idx])


# ─── Random policy (stand-in until real network exists) ───

def random_policy(obs, action_size):
    """Uniform random policy over all actions."""
    priors = {i: 1.0 / action_size for i in range(action_size)}
    value = 0.5
    return priors, value


# ─── Tree → NetworkX graph ───

def tree_to_graph(root: Node, max_nodes=200):
    """Convert MCTS tree to a networkx DiGraph for visualization."""
    G = nx.DiGraph()
    queue = [(root, "root")]
    node_id = 0
    node_map = {id(root): "root"}
    node_attrs = {}

    # root attributes
    rho_max = float(root.obs.rho.max()) if root.obs is not None else 0.0
    node_attrs["root"] = {
        "visit_count": root.visit_count,
        "q_value": round(root.q_value, 3),
        "rho_max": round(rho_max, 3),
        "done": root.done,
        "is_recovery": root.is_recovery,
        "steps_skipped": root.steps_skipped,
        "is_safe": rho_max < 0.98,
    }
    G.add_node("root")

    while queue:
        node, parent_id = queue.pop(0)
        for child in node.children:
            if child.visit_count == 0:
                continue
            node_id += 1
            if node_id > max_nodes:
                break

            child_id = f"n{node_id}"
            node_map[id(child)] = child_id

            rho_max = float(child.obs.rho.max()) if child.obs is not None else 0.0
            node_attrs[child_id] = {
                "visit_count": child.visit_count,
                "q_value": round(child.q_value, 3),
                "rho_max": round(rho_max, 3),
                "done": child.done,
                "is_recovery": child.is_recovery,
                "steps_skipped": child.steps_skipped,
                "is_safe": rho_max < 0.98,
            }

            G.add_node(child_id)
            G.add_edge(parent_id, child_id, action=child.action_index)
            queue.append((child, child_id))

    nx.set_node_attributes(G, node_attrs)
    return G


def get_node_color(attrs):
    if attrs.get("done", False):
        return "#ef4444"    # red — terminal/blackout
    if attrs.get("is_recovery", False):
        return "#22c55e"    # green — recovery
    if attrs.get("is_safe", False):
        return "#3b82f6"    # blue — safe
    return "#f59e0b"        # orange — critical


def visualize_tree(G, title="MCTS Search Tree", save_path=None):
    """Draw the MCTS tree with colored nodes and labels."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if _has_graphviz() else _hierarchy_layout(G, "root", width=8.0, vert_gap=0.25)

    colors = [get_node_color(G.nodes[n]) for n in G.nodes()]
    sizes = [max(300, G.nodes[n].get("visit_count", 1) * 30) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", arrows=True,
                           arrowsize=12, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="#333333", linewidths=1.0)

    # node labels: visit_count / Q / rho_max
    labels = {}
    for n in G.nodes():
        attrs = G.nodes[n]
        vc = attrs.get("visit_count", 0)
        q = attrs.get("q_value", 0)
        rho = attrs.get("rho_max", 0)
        skip = attrs.get("steps_skipped", 0)
        label = f"N={vc}\nQ={q:.2f}\nρ={rho:.2f}"
        if skip > 0:
            label += f"\nskip={skip}"
        labels[n] = label

    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    # edge labels: action index
    edge_labels = {(u, v): f"a{d['action']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5, ax=ax)

    # legend
    legend_patches = [
        mpatches.Patch(color="#3b82f6", label="Safe (ρ < 0.98)"),
        mpatches.Patch(color="#f59e0b", label="Critical (ρ ≥ 0.98)"),
        mpatches.Patch(color="#ef4444", label="Terminal / Blackout"),
        mpatches.Patch(color="#22c55e", label="Recovery (skipped ≥ 10)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved tree visualization to {save_path}")

    plt.show()


def _has_graphviz():
    try:
        import pygraphviz
        return True
    except ImportError:
        return False


def _hierarchy_layout(G, root, width=1.0, vert_gap=0.2, xcenter=0.5):
    """Simple top-down tree layout without graphviz dependency."""
    pos = {}

    def _recurse(node, left, right, depth=0):
        pos[node] = ((left + right) / 2, -depth * vert_gap)
        children = list(G.successors(node))
        if not children:
            return
        dx = (right - left) / len(children)
        for i, child in enumerate(children):
            _recurse(child, left + i * dx, left + (i + 1) * dx, depth + 1)

    _recurse(root, 0, width)
    return pos


# ─── Main test ───

def main():
    print("Setting up Grid2Op environment...")
    env = SimpleGridEnv(env_name="l2rpn_case14_sandbox", max_actions=10)
    print(f"Action space: {env.action_size} actions (1 do-nothing + {env.action_size - 1} topology)")

    obs = env.reset()
    print(f"Initial rho_max: {obs.rho.max():.4f}")
    # Start from the initial safe reset state so obs.simulate() works for nested
    # depth-2 and depth-3 calls. Advancing to congestion (rho >= 0.90) causes most
    # topology actions to immediately blackout, collapsing the tree to depth-2 only.

    # run MCTS
    policy_fn = lambda o: random_policy(o, env.action_size)
    mcts = MCTS(
        env=env,
        policy_fn=policy_fn,
        config={
            "num_simulations": 300,
            "c_puct": 1.41,
            "gamma": 0.99,
            "safe_skip_steps": 0,       # must be 0: forwarded sim-obs can't be re-simulated
            "recovery_threshold": 999,  # no early stopping — let tree reach depth 3+
            "use_heuristic_value": True,
        },
    )

    print(f"\nRunning MCTS with {mcts.config['num_simulations']} simulations...")
    action_probs, root, stats = mcts.search(obs)

    print(f"\n{'='*50}")
    print(f"MCTS Search Results")
    print(f"{'='*50}")
    print(f"Simulations run:  {stats['simulations']}")
    print(f"Recovery nodes:   {stats['recovery_nodes']}")
    print(f"Max tree depth:   {stats['max_depth']}")
    print(f"Early stopped:    {stats['early_stopped']}")

    top_k = 5
    top_actions = np.argsort(action_probs)[::-1][:top_k]
    print(f"\nTop {top_k} actions by visit count:")
    for i, a in enumerate(top_actions):
        if action_probs[a] > 0:
            print(f"  #{i+1}: action {a}, prob = {action_probs[a]:.3f}")

    # visualize
    print("\nBuilding tree graph...")
    G = tree_to_graph(root, max_nodes=500)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    save_path = os.path.join(os.path.dirname(__file__), "..", "mcts_tree.png")
    visualize_tree(
        G,
        title=f"MCTS Tree — {stats['simulations']} sims, ρ_max={obs.rho.max():.3f}, depth={stats['max_depth']}",
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
