import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Constants for states
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

def distance_periodic(pos1, pos2, L):
    """Calculates Euclidean distance with periodic boundary conditions."""
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > L / 2, L - delta, delta)
    return np.sqrt((delta**2).sum())

class Individual:
    def __init__(self, id, L, is_superspreader=False):
        self.id = id
        self.pos = np.random.rand(2) * L  # Random position in L x L space
        self.state = SUSCEPTIBLE
        self.is_superspreader = is_superspreader
        self.infected_by = -1 # For tracking infection path (optional)
        self.infected_others_count = 0 # For secondary infection stats (optional)

def get_infection_probability(r, r0_eff, w0_eff, alpha_eff):
    """
    Calculates infection probability w(r).
    r: distance
    r0_eff: effective cutoff distance
    w0_eff: effective base probability (or constant probability if alpha_eff=0)
    alpha_eff: exponent for decay
    """
    if r == 0: # Cannot infect self
        return 0.0
    if r <= r0_eff:
        if alpha_eff == 0: # Constant probability (e.g., strong infectiousness superspreader)
            return w0_eff
        else:
            return w0_eff * (1 - r / r0_eff)**alpha_eff
    return 0.0

def run_sir_simulation(
    N=200, L=30.0, 
    r0_normal=3.0,                  
    w0_base=1.0, 
    alpha_normal=2.0, 
    gamma_recovery_prob=1.0, 
    lambda_ss_fraction=0.1, 
    superspreader_model_type="hub",  
    initial_infected_count=1,
    max_time_steps=100,
    verbose=False
):
    """
    Runs the SIR simulation with superspreaders.

    Args:
        N (int): Total number of individuals.
        L (float): Size of the square simulation area (L x L).
        r0_normal (float): Infection cutoff distance for normal individuals.
        w0_base (float): Base infection probability factor (w0 in paper).
        alpha_normal (float): Exponent for normal infection probability decay.
        gamma_recovery_prob (float): Probability of an infected individual recovering in a time step.
        lambda_ss_fraction (float): Fraction of the population that are superspreaders.
        superspreader_model_type (str): "strong" or "hub".
        initial_infected_count (int): Number of initially infected individuals.
        max_time_steps (int): Maximum number of simulation steps.
        verbose (bool): If True, print step-by-step info.

    Returns:
        tuple: (S_counts, I_counts, R_counts, time_points)
               Lists of S, I, R counts and corresponding time points.
    """

    individuals = []
    num_superspreaders = int(N * lambda_ss_fraction)
    superspreader_indices = random.sample(range(N), num_superspreaders)

    for i in range(N):
        is_ss = (i in superspreader_indices)
        individuals.append(Individual(i, L, is_superspreader=is_ss))

    # Initial infection
    infected_indices = random.sample(range(N), initial_infected_count)
    for idx in infected_indices:
        individuals[idx].state = INFECTED
        # Paper: "An initial-infected individual is placed on the bottom of the system"
        # For simplicity, we can just pick randomly or first ones.
        # If specific placement is needed:
        # individuals[idx].pos = np.array([L/2, r0_normal * 0.1]) # Example: bottom center-ish

    # Initialize running counts
    current_S = N - initial_infected_count
    current_I = initial_infected_count
    current_R = 0

    S_counts, I_counts, R_counts = [], [], []
    time_points = []

    for t in range(max_time_steps):
        # Save current counts without recomputing from scratch
        S_counts.append(current_S)
        I_counts.append(current_I)
        R_counts.append(current_R)
        time_points.append(t)

        if verbose:
            print(f"Time {t}: S={current_S}, I={current_I}, R={current_R}")

        if current_I == 0:
            if verbose:
                print("Epidemic died out.")
            break

        # Identify currently infected individuals
        infectors_this_step_indices = [i for i, ind in enumerate(individuals) if ind.state == INFECTED]
        newly_infected_in_this_step_indices = []

        for i_idx in infectors_this_step_indices:
            infector = individuals[i_idx]

            # Determine infection parameters for this infector
            if infector.is_superspreader:
                if superspreader_model_type == "strong":
                    current_r0_eff = r0_normal
                    current_w0_eff = w0_base
                    current_alpha_eff = 0.0
                elif superspreader_model_type == "hub":
                    current_r0_eff = r0_normal * np.sqrt(6)
                    current_w0_eff = w0_base
                    current_alpha_eff = alpha_normal
                else:
                    raise ValueError("Invalid superspreader_model_type")
            else:  # Normal individual
                current_r0_eff = r0_normal
                current_w0_eff = w0_base
                current_alpha_eff = alpha_normal

            # Attempt to infect susceptible individuals
            for s_idx in range(N):
                if individuals[s_idx].state == SUSCEPTIBLE:
                    susceptible_ind = individuals[s_idx]
                    dist = distance_periodic(infector.pos, susceptible_ind.pos, L)
                    
                    prob_infection = get_infection_probability(
                        dist, current_r0_eff, current_w0_eff, current_alpha_eff
                    )

                    if np.random.rand() < prob_infection:
                        if s_idx not in newly_infected_in_this_step_indices:
                            newly_infected_in_this_step_indices.append(s_idx)
                            current_S -= 1  # update running count

            # Recovery attempt for the current infector
            if np.random.rand() < gamma_recovery_prob:
                individuals[i_idx].state = RECOVERED
                current_I -= 1
                current_R += 1

        # Update states of newly infected individuals
        for ni_idx in newly_infected_in_this_step_indices:
            individuals[ni_idx].state = INFECTED
            current_I += 1

    # Append final counts (no loop needed since we're tracking running counts)
    S_counts.append(current_S)
    I_counts.append(current_I)
    R_counts.append(current_R)
    time_points.append(max_time_steps if current_I > 0 else t + 1)
    
    if verbose and current_I > 0 and t == max_time_steps - 1:
        print(f"Time {max_time_steps}: S={S_counts[-1]}, I={I_counts[-1]}, R={R_counts[-1]}")
        print("Max time steps reached.")

    return S_counts, I_counts, R_counts, time_points, individuals

def run_sir_simulation_with_history( # Renamed for clarity
    N=200, L=30.0,
    r0_normal=3.0,
    w0_base=1.0,
    alpha_normal=2.0,
    gamma_recovery_prob=1.0,
    lambda_ss_fraction=0.1,
    superspreader_model_type="hub",
    initial_infected_count=1,
    max_time_steps=100,
    verbose=False
):
    individuals = []
    num_superspreaders = int(N * lambda_ss_fraction)
    # Ensure unique indices for superspreaders if N is small
    superspreader_indices = random.sample(range(N), min(num_superspreaders, N))


    for i in range(N):
        is_ss = (i in superspreader_indices)
        individuals.append(Individual(i, L, is_superspreader=is_ss))

    # Initial infection
    # Ensure initial_infected_count does not exceed N
    actual_initial_infected_count = min(initial_infected_count, N)
    infected_indices = random.sample(range(N), actual_initial_infected_count)
    for idx in infected_indices:
        individuals[idx].state = INFECTED

    # Initialize running counts
    current_S = N - actual_initial_infected_count
    current_I = actual_initial_infected_count
    current_R = 0

    S_counts, I_counts, R_counts = [], [], []
    time_points = []
    simulation_history = [] # To store state of all individuals at each step
    
    t = 0 # Initialize t for potential early exit
    for t in range(max_time_steps):
        # Save current counts (already updated)
        S_counts.append(current_S)
        I_counts.append(current_I)
        R_counts.append(current_R)
        time_points.append(t)

        # --- Store current state of all individuals for visualization ---
        current_step_snapshot = []
        for ind in individuals:
            current_step_snapshot.append({
                'id': ind.id,
                'pos': ind.pos.copy(), # Important to copy numpy arrays
                'state': ind.state,
                'is_superspreader': ind.is_superspreader
            })
        simulation_history.append(current_step_snapshot)
        # --- End storing state ---

        if verbose:
            print(f"Time {t}: S={current_S}, I={current_I}, R={current_R}")

        if current_I == 0:
            if verbose: print("Epidemic died out.")
            break

        infectors_this_step_indices = [i for i, ind in enumerate(individuals) if ind.state == INFECTED]
        newly_infected_in_this_step_indices = []

        for i_idx in infectors_this_step_indices:
            infector = individuals[i_idx]
            if infector.is_superspreader:
                if superspreader_model_type == "strong":
                    current_r0_eff = r0_normal
                    current_w0_eff = w0_base
                    current_alpha_eff = 0.0
                elif superspreader_model_type == "hub":
                    current_r0_eff = r0_normal * np.sqrt(6)
                    current_w0_eff = w0_base
                    current_alpha_eff = alpha_normal
                else:
                    raise ValueError("Invalid superspreader_model_type")
            else:
                current_r0_eff = r0_normal
                current_w0_eff = w0_base
                current_alpha_eff = alpha_normal

            for s_idx in range(N):
                if individuals[s_idx].state == SUSCEPTIBLE:
                    susceptible_ind = individuals[s_idx]
                    dist = distance_periodic(infector.pos, susceptible_ind.pos, L)
                    prob_infection = get_infection_probability(
                        dist, current_r0_eff, current_w0_eff, current_alpha_eff
                    )
                    if np.random.rand() < prob_infection:
                        if s_idx not in newly_infected_in_this_step_indices:
                             newly_infected_in_this_step_indices.append(s_idx)
                             # current_S -= 1 # Decrement S when added to newly_infected list

            # Recovery attempt for the current infector
            if np.random.rand() < gamma_recovery_prob:
                individuals[i_idx].state = RECOVERED
                current_I -= 1
                current_R += 1
        
        # Update states of newly infected individuals and counts
        for ni_idx in newly_infected_in_this_step_indices:
            # Check if still susceptible (could have been marked for infection multiple times in one step by different infectors)
            # but only transition once and update counts once.
            if individuals[ni_idx].state == SUSCEPTIBLE:
                individuals[ni_idx].state = INFECTED
                current_S -= 1
                current_I += 1

    # Add final state to history
    final_step_snapshot = []
    for ind in individuals:
        final_step_snapshot.append({
            'id': ind.id,
            'pos': ind.pos.copy(),
            'state': ind.state,
            'is_superspreader': ind.is_superspreader
        })
    simulation_history.append(final_step_snapshot)

    # Append final SIR counts
    # If loop completed fully or broke early, the counts for the *next* time step (t or t+1) are these.
    S_counts.append(current_S)
    I_counts.append(current_I)
    R_counts.append(current_R)
    time_points.append(t + 1) # t is the last completed step, so this is the state *after* step t

    return S_counts, I_counts, R_counts, time_points, simulation_history


def visualize_simulation(history, L, N, model_name=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    title = ax.text(0.5, 1.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")

    # Create scatter plot objects once. We will update their data.
    # Normal individuals
    scatter_s = ax.scatter([], [], s=30, color='blue', alpha=0.7, label='Susceptible')
    scatter_i = ax.scatter([], [], s=30, color='red', alpha=0.7, label='Infected')
    scatter_r = ax.scatter([], [], s=30, color='grey', alpha=0.7, label='Recovered')
    # Superspreaders (distinguished by edge color or different marker)
    scatter_s_ss = ax.scatter([], [], s=50, facecolors='blue', edgecolors='black', linewidth=1.5, alpha=0.9, label='Susceptible (SS)')
    scatter_i_ss = ax.scatter([], [], s=50, facecolors='red', edgecolors='black', linewidth=1.5, alpha=0.9, label='Infected (SS)')
    scatter_r_ss = ax.scatter([], [], s=50, facecolors='grey', edgecolors='black', linewidth=1.5, alpha=0.9, label='Recovered (SS)')

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    def update(frame_num):
        current_snapshot = history[frame_num]
        
        s_pos, i_pos, r_pos = [], [], []
        s_ss_pos, i_ss_pos, r_ss_pos = [], [], []

        num_s, num_i, num_r = 0, 0, 0

        for ind_data in current_snapshot:
            pos = ind_data['pos']
            state = ind_data['state']
            is_ss = ind_data['is_superspreader']

            if state == SUSCEPTIBLE:
                num_s += 1
                if is_ss: s_ss_pos.append(pos)
                else: s_pos.append(pos)
            elif state == INFECTED:
                num_i += 1
                if is_ss: i_ss_pos.append(pos)
                else: i_pos.append(pos)
            elif state == RECOVERED:
                num_r += 1
                if is_ss: r_ss_pos.append(pos)
                else: r_pos.append(pos)
        
        # Update normal individuals
        scatter_s.set_offsets(np.array(s_pos) if s_pos else np.empty((0,2)))
        scatter_i.set_offsets(np.array(i_pos) if i_pos else np.empty((0,2)))
        scatter_r.set_offsets(np.array(r_pos) if r_pos else np.empty((0,2)))
        # Update superspreaders
        scatter_s_ss.set_offsets(np.array(s_ss_pos) if s_ss_pos else np.empty((0,2)))
        scatter_i_ss.set_offsets(np.array(i_ss_pos) if i_ss_pos else np.empty((0,2)))
        scatter_r_ss.set_offsets(np.array(r_ss_pos) if r_ss_pos else np.empty((0,2)))

        title.set_text(f"{model_name} - Time Step: {frame_num}\nS: {num_s}, I: {num_i}, R: {num_r} (Total: {N})")
        return scatter_s, scatter_i, scatter_r, scatter_s_ss, scatter_i_ss, scatter_r_ss, title

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                  interval=200, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show()
    return ani # Return animation object so it can be saved if needed