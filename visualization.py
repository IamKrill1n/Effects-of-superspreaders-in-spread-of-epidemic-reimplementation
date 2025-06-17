import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy # For deep copying individual states

# Constants for states
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# --- (Keep Individual class and distance_periodic function as before) ---
class Individual:
    def __init__(self, id, L, is_superspreader=False):
        self.id = id
        self.pos = np.random.rand(2) * L
        self.state = SUSCEPTIBLE
        self.is_superspreader = is_superspreader
        self.infected_by = -1
        self.infected_others_count = 0

def distance_periodic(pos1, pos2, L):
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > L / 2, L - delta, delta)
    return np.sqrt((delta**2).sum())

def get_infection_probability(r, r0_eff, w0_eff, alpha_eff):
    if r == 0: return 0.0
    if r <= r0_eff:
        if alpha_eff == 0:
            return w0_eff
        else:
            return w0_eff * (1 - r / r0_eff)**alpha_eff
    return 0.0

def run_sir_simulation_with_history( # Renamed for clarity
    N=200, L=30.0,
    r0_normal=3.0,
    w0_base=1.0,
    alpha_normal=2.0,
    gamma_recovery_prob=1.0,
    lambda_ss_fraction=0.1,
    superspreader_model_type="hub",
    initial_infected_count=1,
    initial_infected_pos="bottom",
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
        if initial_infected_pos == "bottom":
            individuals[idx].pos = min(individuals, key=lambda ind: ind.pos[1]).pos.copy()
        
        print(f"Individual {idx} initialized as infected at position {individuals[idx].pos}") if verbose else None

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
            print(infector.pos)
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
                                  interval=1000, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show()
    return ani # Return animation object so it can be saved if needed

# --- Example Usage ---
if __name__ == "__main__":
    N_pop = 100 # Smaller N for faster visualization
    r0_normal_val = 1.0
    target_density_param = 10.0 # rho * pi * r0^2
    L_val = np.sqrt(N_pop * np.pi * r0_normal_val**2 / target_density_param)
    
    print(f"Running simulation for visualization: N={N_pop}, L={L_val:.2f}")

    # Run simulation to get history
    # Use smaller lambda for visualization to clearly see SS effect
    S_h, I_h, R_h, t_h, history_hub = run_sir_simulation_with_history(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=0.2, # Slower recovery for longer viz
        lambda_ss_fraction=0.05, # 5% superspreaders
        superspreader_model_type="strong",
        initial_infected_count=1,
        max_time_steps=50, # Number of frames for animation
        verbose=True
    )
    
    # Plot the SIR curves as before (optional, but good for context)
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_h, S_h, label="Susceptible (Hub)", color='blue')
    # plt.plot(t_h, I_h, label="Infected (Hub)", color='red')
    # plt.plot(t_h, R_h, label="Recovered (Hub)", color='grey')
    # plt.xlabel("Time Steps")
    # plt.ylabel("Number of Individuals")
    # plt.title(f"SIR Dynamics (Hub Model, N={N_pop}, λ=0.05)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Visualize the simulation
    print(f"Starting animation for Hub model... (Close plot window to continue if script hangs)")
    # Ensure history has enough frames
    if history_hub and len(history_hub) > 1:
        ani_hub = visualize_simulation(history_hub, L_val, N_pop, model_name="Hub Model")
        ani_hub.save("fig/sir_hub_animation.gif", writer='pillow', fps=5)
    else:
        print("History is too short to animate.")
# 
    # Example for "strong" model (optional)
    # S_s, I_s, R_s, t_s, history_strong = run_sir_simulation_with_history(
    #     N=N_pop, L=L_val, r0_normal=r0_normal_val,
    #     w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=0.2,
    #     lambda_ss_fraction=0.05,
    #     superspreader_model_type="strong",
    #     initial_infected_count=2,
    #     max_time_steps=50,
    #     verbose=False
    # )
    # if history_strong and len(history_strong) > 1:
    #     ani_strong = visualize_simulation(history_strong, L_val, N_pop, model_name="Strong Infectiousness Model")
    #     ani_strong.save("fig/sir_strong_animation.gif", writer='pillow', fps=5)