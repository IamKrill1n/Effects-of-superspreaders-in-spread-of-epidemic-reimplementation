import numpy as np
import matplotlib.pyplot as plt
import random

# --- (Individual class, distance_periodic, get_infection_probability as before) ---
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

class Individual:
    def __init__(self, id, pos_array, is_superspreader=False):
        self.id = id
        self.pos = pos_array
        self.state = SUSCEPTIBLE
        self.is_superspreader = is_superspreader

def distance_periodic(pos1, pos2, L):
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > L / 2, L - delta, delta)
    return np.sqrt((delta**2).sum())

def get_infection_probability(r, r0_eff, w0_eff, alpha_eff):
    if r == 0: return 0.0
    if r <= r0_eff:
        if alpha_eff == 0: return w0_eff
        else: return w0_eff * (1 - r / r0_eff)**alpha_eff
    return 0.0

def run_sir_for_epicurve(
    N, L, r0_normal,
    w0_base, alpha_normal, gamma_recovery_prob,
    lambda_ss_fraction, superspreader_model_type,
    initial_infected_count=1,
    max_time_steps=50
):
    individuals = []
    num_superspreaders = int(N * lambda_ss_fraction)
    all_indices = list(range(N))
    
    is_superspreader_flags = [False] * N
    if num_superspreaders > 0 and N > 0: # Check N>0
        chosen_ss_indices = random.sample(all_indices, min(num_superspreaders, N)) # Handle N_ss > N
        for ss_idx in chosen_ss_indices:
            is_superspreader_flags[ss_idx] = True

    for i in range(N):
        pos_random = np.random.rand(2) * L
        individuals.append(Individual(i, pos_random, is_superspreader=is_superspreader_flags[i]))

    if initial_infected_count > 0 and N > 0:
        infected_indices = random.sample(range(N), min(initial_infected_count, N))
        for idx in infected_indices:
            individuals[idx].state = INFECTED
    
    S_counts, I_counts, R_counts = [], [], []
    newly_infected_counts_per_step = []
    time_points = []

    for t in range(max_time_steps):
        current_S = sum(1 for ind in individuals if ind.state == SUSCEPTIBLE)
        current_I = sum(1 for ind in individuals if ind.state == INFECTED)
        current_R = sum(1 for ind in individuals if ind.state == RECOVERED)

        S_counts.append(current_S)
        I_counts.append(current_I)
        R_counts.append(current_R)
        time_points.append(t)
        
        if current_I == 0 and t > 0:
            newly_infected_counts_per_step.extend([0] * (max_time_steps - len(newly_infected_counts_per_step)))
            break

        infectors_this_step_indices = [i for i, ind in enumerate(individuals) if ind.state == INFECTED]
        marked_for_infection_this_step_indices = []

        for i_idx in infectors_this_step_indices:
            infector = individuals[i_idx]
            if infector.is_superspreader:
                if superspreader_model_type == "strong":
                    r_eff, w_eff, a_eff = r0_normal, w0_base, 0.0
                elif superspreader_model_type == "hub":
                    r_eff, w_eff, a_eff = r0_normal * np.sqrt(6), w0_base, alpha_normal
            else:
                r_eff, w_eff, a_eff = r0_normal, w0_base, alpha_normal

            for s_idx in range(N):
                if individuals[s_idx].state == SUSCEPTIBLE:
                    dist = distance_periodic(infector.pos, individuals[s_idx].pos, L)
                    prob_infection = get_infection_probability(dist, r_eff, w_eff, a_eff)
                    if np.random.rand() < prob_infection:
                        if s_idx not in marked_for_infection_this_step_indices:
                             marked_for_infection_this_step_indices.append(s_idx)
            
            if np.random.rand() < gamma_recovery_prob:
                individuals[i_idx].state = RECOVERED
        
        actual_newly_infected_this_step = len(marked_for_infection_this_step_indices)
        newly_infected_counts_per_step.append(actual_newly_infected_this_step)
        
        for ni_idx in marked_for_infection_this_step_indices:
            individuals[ni_idx].state = INFECTED
            
    # Pad if loop completed fully without breaking
    if len(newly_infected_counts_per_step) < max_time_steps:
         newly_infected_counts_per_step.extend([0] * (max_time_steps - len(newly_infected_counts_per_step)))

    return S_counts, I_counts, R_counts, newly_infected_counts_per_step, time_points


# --- Main script for Figure 15 ---
if __name__ == "__main__":
    N_fig15 = 477
    rho_pi_r0_sq_fig15 = 15.0
    r0_normal_fig15 = 1.0

    L_fig15 = np.sqrt((N_fig15 * np.pi * r0_normal_fig15**2) / rho_pi_r0_sq_fig15)
    print(f"N={N_fig15}, ρπr₀²={rho_pi_r0_sq_fig15}, r₀={r0_normal_fig15} => L={L_fig15:.2f}")

    w0_base_val = 1.0
    alpha_normal_val = 2.0
    gamma_recovery_prob_val = 1.0
    initial_infected_fig15 = 1
    max_t_fig15 = 25 # Figure 15 x-axis goes up to 25

    num_simulation_runs_for_average = 100 # Adjust for speed/accuracy

    models_for_fig15 = {
        "Strong infectiousness model (λ=0.2)": {"type": "strong", "lambda": 0.2, "color": "red", "marker": "o", "linestyle": "None"}, # No line, just markers
        "Hub model (λ=0.2)": {"type": "hub", "lambda": 0.2, "color": "blue", "marker": "s", "linestyle": "None"},  # No line, just markers
        "(λ=0.0)": {"type": "hub", "lambda": 0.0, "color": "cyan", "marker": "^", "linestyle": "None"} # No line, just markers
    }

    plt.figure(figsize=(8, 6)) # Adjusted figsize to better match original aspect ratio

    # --- Visually Extracted SARS Data from Figure 15 ---
    sars_timesteps_data_extracted = [
        0, 0, 3, 10, 20, 52, 18, 16, 40, 28, 13, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ] # Padded to 25 time steps
    sars_time_points_extracted = np.arange(len(sars_timesteps_data_extracted))

    # Plot Real SARS Data (Using Extracted)
    plt.bar(sars_time_points_extracted, sars_timesteps_data_extracted, 
            color='orange', alpha=1.0, width=0.9, # Adjusted alpha and width
            label="data of SARS in Singapore")

    all_simulation_results = {}

    for model_name, params in models_for_fig15.items():
        print(f"Running: {model_name}")
        all_runs_newly_infected = []
        
        for i_run in range(num_simulation_runs_for_average):
            if (i_run + 1) % (num_simulation_runs_for_average // 10 or 1) == 0: # Print progress
                 print(f"  Run {i_run+1}/{num_simulation_runs_for_average}")
            _, _, _, newly_infected_single_run, _ = run_sir_for_epicurve(
                N=N_fig15, L=L_fig15, r0_normal=r0_normal_fig15,
                w0_base=w0_base_val, alpha_normal=alpha_normal_val,
                gamma_recovery_prob=gamma_recovery_prob_val,
                lambda_ss_fraction=params["lambda"],
                superspreader_model_type=params["type"],
                initial_infected_count=initial_infected_fig15,
                max_time_steps=max_t_fig15
            )
            all_runs_newly_infected.append(newly_infected_single_run[:max_t_fig15])

        if all_runs_newly_infected:
            avg_newly_infected = np.mean(np.array(all_runs_newly_infected), axis=0)
            sim_time_points = np.arange(len(avg_newly_infected))
            
            plt.plot(sim_time_points, avg_newly_infected,
                     label=model_name, color=params["color"], marker=params["marker"],
                     linestyle=params["linestyle"], markersize=6) # Increased marker size
        else:
            print(f"Warning: No simulation data for {model_name}")

    plt.xlabel("time step", fontsize=12)
    plt.ylabel("number of patients", fontsize=12)
    # plt.title("Fig 15 Replication: Epidemic Curves for SARS") # Original figure has no title
    
    # Legend styling to match Figure 15
    handles, labels = plt.gca().get_legend_handles_labels()
    # Manually reorder if necessary or create custom legend entries for better match
    # For simplicity, using default order but placing it similar to original
    plt.legend(handles, labels, loc='upper right', frameon=False, bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.grid(False) # Original figure has no grid
    plt.xlim(left=-0.5, right=max_t_fig15 - 0.5)
    plt.ylim(bottom=0, top=80) # Match y-axis limit
    
    # Match tick marks (approximate)
    plt.xticks(np.arange(0, max_t_fig15 + 1, 5))
    plt.yticks(np.arange(0, 81, 20))
    
    plt.tight_layout()
    plt.show()