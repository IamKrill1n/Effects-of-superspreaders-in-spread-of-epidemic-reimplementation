import numpy as np
import matplotlib.pyplot as plt
import random
# from tqdm import tqdm # Optional: for progress bars

# --- (Keep Individual class, distance_periodic, get_infection_probability as before) ---
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

class Individual:
    def __init__(self, id, pos_array, is_superspreader=False): # Accept position directly
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
        if alpha_eff == 0:
            return w0_eff
        else:
            return w0_eff * (1 - r / r0_eff)**alpha_eff
    return 0.0

def run_single_sir_for_percolation(
    N, L, r0_normal,
    w0_base, alpha_normal, gamma_recovery_prob,
    lambda_ss_fraction, superspreader_model_type,
    max_time_steps_per_run=200 # Max time for one MC run; percolation might happen sooner
):
    """
    Runs a single SIR simulation and checks for percolation.
    Returns True if percolated, False otherwise.
    """
    individuals = []
    num_superspreaders = int(N * lambda_ss_fraction)
    all_indices = list(range(N))

    # 1. Initial infected individual at the bottom center
    # Ensure it's not chosen as a superspreader initially if that matters for placement
    # Or, decide its superspreader status like others. For simplicity, let it be random.
    
    initial_infected_idx = 0 # Let's use index 0 for the initial infector
    pos_initial_infected = np.array([L / 2, 0.01 * L]) # Bottom center
    
    # Assign superspreader status randomly, EXCLUDING the initial infected for now
    # Then assign it separately to ensure we know its status
    potential_ss_indices = list(range(1, N)) # Exclude index 0
    
    # If N is very small, handle potential sampling errors
    actual_num_ss_to_pick = min(num_superspreaders, len(potential_ss_indices))
    
    superspreader_indices_others = random.sample(potential_ss_indices, actual_num_ss_to_pick)
    
    # Create individuals
    for i in range(N):
        is_ss = False
        if i == initial_infected_idx:
            # Decide if the initial infector can be a superspreader
            # For now, let's assume it can be, based on the overall lambda
            if num_superspreaders > 0 and random.random() < lambda_ss_fraction: # Probabilistic for the first one
                 is_ss = True
                 if i not in superspreader_indices_others and len(superspreader_indices_others) < num_superspreaders:
                    # This logic is getting a bit complex. Simpler:
                    # Assign SS status completely randomly to all N, then pick one infected.
                    pass # Handled by global assignment below

        # Simpler SS assignment:
    is_superspreader_flags = [False] * N
    chosen_ss_indices = random.sample(all_indices, num_superspreaders)
    for ss_idx in chosen_ss_indices:
        is_superspreader_flags[ss_idx] = True

    # Create individuals
    individuals.append(Individual(0, pos_initial_infected, is_superspreader=is_superspreader_flags[0]))
    individuals[0].state = INFECTED

    for i in range(1, N): # Remaining N-1 individuals
        pos_random = np.random.rand(2) * L
        individuals.append(Individual(i, pos_random, is_superspreader=is_superspreader_flags[i]))

    # Define top region for percolation
    top_percolation_threshold_y = L * 0.99 # Reaches very close to the top

    for t in range(max_time_steps_per_run):
        current_I_count = sum(1 for ind in individuals if ind.state == INFECTED)
        if current_I_count == 0:
            return False # Epidemic died out before percolating

        infectors_this_step_indices = [i for i, ind in enumerate(individuals) if ind.state == INFECTED]
        newly_infected_in_this_step_indices = []

        for i_idx in infectors_this_step_indices:
            infector = individuals[i_idx]
            if infector.is_superspreader:
                if superspreader_model_type == "strong":
                    current_r0_eff, current_w0_eff, current_alpha_eff = r0_normal, w0_base, 0.0
                elif superspreader_model_type == "hub":
                    current_r0_eff, current_w0_eff, current_alpha_eff = r0_normal * np.sqrt(6), w0_base, alpha_normal
            else:
                current_r0_eff, current_w0_eff, current_alpha_eff = r0_normal, w0_base, alpha_normal

            for s_idx in range(N):
                if individuals[s_idx].state == SUSCEPTIBLE:
                    dist = distance_periodic(infector.pos, individuals[s_idx].pos, L)
                    prob_infection = get_infection_probability(dist, current_r0_eff, current_w0_eff, current_alpha_eff)
                    if np.random.rand() < prob_infection:
                        if s_idx not in newly_infected_in_this_step_indices:
                             newly_infected_in_this_step_indices.append(s_idx)
            
            if np.random.rand() < gamma_recovery_prob:
                individuals[i_idx].state = RECOVERED
        
        for ni_idx in newly_infected_in_this_step_indices:
            individuals[ni_idx].state = INFECTED
            # Check for percolation
            if individuals[ni_idx].pos[1] > top_percolation_threshold_y:
                return True # Percolated!
                
    return False # Max time steps reached without percolation or epidemic died

# --- Main script to generate data for Figs 3 & 4 ---
if __name__ == "__main__":
    N_population = 200  # Fixed population size for these plots
    r0_normal_val = 1.0 # Unit length for infection range
    
    # Common simulation parameters from paper
    w0_base_val = 1.0
    alpha_normal_val = 2.0
    gamma_recovery_prob_val = 1.0 # y=1
    
    num_mc_runs_per_point = 100 # Reduced for faster execution, paper uses 1000

    density_param_values = np.linspace(0.1, 25, 26) # ρπr₀² values for x-axis (0 to 25 with 26 points)
    lambda_ss_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Fractions of superspreaders

    superspreader_models_to_run = {
        "Strong Infectiousness": "strong", # Fig 3
        "Hub Model": "hub"                 # Fig 4
    }

    plt.figure(figsize=(12, 5))

    plot_idx = 1
    for fig_title, model_type_key in superspreader_models_to_run.items():
        plt.subplot(1, 2, plot_idx)
        print(f"\n--- Running for: {fig_title} ---")

        for lambda_val in lambda_ss_values:
            print(f"  λ = {lambda_val:.1f}")
            percolation_probabilities = []
            
            # Use tqdm for progress bar over density_param_values if installed
            # for density_param in tqdm(density_param_values, desc=f"λ={lambda_val:.1f}"):
            for density_param in density_param_values:
                if density_param == 0: # Avoid division by zero if L is calculated from it
                    percolation_probabilities.append(0.0)
                    continue

                L_system = np.sqrt((N_population * np.pi * r0_normal_val**2) / density_param)
                
                percolated_count = 0
                for _ in range(num_mc_runs_per_point):
                    if run_single_sir_for_percolation(
                        N=N_population, L=L_system, r0_normal=r0_normal_val,
                        w0_base=w0_base_val, alpha_normal=alpha_normal_val,
                        gamma_recovery_prob=gamma_recovery_prob_val,
                        lambda_ss_fraction=lambda_val,
                        superspreader_model_type=model_type_key
                    ):
                        percolated_count += 1
                
                prob = percolated_count / num_mc_runs_per_point
                percolation_probabilities.append(prob)
                # print(f"    ρπr₀² = {density_param:.2f}, L = {L_system:.2f}, P_perc = {prob:.3f}")


            plt.plot(density_param_values, percolation_probabilities, marker='o', linestyle='-', markersize=4, label=f"λ = {lambda_val:.1f}")

        plt.xlabel("ρπr₀²")
        plt.ylabel("Percolation Probability")
        plt.title(f"{fig_title}")
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.legend()
        plot_idx += 1

    plt.tight_layout()
    plt.show()