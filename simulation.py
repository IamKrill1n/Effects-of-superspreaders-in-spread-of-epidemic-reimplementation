import numpy as np
import matplotlib.pyplot as plt
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
    initial_infected_pos="bottom",
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
        if initial_infected_pos == "bottom":
            individuals[idx].pos = min(individuals, key=lambda ind: ind.pos[1]).pos.copy()
          
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


# --- Example Usage and Plotting ---
if __name__ == "__main__":
    N_pop = 477 # From Fig 15 caption
    # rho*pi*r0^2 = 15.0 (from Fig 15 caption)
    # rho = N / L^2
    # N / L^2 * pi * r0^2 = 15.0
    # Let r0_normal_val = 1.0 for simplicity in relative scaling
    # Then N / L^2 * pi = 15.0
    # L^2 = N * pi / 15.0
    # L = sqrt(N * pi / 15.0)

    r0_normal_val = 1.0 # Let's use r0 as a unit length
    target_density_param = 15.0 # This is rho * pi * r0^2
    
    L_val = np.sqrt(N_pop * np.pi * r0_normal_val**2 / target_density_param)
    print(f"N={N_pop}, r0_normal={r0_normal_val:.2f}, Calculated L={L_val:.2f} for density param={target_density_param}")
    
    # Parameters from paper/figures (approximated or stated)
    # w0_base_val = 1.0 (Fig 1, 2 imply w(r)/w0 peaks at 1 or is 1)
    # gamma_recovery_prob_val = 1.0 (stated in paper text "y = 1")
    # alpha_normal_val = 2.0 (stated in paper text "We set a = 2 for the normal...")

    # Run for Hub Model (as it was concluded to be more explanatory)
    S_hub, I_hub, R_hub, t_hub, inds_hub = run_sir_simulation(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=1.0,
        lambda_ss_fraction=0.04, # Fig 15 uses lambda=0.4, but in their system, lambda is density of SS.
                               # The paper is a bit unclear if lambda is fraction or density.
                               # Let's assume fraction here based on "fraction is denoted by lambda".
                               # The caption for Fig 15 is "lambda=0.4".
                               # The definition of lambda is "fraction is denoted by lambda".
                               # This is a bit confusing. CDC says superspreaders infect >10.
                               # SARS patients infecting 12,21,23,40 are SS. In Singapore data [2], there were ~200 cases.
                               # 4 superspreaders out of ~200 cases is ~2%.
                               # Let's try a smaller lambda for now.
                               # The paper's Fig 15 comparison uses lambda=0.4. Let's use that.
        superspreader_model_type="hub",
        max_time_steps=50, # Fig 15 goes up to ~25 "time steps", each is 6 days.
        verbose=True
    )

    # Run for Strong Infectiousness Model
    S_strong, I_strong, R_strong, t_strong, inds_strong = run_sir_simulation(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=1.0,
        lambda_ss_fraction=0.4, # Using paper's value for Fig 15
        superspreader_model_type="strong",
        max_time_steps=50,
        verbose=False
    )
    
    # Run for No Superspreaders
    S_no_ss, I_no_ss, R_no_ss, t_no_ss, inds_no_ss = run_sir_simulation(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=1.0,
        lambda_ss_fraction=0.0, # No superspreaders
        superspreader_model_type="hub", # Model type doesn't matter if lambda is 0
        max_time_steps=50,
        verbose=False
    )


    plt.figure(figsize=(12, 8))

    plt.subplot(2,1,1)
    plt.plot(t_hub, I_hub, label=f"Infected (Hub, λ=0.4)", color='blue', marker='s', linestyle='--')
    plt.plot(t_strong, I_strong, label=f"Infected (Strong, λ=0.4)", color='red', marker='o', linestyle=':')
    plt.plot(t_no_ss, I_no_ss, label=f"Infected (No SS, λ=0.0)", color='green', marker='^', linestyle='-')
    plt.xlabel("Time Steps (1 time step = 6 days from Fig. 15)")
    plt.ylabel("Number of Individuals")
    plt.title("SIR Model with Superspreaders: Infected Count")
    plt.legend()
    plt.grid(True)

    # For Fig 8 comparison - Number of NEWLY infected
    # This requires modifying the simulation to return newly_infected_counts per step
    # The current I_counts are prevalent infected.
    # Epidemic curve in paper's Fig 8 is "number of newly infected individuals per timestep"
    
    # Let's plot the S, I, R for one model
    plt.subplot(2,1,2)
    plt.plot(t_hub, S_hub, label=f"Susceptible (Hub, λ=0.4)", color='skyblue')
    plt.plot(t_hub, I_hub, label=f"Infected (Hub, λ=0.4)", color='blue')
    plt.plot(t_hub, R_hub, label=f"Recovered (Hub, λ=0.4)", color='grey')
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Individuals")
    plt.title("SIR Dynamics (Hub Model with Superspreaders λ=0.4)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # To get secondary infection distribution (like Fig 13)
    # You would need to run many simulations and average, or analyze one run.
    # The `inds_hub[i].infected_others_count` (if implemented and filled) would give this.
    # This requires modifications to the main loop to track who infected whom.
    # For example:
    # if np.random.rand() < prob_infection:
    #     if s_idx not in newly_infected_in_this_step_indices:
    #         newly_infected_in_this_step_indices.append(s_idx)
    #         individuals[s_idx].infected_by = infector.id # Track who infected
    #         individuals[i_idx].infected_others_count += 1 # Increment count for infector
            
    # Then after the simulation:
    # secondary_infections = [ind.infected_others_count for ind in inds_hub if ind.infected_others_count > 0 or ind.state != SUSCEPTIBLE]
    # plt.figure()
    # plt.hist(secondary_infections, bins=range(max(secondary_infections)+2), density=True, align='left', rwidth=0.8)
    # plt.xlabel("Number of secondary infections (links)")
    # plt.ylabel("Fraction")
    # plt.title("Distribution of Secondary Infections (Hub Model, one run)")
    # plt.show()