from utils import *

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