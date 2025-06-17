from utils import *
N_pop = 200
r0_normal_val = 1.0
target_density_param = 10.0 # rho * pi * r0^2
L_val = np.sqrt(N_pop * np.pi * r0_normal_val**2 / target_density_param)
    
print(f"Running simulation for visualization: N={N_pop}, L={L_val:.2f}")


S_h, I_h, R_h, t_h, individuals = run_sir_simulation(
    N=N_pop, L=L_val, r0_normal=r0_normal_val,
    w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=0.2, # Slower recovery for longer viz
    lambda_ss_fraction=0.05, # 5% superspreaders
    superspreader_model_type="strong",
    initial_infected_count=1,
    max_time_steps=50, # Number of frames for animation
    verbose=True
)

percolated = 1 if min(S_h) == 0 else 0
print(f"Percolated: {percolated}")