from utils import *

# --- Example Usage ---
if __name__ == "__main__":
    N_pop = 200 # Smaller N for faster visualization
    r0_normal_val = 1.0
    target_density_param = 20.0 # rho * pi * r0^2
    L_val = np.sqrt(N_pop * np.pi * r0_normal_val**2 / target_density_param)
    
    print(f"Running simulation for visualization: N={N_pop}, L={L_val:.2f}")

    # Run simulation to get history
    # Use smaller lambda for visualization to clearly see SS effect
    S_h, I_h, R_h, t_h, history_hub = run_sir_simulation_with_history(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=1, # Slower recovery for longer viz
        lambda_ss_fraction=0.5, 
        superspreader_model_type="hub",
        initial_infected_count=1,
        max_time_steps=10, # Number of frames for animation
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
        for step in range(0, len(history_hub)):
            visualize_simulation_at_step(history_hub, L_val, N_pop, step=step, model_name="Hub Model", save_path=f"fig/sir_hub_step_{step}.png")
        # ani_hub = visualize_simulation(history_hub, L_val, N_pop, model_name="Hub Model")
        # ani_hub.save("fig/sir_hub_animation.gif", writer='pillow', fps=5)
    else:
        print("History is too short to animate.")
# 
    # Example for "strong" model (optional)
    S_s, I_s, R_s, t_s, history_strong = run_sir_simulation_with_history(
        N=N_pop, L=L_val, r0_normal=r0_normal_val,
        w0_base=1.0, alpha_normal=2.0, gamma_recovery_prob=1,
        lambda_ss_fraction=0.5,
        superspreader_model_type="strong",
        initial_infected_count=1,
        max_time_steps=10,
        verbose=True
    )
    if history_strong and len(history_strong) > 1:
        for step in range(0, len(history_hub)):
            visualize_simulation_at_step(history_strong, L_val, N_pop, step=step, model_name="Strong Model", save_path=f"fig/sir_strong_step_{step}.png")
        # ani_strong = visualize_simulation(history_strong, L_val, N_pop, model_name="Strong Infectiousness Model")
        # ani_strong.save("fig/sir_strong_animation.gif", writer='pillow', fps=5)