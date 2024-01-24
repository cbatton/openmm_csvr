from openmm_csvr import WCA

wca_int = WCA(folder_name='wca', density=15.0, num_particles=4800, integrator_scheme="middle")
wca_int.generate_long_trajectory(num_data_points=10000, save_freq=100)
