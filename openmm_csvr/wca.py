import time

import numpy as np
from mdtraj.reporters import HDF5Reporter
from openmm import CMMotionRemover, CustomNonbondedForce, Platform, System
from openmm.app import CheckpointReporter, Element, Simulation, Topology
from openmm.unit import (
    AVOGADRO_CONSTANT_NA,
    BOLTZMANN_CONSTANT_kB,
    Quantity,
    amu,
    angstrom,
    kelvin,
    kilojoules_per_mole,
    md_unit_system,
    nanometer,
    nanometers,
    picoseconds,
)

from .csvr import CSVRIntegrator
from .sobol import i4_sobol_generate

kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA


class WCA:
    """
    WCA potential class
    """

    def __init__(
        self,
        num_particles=216,
        platform="CUDA",
        precision="single",
        temperature=0.824,
        density=0.96,
        mass=39.948 * amu,
        epsilon=120.0 * kB * kelvin,
        sigma=3.4 * angstrom,
        time_step=0.001,
        friction=0.1,
        seed=1,
        save_int=1000,
        count=0,
        folder_name="",
        integrator_scheme="middle",
    ):
        self.folder_name = folder_name
        self.count = count
        self.internal_count = 0
        self.base_filename = f"{self.folder_name}_{self.count}"
        self.save_int = save_int
        tau = (sigma**2 * mass / epsilon) ** 0.5
        # convert tau to ps
        tau = tau.in_units_of(picoseconds)
        temperature = temperature / (kB / epsilon)
        system = self._init_system(num_particles, mass)

        volume = num_particles / density
        length = volume ** (1.0 / 3.0)
        a = (
            Quantity(np.array([1.0, 0.0, 0.0], np.float32), nanometer)
            * length
            / nanometer
        )
        b = (
            Quantity(np.array([0.0, 1.0, 0.0], np.float32), nanometer)
            * length
            / nanometer
        )
        c = (
            Quantity(np.array([0.0, 0.0, 1.0], np.float32), nanometer)
            * length
            / nanometer
        )
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # WCA interaction
        energy_expression = "4.0*epsilon*((sigma/r)^12 - (sigma/r)^6) + epsilon;"
        energy_expression += "sigma = %f;" % self.in_openmm_units(sigma)
        energy_expression += "epsilon = %f;" % self.in_openmm_units(epsilon)

        # Create force.
        force = CustomNonbondedForce(energy_expression)

        # Add particles
        for n in range(num_particles):
            force.addParticle([])

        # Set periodic boundary conditions with cutoff.
        force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        rmin = self.in_openmm_units(
            2.0 ** (1.0 / 6.0) * sigma
        )  # distance of minimum energy for Lennard-Jones potential
        force.setCutoffDistance(rmin)

        # Add nonbonded force term to the system.
        system.addForce(force)
        system.addForce(CMMotionRemover())

        # Generate Argon topology
        topology = Topology()
        element = Element.getBySymbol("Ar")
        chain = topology.addChain()
        for _ in range(num_particles):
            residue = topology.addResidue("Ar", chain)
            topology.addAtom("Ar", element, residue)

        # Generate initial positions
        init_pos = self.subrandom_particle_positions(
            num_particles, system.getDefaultPeriodicBoxVectors()
        )

        platform = Platform.getPlatformByName(platform)
        properties = {"Precision": precision}

        if integrator_scheme == "verlet" or integrator_scheme == "middle":
            integrator = CSVRIntegrator(
                system=system,
                temperature=temperature,
                tau=friction * tau,
                timestep=time_step * tau,
                scheme=integrator_scheme,
            )
        else:
            raise ValueError("integrator_scheme must be 'verlet' or 'middle'")
        integrator.setRandomNumberSeed(seed)
        self.simulation = self._init_simulation(
            system, integrator, platform, properties, topology
        )
        reporter = HDF5Reporter(
            f"{self.base_filename}_sim_{self.internal_count}.h5", self.save_int
        )
        restart_reporter = CheckpointReporter(
            f"{self.base_filename}_restart.chk", self.save_int
        )
        self.simulation.reporters.append(restart_reporter)
        self.simulation.reporters.append(reporter)
        self.simulation.context.setPositions(init_pos)
        # Minimize energy
        self.simulation.minimizeEnergy(
            tolerance=Quantity(0.01, kilojoules_per_mole / nanometer), maxIterations=0
        )
        # save initial state
        self.simulation.context.setVelocitiesToTemperature(temperature)
        self.simulation.saveCheckpoint(f"{self.base_filename}_restart.chk")
        self.simulation.saveState(f"{self.base_filename}_restart.xml")

    def subrandom_particle_positions(self, num_particles, box_vectors):
        # Create positions array.
        positions = Quantity(np.zeros([num_particles, 3], np.float32), nanometers)

        # Generate Sobol' sequence.
        ivec = i4_sobol_generate(3, num_particles, 1)
        x = np.array(ivec, np.float32)
        for dim in range(3):
            length = box_vectors[dim][dim]
            positions[:, dim] = Quantity(x[dim, :] * length / length.unit, length.unit)

        return positions

    def in_openmm_units(self, value):
        """Strip the units from a openmm.unit.Quantity object after converting to natural OpenMM units

        Parameters
        ----------
        value : openmm.unit.Quantity
           The value to convert

        Returns
        -------
        unitless_value : float
           The value in natural OpenMM units, stripped of units.

        """

        unitless_value = value.in_unit_system(md_unit_system)
        unitless_value /= unitless_value.unit
        return unitless_value

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for steps steps
        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[1].close()

    def _init_system(self, num_particles, mass):
        """Initializes an OpenMM system
        Arguments:
            num_particles: An int specifying the number of particles in the system
            mass: A float specifying the mass of each particle
        Returns:
            system: An OpenMM system
        """
        system = System()
        for _ in range(num_particles):
            system.addParticle(mass)
        return system

    def _init_simulation(self, system, integrator, platform, properties, topology):
        """Initializes an OpenMM simulation
        Arguments:
            system: An OpenMM system
            integrator: An OpenMM integrator
            platform: An OpenMM platform specifying the device information
            num_beads: An int specifying the number of beads to use
        Returns:
            simulation: An OpenMM simulation object
        """
        simulation = Simulation(topology, system, integrator, platform, properties)
        return simulation

    def get_energy(self, positions, enforce_periodic_box=True):
        """Updates position and velocities of the system
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
        """
        self.simulation.context.setPositions(positions)
        state = self.simulation.context.getState(
            getPositions=False,
            getEnergy=True,
            enforcePeriodicBox=enforce_periodic_box,
        )
        pe = state.getPotentialEnergy().in_units_of(kilojoules_per_mole)._value
        return pe

    def get_information(self, as_numpy=True, enforce_periodic_box=True):
        """Gets information (positions, forces and PE of system)
        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kcal/mol*Angstroms
            pe: A float coressponding to the potential energy in kJ/mol
            ke: A float coressponding to the kinetic energy in kJ/mol
        """
        state = self.simulation.context.getState(
            getForces=True,
            getEnergy=True,
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=enforce_periodic_box,
        )
        positions = state.getPositions(asNumpy=as_numpy).in_units_of(nanometers)
        forces = state.getForces(asNumpy=as_numpy).in_units_of(
            kilojoules_per_mole / nanometers
        )
        velocities = state.getVelocities(asNumpy=as_numpy).in_units_of(
            nanometers / picoseconds
        )

        pe = state.getPotentialEnergy().in_units_of(kilojoules_per_mole)._value
        ke = state.getKineticEnergy().in_units_of(kilojoules_per_mole)._value

        return positions, velocities, forces, pe, ke

    def generate_long_trajectory(
        self,
        init_pos=None,
        num_data_points=int(1e8),
        save_freq=1000,
        enforce_periodic_box=True,
        tag=None,
        time_max=117 * 60,
    ):
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke)
           are saved every save_freq time steps
        Arguments:
            num_data_points: An int representing the number of data points to generate
            save_freq: An int representing the frequency for which to save data points
            tag: A string representing the prefix to add to a file
        Saves:
            tag + "_positions.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the positions of the trajectory in units of Angstroms
            tag + "_velocities.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the velocities of the trajectory in units of  Angstroms/picoseconds
            tag + "_forces.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the forces of the trajectory in units of kcal/mol*Angstroms
            tag + "_pe.txt": A numpy array of shape (num_data_points,) representing the pe of the trajectory in units of kcal/mol
            tag + "_ke.txt": A numpy array of shape (num_data_points,) representing the ke of the trajectory in units of kcal/mol
        """

        if tag is None:
            tag = self.base_filename

        if init_pos is not None:
            self.simulation.context.setPositions(init_pos)

        # Start a timer
        time_start = time.time()
        start_iter = 0
        try:
            data = np.loadtxt(tag + "_pe.txt")
            start_iter = len(data)
        except Exception:
            start_iter = 0
            pass
        for _ in range(start_iter, num_data_points):
            self.run_sim(save_freq)
            (
                positions,
                velocities,
                forces,
                pe,
                ke,
            ) = self.get_information()
            f = open(f"{tag}_{self.internal_count}_positions.txt", "ab")
            # use 64-bit precision
            np.savetxt(f, positions, fmt="%.18e")
            f.close()

            f = open(
                f"{tag}_{self.internal_count}_velocities.txt",
                "ab",
            )
            np.savetxt(f, velocities, fmt="%.18e")
            f.close()

            f = open(f"{tag}_{self.internal_count}_forces.txt", "ab")
            np.savetxt(f, forces, fmt="%.18e")
            f.close()

            f = open(f"{tag}_{self.internal_count}_pe.txt", "ab")
            np.savetxt(f, np.expand_dims(pe, 0), fmt="%.18e")
            f.close()

            f = open(f"{tag}_{self.internal_count}_ke.txt", "ab")
            np.savetxt(f, np.expand_dims(ke, 0), fmt="%.18e")
            f.close()

            # save plainly to pe to keep a count
            f = open(f"{tag}_pe.txt", "ab")
            np.savetxt(f, np.expand_dims(pe, 0), fmt="%.18e")
            f.close()

            # Basically want to have something here that will close current h5 and open a new one
            # once it reaches a certain length
            if _ % 100 == 0:
                self.simulation.reporters[1].close()
                self.internal_count += 1
                self.simulation.reporters[1] = HDF5Reporter(
                    f"{self.base_filename}_sim_{self.internal_count}.h5",
                    self.save_int,
                )

            # End timer
            time_end = time.time()
            # If time is greater than 1 hour 57 minutes, end
            if time_end - time_start > 117 * 60:
                self.simulation.reporters[1].close()
                break
