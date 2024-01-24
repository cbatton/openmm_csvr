from openmm import CMMotionRemover, CustomIntegrator
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    dalton,
    kelvin,
    kilojoules_per_mole,
    picoseconds,
)

_OPENMM_ENERGY_UNIT = kilojoules_per_mole


class PrettyPrintableIntegrator(object):
    """A PrettyPrintableIntegrator can format the contents of its step program for printing.

    This is a mix-in.

    TODO: We should check that the object (`self`) is a CustomIntegrator or subclass.

    """

    def pretty_format(self, as_list=False, step_types_to_highlight=None):
        """Generate a human-readable version of each integrator step.

        Parameters
        ----------
        as_list : bool, optional, default=False
           If True, a list of human-readable strings will be returned.
           If False, these will be concatenated into a single human-readable string.
        step_types_to_highlight : list of int, optional, default=None
           If specified, these step types will be highlighted.

        Returns
        -------
        readable_lines : list of str
           A list of human-readable versions of each step of the integrator
        """
        step_type_dict = {
            0: "{target} <- {expr}",
            1: "{target} <- {expr}",
            2: "{target} <- sum({expr})",
            3: "constrain positions",
            4: "constrain velocities",
            5: "allow forces to update the context state",
            6: "if({expr}):",
            7: "while({expr}):",
            8: "end",
        }

        if not hasattr(self, "getNumComputations"):
            raise Exception("This integrator is not a CustomIntegrator.")

        readable_lines = []
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ""
            step_type, target, expr = self.getComputationStep(step)
            highlight = (
                True
                if (step_types_to_highlight is not None)
                and (step_type in step_types_to_highlight)
                else False
            )
            if step_type in [8]:
                indent_level -= 1
            if highlight:
                line += "\x1b[6;30;42m"
            line += (
                "step {:6d} : ".format(step)
                + "   " * indent_level
                + step_type_dict[step_type].format(target=target, expr=expr)
            )
            if highlight:
                line += "\x1b[0m"
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)

        if as_list:
            return readable_lines
        else:
            return "\n".join(readable_lines)

    def pretty_print(self):
        """Pretty-print the computation steps of this integrator."""
        print(self.pretty_format())


class ThermostatedIntegrator(PrettyPrintableIntegrator, CustomIntegrator):
    """Add temperature functions to a CustomIntegrator.

    This class is intended to be inherited by integrators that maintain the
    stationary distribution at a given temperature. The constructor adds a
    global variable named "kT" defining the thermal energy at the given
    temperature. This global variable is updated through the temperature
    setter and getter.

    It also provide a utility function to handle per-DOF constants that
    must be computed only when the temperature changes.

    Notice that the CustomIntegrator internally stored by a Context object
    will loose setter and getter and any extra function you define. The same
    happens when you copy your integrator. You can restore the methods with
    the static method ThermostatedIntegrator.restore_interface().

    Parameters
    ----------
    temperature : unit.Quantity
        The temperature of the integrator heat bath (temperature units).
    timestep : unit.Quantity
        The timestep to pass to the CustomIntegrator constructor (time
        units).

    Examples
    --------
    We can inherit from ThermostatedIntegrator to automatically define
    setters and getters for the temperature and to add a per-DOF constant
    "sigma" that we need to update only when the temperature is changed.

    >>> import openmm
    >>> from openmm import unit
    >>> class TestIntegrator(ThermostatedIntegrator):
    ...     def __init__(self, temperature=298.0*unit.kelvin, timestep=1.0*unit.femtoseconds):
    ...         super(TestIntegrator, self).__init__(temperature, timestep)
    ...         self.addPerDofVariable("sigma", 0)  # velocity standard deviation
    ...         self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})
    ...

    We instantiate the integrator normally.

    >>> integrator = TestIntegrator(temperature=350*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=350.0, unit=kelvin)
    >>> integrator.setTemperature(380.0*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=380.0, unit=kelvin)
    >>> integrator.getGlobalVariableByName('kT')
    3.1594995390636815

    Notice that a CustomIntegrator loses any extra method after a serialization cycle.

    >>> integrator_serialization = openmm.XmlSerializer.serialize(integrator)
    >>> deserialized_integrator = openmm.XmlSerializer.deserialize(integrator_serialization)
    >>> deserialized_integrator.getTemperature()
    Traceback (most recent call last):
    ...
    AttributeError: type object 'object' has no attribute '__getattr__'

    We can restore the original interface with a class method

    >>> ThermostatedIntegrator.restore_interface(integrator)
    True
    >>> integrator.getTemperature()
    Quantity(value=380.0, unit=kelvin)
    >>> integrator.setTemperature(400.0*unit.kelvin)
    >>> isinstance(integrator, TestIntegrator)
    True

    """

    def __init__(self, temperature, *args, **kwargs):
        super(ThermostatedIntegrator, self).__init__(*args, **kwargs)
        self.addGlobalVariable(
            "kT", MOLAR_GAS_CONSTANT_R * temperature
        )  # thermal energy

    @property
    def global_variable_names(self):
        """The set of global variable names defined for this integrator."""
        return set(
            [
                self.getGlobalVariableName(index)
                for index in range(self.getNumGlobalVariables())
            ]
        )

    def getTemperature(self):
        """Return the temperature of the heat bath.

        Returns
        -------
        temperature : unit.Quantity
            The temperature of the heat bath in kelvins.

        """
        # Do most unit conversion first for precision
        conversion = _OPENMM_ENERGY_UNIT / MOLAR_GAS_CONSTANT_R
        temperature = self.getGlobalVariableByName("kT") * conversion
        return temperature

    def setTemperature(self, temperature):
        """Set the temperature of the heat bath.

        Parameters
        ----------
        temperature : unit.Quantity
            The new temperature of the heat bath (temperature units).

        """
        kT = MOLAR_GAS_CONSTANT_R * temperature
        self.setGlobalVariableByName("kT", kT)

        # Update the changed flag if it exist.
        if "has_kT_changed" in self.global_variable_names:
            self.setGlobalVariableByName("has_kT_changed", 1)

    def addComputeTemperatureDependentConstants(self, compute_per_dof):
        """Wrap the ComputePerDof into an if-block executed only when kT changes.

        Parameters
        ----------
        compute_per_dof : dict of str: str
            A dictionary of variable_name: expression.

        """
        # First check if flag variable already exist.
        if "has_kT_changed" not in self.global_variable_names:
            self.addGlobalVariable("has_kT_changed", 1)

        # Create if-block that conditionally update the per-DOF variables.
        self.beginIfBlock("has_kT_changed = 1")
        for variable, expression in compute_per_dof.items():
            self.addComputePerDof(variable, expression)
        self.addComputeGlobal("has_kT_changed", "0")
        self.endBlock()

    @classmethod
    def is_thermostated(cls, integrator):
        """Return true if the integrator is a ThermostatedIntegrator.

        This can be useful when you only have access to the Context
        CustomIntegrator, which loses all extra function during serialization.

        Parameters
        ----------
        integrator : openmm.Integrator
            The integrator to check.

        Returns
        -------
        True if the original CustomIntegrator class inherited from
        ThermostatedIntegrator, False otherwise.

        """
        global_variable_names = set(
            [
                integrator.getGlobalVariableName(index)
                for index in range(integrator.getNumGlobalVariables())
            ]
        )
        if "kT" not in global_variable_names:
            return False
        return super(ThermostatedIntegrator, cls).is_restorable(integrator)

    @classmethod
    def restore_interface(cls, integrator):
        """Restore the original interface of a CustomIntegrator.

        The function restore the methods of the original class that
        inherited from ThermostatedIntegrator. Return False if the interface
        could not be restored.

        Parameters
        ----------
        integrator : openmm.CustomIntegrator
            The integrator to which add methods.

        Returns
        -------
        True if the original class interface could be restored, False otherwise.

        """
        restored = super(ThermostatedIntegrator, cls).restore_interface(integrator)
        # Warn the user if he is implementing a CustomIntegrator
        # that may keep the stationary distribution at a certain
        # temperature without exposing getters and setters.
        return restored

    @property
    def kT(self):
        """The thermal energy in openmm.Quantity"""
        return self.getGlobalVariableByName("kT") * _OPENMM_ENERGY_UNIT


class CSVRIntegrator(ThermostatedIntegrator):

    """CSVR thermostat"""

    def __init__(
        self,
        system=None,
        temperature=298 * kelvin,
        tau=0.2 * picoseconds,
        timestep=0.002 * picoseconds,
        constraint_tolerance=1e-8,
    ):
        """Construct a CSVR integrator"""

        super(CSVRIntegrator, self).__init__(temperature, timestep)
        self.addGlobalVariable("tau", tau)
        self.setConstraintTolerance(constraint_tolerance)

        # Compute the number of degrees of freedom
        if system is None:
            raise ValueError("Please specify a system")
        else:
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(
                type(system.getForce(i)) == CMMotionRemover
                for i in range(system.getNumForces())
            ):
                dof -= 3

            self.addGlobalVariable("ndf", dof)
            self.dof = dof

        self.addGlobalVariable("scale", 0)
        self.addGlobalVariable("KE", 0)
        self.addGlobalVariable("KEratio", 0)
        self.addGlobalVariable("c1", 0)
        self.addComputeGlobal("c1", "exp(-0.5*dt/tau)")
        self.addGlobalVariable("c2", 0)
        self.addGlobalVariable("r1", 0)
        self.addGlobalVariable("r2", 0)
        self.addGlobalVariable("nn", 0)
        self.addGlobalVariable("d_gamma", 0)
        self.addGlobalVariable("c_gamma", 0)
        self.addGlobalVariable("end_loop", 0)
        self.addGlobalVariable("v_gamma", 0)
        self.addGlobalVariable("x_gamma", 0)
        self.addGlobalVariable("u_gamma", 0)
        self.addPerDofVariable("x1", 0)

        # Perform velocity Verlet step with thermostat propagated before and after
        self.thermostat_step()
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()
        self.addComputeSum("KE", "0.5*m*v^2")
        self.thermostat_step()

    def thermostat_step(self):
        """Perform a CSVR thermostat step"""
        self.addComputeGlobal("scale", "0.0")
        self.addComputeSum("KE", "0.5*m*v^2")
        self.addComputeGlobal("KEratio", "0.5*kT/KE")
        self.addComputeGlobal("c2", "(1-c1)*KEratio")
        self.addComputeGlobal("r1", "gaussian")
        self.sum_noises()
        self.addComputeGlobal("scale", "scale+c1")
        self.addComputeGlobal("scale", "scale+c2*(r1^2+r2)")
        self.addComputeGlobal("scale", "scale+2*r1*sqrt(c1*c2)")
        self.addComputeGlobal("scale", "sqrt(scale)")
        self.addComputePerDof("v", "scale*v")

    def sum_noises(self):
        """
        Calculate the same of self.dof-1 gaussian random numbers.
        Can use a Gamma distribution to do it more efficiently.
        """

        self.addComputeGlobal("r2", "0")
        if (self.dof - 1) == 0:
            self.addComputeGlobal("r2", "0")
        elif (self.dof - 1) == 1:
            self.addComputeGlobal("r2", "gaussian")
            self.addComputeGlobal("r2", "r2*r2")
        elif (self.dof - 1) % 2 == 0:
            self.gamma_dist()
            self.addComputeGlobal("r2", "2*r2")
        else:
            self.gamma_dist()
            self.addComputeGlobal("r2", "2*r2+gaussian^2")

    def gamma_dist(self):
        """
        Calculate the result of a Gamma distribution.
        Have different cases for different values of self.dof-1.
        Follow routine of https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
        """
        self.addComputeGlobal("nn", "0")
        if (self.dof - 1) % 2 == 0:
            self.addComputeGlobal("nn", "(ndf-1)/2")
        else:
            self.addComputeGlobal("nn", "(ndf-2)/2")
        self.addComputeGlobal("d_gamma", "nn-1/3")
        self.addComputeGlobal("c_gamma", "1/sqrt(9*d_gamma)")
        self.addComputeGlobal("end_loop", "0")
        self.beginWhileBlock("end_loop = 0")
        self.addComputeGlobal("v_gamma", "-1")
        self.beginWhileBlock("v_gamma <= 0")
        self.addComputeGlobal("x_gamma", "gaussian")
        self.addComputeGlobal("v_gamma", "1+c_gamma*x_gamma")
        self.endBlock()
        self.addComputeGlobal("v_gamma", "v_gamma^3")
        self.addComputeGlobal("u_gamma", "uniform")
        self.beginIfBlock(
            "log(u_gamma) < 0.5*x_gamma^2+d_gamma*(1-v_gamma+log(v_gamma))"
        )
        self.addComputeGlobal("end_loop", "1")
        self.endBlock()
        self.endBlock()
        self.addComputeGlobal("r2", "d_gamma*v_gamma")
