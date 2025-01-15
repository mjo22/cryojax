# Representing poses

`cryojax` provides different parameterizations for the pose of a structure. These are captured through the abstract base class called `AbstractPose`.

???+ abstract "`cryojax.simulator.AbstractPose`"
    ::: cryojax.simulator.AbstractPose
        options:
            members:
                - compute_shifts
                - rotate_coordinates
                - offset_in_angstroms
                - rotation
                - from_rotation
                - from_rotation_and_translation

::: cryojax.simulator.EulerAnglePose
        options:
            members:
                - __init__

---

::: cryojax.simulator.QuaternionPose
        options:
            members:
                - __init__

---

::: cryojax.simulator.AxisAnglePose
        options:
            members:
                - __init__
