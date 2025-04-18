# Representing poses

`cryojax` provides different parameterizations for the pose of a structure. These are captured through the abstract base class called `AbstractPose`.

!!! info "Rotation and translation conventions"
    Pose transformations in `cryojax` can be captured by the equation

    $$\vec{x}' = R \vec{x} + \vec{t},$$

    where $\vec{x}'$ and $\vec{x}$ are the 3D coordinate vectors in the rotated and unrotated frames, and $R$ and $\vec{t} = (t_x, t_y, 0)$ are the rotation and in-plane translation vector.

    Some standard softwares define the rotation and translation to "undo" the observed pose.
    This can be captured by a translation to the center, followed by a intrinsic rotation:

    $$\vec{x} = (\vec{x}' + \vec{t}^*) R^*.$$

    When $R^* = R^T$ and $\vec{t}^* = -\vec{t}$, this equation can be inverted for $\vec{x}'$ to recover the `cryojax` convention. To actually convert between conventions for standard softwares such as RELION, for the translation we indeed have $\vec{t}^* = -\vec{t}$, but for the rotation we actually have $R^* = R$ (therefore, no conversion is necessary). This discrepency can perhaps be explained by the fact that `cryojax` defines its rotation parameterization with respect to a real-space rotation, but other softwares may define them with respect to fourier space. Real vs fourier space coordinate rotations differ only by transpose.

!!! info "Degrees vs radians conventions"
    Angular quantities in `cryojax` are always in *degrees*.
    Therefore concrete classes of the `AbstractPose` have
    angles in degrees, e.g.

    ```python
    import cryojax.simulator as cxs

    phi_in_degrees, theta_in_degrees, psi_in_degrees = 10.0, 30.0, 40.0
    pose = cxs.EulerAnglePose(
        phi_angle=phi_in_degrees,
        theta_angle=theta_in_degrees,
        psi_angle=psi_in_degrees,
    )
    ```

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
