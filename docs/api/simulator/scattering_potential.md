# Scattering potential representations

`cryojax` provides different options for how to represent spatial potential energy distributions in cryo-EM.

???+ abstract "`cryojax.simulator.AbstractPotentialRepresentation`"
    ::: cryojax.simulator.AbstractPotentialRepresentation
        options:
            members:
                - rotate_to_pose

## Voxel-based scattering potentials

???+ abstract "`cryojax.simulator.AbstractVoxelPotential`"

    ::: cryojax.simulator.AbstractVoxelPotential
        options:
            members:
                - voxel_size
                - is_real
                - shape
                - rotate_to_pose
                - from_real_voxel_grid
                - from_atoms

### Fourier-space voxel representations

??? abstract "`cryojax.simulator.AbstractFourierVoxelGridPotential`"

    ::: cryojax.simulator.AbstractFourierVoxelGridPotential
        options:
            members:
                - __init__

!!! info "Fourier-space conventions"
    - The `fourier_voxel_grid` and `wrapped_frequency_slice` arguments to
    `FourierVoxelGrid.__init__` should be loaded with the zero frequency
    component in the center of the box.
    - The parameters in an `AbstractPose` represent a rotation in real-space. This means that when calling `FourierVoxelGridPotential.rotate_to_pose`,
    frequencies are rotated by the inverse rotation as stored in the pose.

::: cryojax.simulator.FourierVoxelGridPotential
        options:
            members:
                - __init__
                - wrapped_frequency_slice_in_angstroms
                - from_real_voxel_grid
                - from_atoms

---

::: cryojax.simulator.FourierVoxelGridPotentialInterpolator
        options:
            members:
                - __init__
                - wrapped_frequency_slice_in_angstroms
                - from_real_voxel_grid
                - from_atoms

### Real-space voxel representations

::: cryojax.simulator.RealVoxelGridPotential
        options:
            members:
                - __init__
                - wrapped_coordinate_grid_in_angstroms
                - from_real_voxel_grid
                - from_atoms

---

::: cryojax.simulator.RealVoxelCloudPotential
        options:
            members:
                - __init__
                - wrapped_coordinate_list_in_angstroms
                - from_real_voxel_grid
                - from_atoms

### Pure function API

::: cryojax.simulator.build_real_space_voxels_from_atoms
        options:
            heading_level: 5

---

::: cryojax.simulator.evaluate_3d_atom_potential
        options:
            heading_level: 5

---

::: cryojax.simulator.evaluate_3d_real_space_gaussian
        options:
            heading_level: 5
