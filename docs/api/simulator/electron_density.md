# Electron Density representations

`cryojax` provides different options for how to represent electron density distributions in cryo-EM.

???+ abstract "`cryojax.simulator.AbstractElectronDensity`"
    ::: cryojax.simulator.AbstractElectronDensity
        options:
            members:
                - rotate_to_pose

## Voxel-based electron densities

???+ abstract "`cryojax.simulator.AbstractVoxels`"

    ::: cryojax.simulator.AbstractVoxels
        options:
            members:
                - is_real
                - shape
                - from_density_grid
                - from_atoms

### Fourier-space voxel representations

??? abstract "`cryojax.simulator.AbstractFourierVoxelGrid`"

    ::: cryojax.simulator.AbstractFourierVoxelGrid
        options:
            members:
                - __init__

!!! info "Fourier-space conventions"
    - The `fourier_density_grid` and `frequency_slice` arguments to
    `FourierVoxelGrid.__init__` should be loaded with the zero frequency
    component in the center of the box.
    - The parameters in an `AbstractPose` represent a rotation in real-space. This means that when calling `FourierVoxelGrid.rotate_to_pose`,
    frequencies are rotated by the inverse rotation as stored in the pose.

::: cryojax.simulator.FourierVoxelGrid
        options:
            members:
                - __init__
                - fourier_density_grid
                - voxel_size
                - frequency_slice
                - frequency_slice_in_angstroms

---

::: cryojax.simulator.FourierVoxelGridInterpolator
        options:
            members:
                - __init__
                - coefficients
                - voxel_size
                - frequency_slice
                - frequency_slice_in_angstroms

### Real-space voxel representations

::: cryojax.simulator.RealVoxelGrid
        options:
            members:
                - __init__
                - density_grid
                - voxel_size
                - coordinate_grid
                - coordinate_grid_in_angstroms

---

::: cryojax.simulator.RealVoxelCloud
        options:
            members:
                - __init__
                - density_weights
                - voxel_size
                - coordinate_list
                - coordinate_list_in_angstroms