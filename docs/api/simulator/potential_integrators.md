# Potential integration methods

`cryojax` provides different methods for integrating [scattering potentials](./scattering_potential.md#scattering-potential-representations) onto a plane.

???+ abstract "`cryojax.simulator.AbstractPotentialIntegrator`"
    ::: cryojax.simulator.AbstractPotentialIntegrator
        options:
            members:
                - compute_fourier_integrated_potential

## Integration methods for voxel-based potentials

???+ abstract "`cryojax.simulator.AbstractVoxelPotentialIntegrator`"
    ::: cryojax.simulator.AbstractVoxelPotentialIntegrator
        options:
            members:
                - pixel_rescaling_method

??? abstract "`cryojax.simulator.AbstractFourierVoxelExtraction`"
    ::: cryojax.simulator.AbstractFourierVoxelExtraction
        options:
            members:
                - extract_voxels_from_spline_coefficients
                - extract_voxels_from_grid_points

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - compute_fourier_integrated_potential

::: cryojax.simulator.NufftProjection
        options:
            members:
                - __init__
                - compute_fourier_integrated_potential
