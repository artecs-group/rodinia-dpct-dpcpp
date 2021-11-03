// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "helper_cuda.h"
#include "helper_timer.h"

#include <iostream>
#include <fstream>
#include <cmath>

#if CUDART_VERSION < 3000
struct double3
{
	double x, y, z;
};
#endif

/*
 * Options 
 * 
 */ 
#define GAMMA 1.4
#define iterations 2000
#ifndef block_length
	#define block_length 128
#endif

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif


#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	T* t;
        checkCudaErrors((t = (T *)sycl::malloc_device(
                             sizeof(T) * N, dpct::get_default_queue()),
                         0));
        return t;
}

template <typename T>
void dealloc(T* array)
{
        checkCudaErrors((sycl::free((void *)array, dpct::get_default_queue()), 0));
}

template <typename T>
void copy(T* dst, T* src, int N)
{
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
}

template <typename T>
void upload(T* dst, T* src, int N)
{
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
}

template <typename T>
void download(T* dst, T* src, int N)
{
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
}

void dump(double* variables, int nel, int nelr)
{
	double* h_variables = new double[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
dpct::constant_memory<double, 1> ff_variable(NVAR);
dpct::constant_memory<sycl::double3, 1> ff_fc_momentum_x(1);
dpct::constant_memory<sycl::double3, 1> ff_fc_momentum_y(1);
dpct::constant_memory<sycl::double3, 1> ff_fc_momentum_z(1);
dpct::constant_memory<sycl::double3, 1> ff_fc_density_energy(1);

SYCL_EXTERNAL void cuda_initialize_variables(int nelr, double *variables,
                                             sycl::nd_item<3> item_ct1,
                                             double *ff_variable)
{
        const int i =
            (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));
        for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
}
void initialize_variables(int nelr, double* variables)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        int error;
        /*
        DPCT1049:63: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                ff_variable.init();

                auto ff_variable_ptr_ct1 = ff_variable.get_ptr();

                cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         cuda_initialize_variables(
                                             nelr, variables, item_ct1,
                                             ff_variable_ptr_ct1);
                                 });
        });
        /*
        DPCT1010:64: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        error = 0;
}

SYCL_EXTERNAL inline void compute_flux_contribution(
    double &density, sycl::double3 &momentum, double &density_energy,
    double &pressure, sycl::double3 &velocity, sycl::double3 &fc_momentum_x,
    sycl::double3 &fc_momentum_y, sycl::double3 &fc_momentum_z,
    sycl::double3 &fc_density_energy)
{
        fc_momentum_x.x() = velocity.x() * momentum.x() + pressure;
        fc_momentum_x.y() = velocity.x() * momentum.y();
        fc_momentum_x.z() = velocity.x() * momentum.z();

        fc_momentum_y.x() = fc_momentum_x.y();
        fc_momentum_y.y() = velocity.y() * momentum.y() + pressure;
        fc_momentum_y.z() = velocity.y() * momentum.z();

        fc_momentum_z.x() = fc_momentum_x.z();
        fc_momentum_z.y() = fc_momentum_y.z();
        fc_momentum_z.z() = velocity.z() * momentum.z() + pressure;

        double de_p = density_energy+pressure;
        fc_density_energy.x() = velocity.x() * de_p;
        fc_density_energy.y() = velocity.y() * de_p;
        fc_density_energy.z() = velocity.z() * de_p;
}

SYCL_EXTERNAL inline void compute_velocity(double &density,
                                           sycl::double3 &momentum,
                                           sycl::double3 &velocity)
{
        velocity.x() = momentum.x() / density;
        velocity.y() = momentum.y() / density;
        velocity.z() = momentum.z() / density;
}

SYCL_EXTERNAL inline double compute_speed_sqd(sycl::double3 &velocity)
{
        return velocity.x() * velocity.x() + velocity.y() * velocity.y() +
               velocity.z() * velocity.z();
}

SYCL_EXTERNAL inline double
compute_pressure(double &density, double &density_energy, double &speed_sqd)
{
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}

SYCL_EXTERNAL inline double compute_speed_of_sound(double &density,
                                                   double &pressure)
{
        return sycl::sqrt(double(GAMMA) * pressure / density);
}

SYCL_EXTERNAL void cuda_compute_step_factor(int nelr, double *variables,
                                            double *areas, double *step_factors,
                                            sycl::nd_item<3> item_ct1)
{
        const int i =
            (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));

        double density = variables[i + VAR_DENSITY*nelr];
        sycl::double3 momentum;
        momentum.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
        momentum.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
        momentum.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

        double density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];

        sycl::double3 velocity; compute_velocity(density, momentum, velocity);
        double speed_sqd      = compute_speed_sqd(velocity);
	double pressure       = compute_pressure(density, density_energy, speed_sqd);
	double speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = double(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
        step_factors[i] =
            double(0.5) /
            (sycl::sqrt(areas[i]) * (sycl::sqrt(speed_sqd) + speed_of_sound));
}
void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
        int error;
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        /*
        DPCT1049:66: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        /*
        DPCT1010:67: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         cuda_compute_step_factor(
                                             nelr, variables, areas,
                                             step_factors, item_ct1);
                                 });
        });
            error = 0;
}


void cuda_compute_flux_contributions(int nelr, double* variables, double* fc_momentum_x, double* fc_momentum_y, double* fc_momentum_z, double* fc_density_energy,
                                     sycl::nd_item<3> item_ct1)
{
        const int i =
            (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));

        double density_i = variables[i + VAR_DENSITY*nelr];
        sycl::double3 momentum_i;
        momentum_i.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
        momentum_i.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
        momentum_i.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];
        double density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

        sycl::double3 velocity_i; compute_velocity(density_i, momentum_i, velocity_i);
        double speed_sqd_i                          = compute_speed_sqd(velocity_i);
        double speed_i = sycl::sqrt((float)speed_sqd_i);
        double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
        sycl::double3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
        sycl::double3 fc_i_density_energy;
        compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z, fc_i_density_energy);

        fc_momentum_x[i + 0 * nelr] = fc_i_momentum_x.x();
        fc_momentum_x[i + 1 * nelr] = fc_i_momentum_x.y();
        fc_momentum_x[i + 2 * nelr] = fc_i_momentum_x.z();

        fc_momentum_y[i + 0 * nelr] = fc_i_momentum_y.x();
        fc_momentum_y[i + 1 * nelr] = fc_i_momentum_y.y();
        fc_momentum_y[i + 2 * nelr] = fc_i_momentum_y.z();

        fc_momentum_z[i + 0 * nelr] = fc_i_momentum_z.x();
        fc_momentum_z[i + 1 * nelr] = fc_i_momentum_z.y();
        fc_momentum_z[i + 2 * nelr] = fc_i_momentum_z.z();

        fc_density_energy[i + 0 * nelr] = fc_i_density_energy.x();
        fc_density_energy[i + 1 * nelr] = fc_i_density_energy.y();
        fc_density_energy[i + 2 * nelr] = fc_i_density_energy.z();
}
void compute_flux_contributions(int nelr, double* variables, double* fc_momentum_x, double* fc_momentum_y, double* fc_momentum_z, double* fc_density_energy)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        int error;
        /*
        DPCT1049:69: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         cuda_compute_flux_contributions(
                                             nelr, variables, fc_momentum_x,
                                             fc_momentum_y, fc_momentum_z,
                                             fc_density_energy, item_ct1);
                                 });
        });
                    /*
                    DPCT1010:70: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    error = 0;
}


/*
 *
 *
*/
void cuda_compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fc_momentum_x, double* fc_momentum_y, double* fc_momentum_z, double* fc_density_energy, double* fluxes,
                       sycl::nd_item<3> item_ct1, double *ff_variable,
                       sycl::double3 *ff_fc_momentum_x,
                       sycl::double3 *ff_fc_momentum_y,
                       sycl::double3 *ff_fc_momentum_z,
                       sycl::double3 *ff_fc_density_energy)
{
	const double smoothing_coefficient = double(0.2f);
        const int i =
            (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));

        int j, nb;
        sycl::double3 normal; double normal_len;
        double factor;
	
	double density_i = variables[i + VAR_DENSITY*nelr];
        sycl::double3 momentum_i;
        momentum_i.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
        momentum_i.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
        momentum_i.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

        double density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

        sycl::double3 velocity_i; compute_velocity(density_i, momentum_i, velocity_i);
        double speed_sqd_i                          = compute_speed_sqd(velocity_i);
        double speed_i = sycl::sqrt(speed_sqd_i);
        double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
        sycl::double3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
        sycl::double3 fc_i_density_energy;

        fc_i_momentum_x.x() = fc_momentum_x[i + 0 * nelr];
        fc_i_momentum_x.y() = fc_momentum_x[i + 1 * nelr];
        fc_i_momentum_x.z() = fc_momentum_x[i + 2 * nelr];

        fc_i_momentum_y.x() = fc_momentum_y[i + 0 * nelr];
        fc_i_momentum_y.y() = fc_momentum_y[i + 1 * nelr];
        fc_i_momentum_y.z() = fc_momentum_y[i + 2 * nelr];

        fc_i_momentum_z.x() = fc_momentum_z[i + 0 * nelr];
        fc_i_momentum_z.y() = fc_momentum_z[i + 1 * nelr];
        fc_i_momentum_z.z() = fc_momentum_z[i + 2 * nelr];

        fc_i_density_energy.x() = fc_density_energy[i + 0 * nelr];
        fc_i_density_energy.y() = fc_density_energy[i + 1 * nelr];
        fc_i_density_energy.z() = fc_density_energy[i + 2 * nelr];

        double flux_i_density = double(0.0);
        sycl::double3 flux_i_momentum;
        flux_i_momentum.x() = double(0.0);
        flux_i_momentum.y() = double(0.0);
        flux_i_momentum.z() = double(0.0);
        double flux_i_density_energy = double(0.0);

        sycl::double3 velocity_nb;
        double density_nb, density_energy_nb;
        sycl::double3 momentum_nb;
        sycl::double3 fc_nb_momentum_x, fc_nb_momentum_y, fc_nb_momentum_z;
        sycl::double3 fc_nb_density_energy;
        double speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
                normal.x() = normals[i + (j + 0 * NNB) * nelr];
                normal.y() = normals[i + (j + 1 * NNB) * nelr];
                normal.z() = normals[i + (j + 2 * NNB) * nelr];
                normal_len = sycl::sqrt(normal.x() * normal.x() +
                                        normal.y() * normal.y() +
                                        normal.z() * normal.z());

                if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
                        momentum_nb.x() = variables[nb + (VAR_MOMENTUM + 0) * nelr];
                        momentum_nb.y() = variables[nb + (VAR_MOMENTUM + 1) * nelr];
                        momentum_nb.z() = variables[nb + (VAR_MOMENTUM + 2) * nelr];
                        density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
												compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);

                        fc_nb_momentum_x.x() = fc_momentum_x[nb + 0 * nelr];
                        fc_nb_momentum_x.y() = fc_momentum_x[nb + 1 * nelr];
                        fc_nb_momentum_x.z() = fc_momentum_x[nb + 2 * nelr];

                        fc_nb_momentum_y.x() = fc_momentum_y[nb + 0 * nelr];
                        fc_nb_momentum_y.y() = fc_momentum_y[nb + 1 * nelr];
                        fc_nb_momentum_y.z() = fc_momentum_y[nb + 2 * nelr];

                        fc_nb_momentum_z.x() = fc_momentum_z[nb + 0 * nelr];
                        fc_nb_momentum_z.y() = fc_momentum_z[nb + 1 * nelr];
                        fc_nb_momentum_z.z() = fc_momentum_z[nb + 2 * nelr];

                        fc_nb_density_energy.x() = fc_density_energy[nb + 0 * nelr];
                        fc_nb_density_energy.y() = fc_density_energy[nb + 1 * nelr];
                        fc_nb_density_energy.z() = fc_density_energy[nb + 2 * nelr];

                        // artificial viscosity
                        factor = -normal_len * smoothing_coefficient *
                                 double(0.5) *
                                 (speed_i + sycl::sqrt(speed_sqd_nb) +
                                  speed_of_sound_i + speed_of_sound_nb);
                        flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
                        flux_i_momentum.x() += factor * (momentum_i.x() - momentum_nb.x());
                        flux_i_momentum.y() += factor * (momentum_i.y() - momentum_nb.y());
                        flux_i_momentum.z() += factor * (momentum_i.z() - momentum_nb.z());

                        // accumulate cell-centered fluxes
                        factor = double(0.5) * normal.x();
                        flux_i_density += factor * (momentum_nb.x() + momentum_i.x());
                        flux_i_density_energy +=
                            factor * (fc_nb_density_energy.x() +
                                      fc_i_density_energy.x());
                        flux_i_momentum.x() += factor * (fc_nb_momentum_x.x() +
                                                         fc_i_momentum_x.x());
                        flux_i_momentum.y() += factor * (fc_nb_momentum_y.x() +
                                                         fc_i_momentum_y.x());
                        flux_i_momentum.z() += factor * (fc_nb_momentum_z.x() +
                                                         fc_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density += factor * (momentum_nb.y() + momentum_i.y());
                        flux_i_density_energy +=
                            factor * (fc_nb_density_energy.y() +
                                      fc_i_density_energy.y());
                        flux_i_momentum.x() += factor * (fc_nb_momentum_x.y() +
                                                         fc_i_momentum_x.y());
                        flux_i_momentum.y() += factor * (fc_nb_momentum_y.y() +
                                                         fc_i_momentum_y.y());
                        flux_i_momentum.z() += factor * (fc_nb_momentum_z.y() +
                                                         fc_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density += factor * (momentum_nb.z() + momentum_i.z());
                        flux_i_density_energy +=
                            factor * (fc_nb_density_energy.z() +
                                      fc_i_density_energy.z());
                        flux_i_momentum.x() += factor * (fc_nb_momentum_x.z() +
                                                         fc_i_momentum_x.z());
                        flux_i_momentum.y() += factor * (fc_nb_momentum_y.z() +
                                                         fc_i_momentum_y.z());
                        flux_i_momentum.z() += factor * (fc_nb_momentum_z.z() +
                                                         fc_i_momentum_z.z());
                }
		else if(nb == -1)	// a wing boundary
		{
                        flux_i_momentum.x() += normal.x() * pressure_i;
                        flux_i_momentum.y() += normal.y() * pressure_i;
                        flux_i_momentum.z() += normal.z() * pressure_i;
                }
		else if(nb == -2) // a far field boundary
		{
                        factor = double(0.5) * normal.x();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x());
                        flux_i_density_energy +=
                            factor * (ff_fc_density_energy[0].x() +
                                      fc_i_density_energy.x());
                        flux_i_momentum.x() +=
                            factor *
                            (ff_fc_momentum_x[0].x() + fc_i_momentum_x.x());
                        flux_i_momentum.y() +=
                            factor *
                            (ff_fc_momentum_y[0].x() + fc_i_momentum_y.x());
                        flux_i_momentum.z() +=
                            factor *
                            (ff_fc_momentum_z[0].x() + fc_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y());
                        flux_i_density_energy +=
                            factor * (ff_fc_density_energy[0].y() +
                                      fc_i_density_energy.y());
                        flux_i_momentum.x() +=
                            factor *
                            (ff_fc_momentum_x[0].y() + fc_i_momentum_x.y());
                        flux_i_momentum.y() +=
                            factor *
                            (ff_fc_momentum_y[0].y() + fc_i_momentum_y.y());
                        flux_i_momentum.z() +=
                            factor *
                            (ff_fc_momentum_z[0].y() + fc_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z());
                        flux_i_density_energy +=
                            factor * (ff_fc_density_energy[0].z() +
                                      fc_i_density_energy.z());
                        flux_i_momentum.x() +=
                            factor *
                            (ff_fc_momentum_x[0].z() + fc_i_momentum_x.z());
                        flux_i_momentum.y() +=
                            factor *
                            (ff_fc_momentum_y[0].z() + fc_i_momentum_y.z());
                        flux_i_momentum.z() +=
                            factor *
                            (ff_fc_momentum_z[0].z() + fc_i_momentum_z.z());
                }
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
        fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x();
        fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y();
        fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z();
        fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}
void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fc_momentum_x, double* fc_momentum_y, double* fc_momentum_z, double* fc_density_energy, double* fluxes)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        int error;
        /*
        DPCT1049:72: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                ff_variable.init();
                ff_fc_momentum_x.init();
                ff_fc_momentum_y.init();
                ff_fc_momentum_z.init();
                ff_fc_density_energy.init();

                auto ff_variable_ptr_ct1 = ff_variable.get_ptr();
                auto ff_fc_momentum_x_ptr_ct1 = ff_fc_momentum_x.get_ptr();
                auto ff_fc_momentum_y_ptr_ct1 = ff_fc_momentum_y.get_ptr();
                auto ff_fc_momentum_z_ptr_ct1 = ff_fc_momentum_z.get_ptr();
                auto ff_fc_density_energy_ptr_ct1 =
                    ff_fc_density_energy.get_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(Dg * Db, Db),
                    [=](sycl::nd_item<3> item_ct1) {
                            cuda_compute_flux(
                                nelr, elements_surrounding_elements, normals,
                                variables, fc_momentum_x, fc_momentum_y,
                                fc_momentum_z, fc_density_energy, fluxes,
                                item_ct1, ff_variable_ptr_ct1,
                                ff_fc_momentum_x_ptr_ct1,
                                ff_fc_momentum_y_ptr_ct1,
                                ff_fc_momentum_z_ptr_ct1,
                                ff_fc_density_energy_ptr_ct1);
                    });
        });
                    /*
                    DPCT1010:73: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    error = 0;
}

SYCL_EXTERNAL void cuda_time_step(int j, int nelr, double *old_variables,
                                  double *variables, double *step_factors,
                                  double *fluxes, sycl::nd_item<3> item_ct1)
{
        const int i =
            (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));

        double factor = step_factors[i]/double(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}
void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
        int error;
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        /*
        DPCT1049:75: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         cuda_time_step(j, nelr, old_variables,
                                                        variables, step_factors,
                                                        fluxes, item_ct1);
                                 });
        });
                    /*
                    DPCT1010:76: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    error = 0;
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];

        dpct::device_info prop;
        int dev;

    // get number of devices
    int n_dev = cl::sycl::device::get_devices(cl::sycl::info::device_type::all).size();

    for (int i = 0; i < n_dev; i++) {
        dpct::dev_mgr::instance().get_device(i).get_device_info(prop);
        std::string name = prop.get_name();
        bool is_gpu = dpct::dev_mgr::instance().get_device(i).is_gpu();
#ifdef NVIDIA_GPU
        if(is_gpu && (name.find("NVIDIA") != std::string::npos)) {
            dev = i;
            break;
        }
#elif INTEL_GPU
        if(is_gpu && (name.find("Intel(R)") != std::string::npos)) {
            dev = i;
            break;
        }
#endif
    }

        /*
        DPCT1003:78: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::dev_mgr::instance().select_device(dev), 0));
        //checkCudaErrors(dev = dpct::dev_mgr::instance().current_device_id());
        /*
        DPCT1003:79: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::dev_mgr::instance().get_device(dev).get_device_info(prop),
             0));

        printf("Name:                     %s\n", prop.get_name());

        // set far field conditions and load them into constant memory on the gpu
	{
		double h_ff_variable[NVAR];
		const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = double(1.4);
		
		double ff_pressure = double(1.0);
		double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		double ff_speed = double(ff_mach)*ff_speed_of_sound;

                sycl::double3 ff_velocity;
                ff_velocity.x() = ff_speed * double(cos((double)angle_of_attack));
                ff_velocity.y() = ff_speed * double(sin((double)angle_of_attack));
                ff_velocity.z() = 0.0;

                h_ff_variable[VAR_MOMENTUM + 0] =
                    h_ff_variable[VAR_DENSITY] * ff_velocity.x();
                h_ff_variable[VAR_MOMENTUM + 1] =
                    h_ff_variable[VAR_DENSITY] * ff_velocity.y();
                h_ff_variable[VAR_MOMENTUM + 2] =
                    h_ff_variable[VAR_DENSITY] * ff_velocity.z();

                h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(double(0.5)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0));

                sycl::double3 h_ff_momentum;
                h_ff_momentum.x() = *(h_ff_variable + VAR_MOMENTUM + 0);
                h_ff_momentum.y() = *(h_ff_variable + VAR_MOMENTUM + 1);
                h_ff_momentum.z() = *(h_ff_variable + VAR_MOMENTUM + 2);
                sycl::double3 h_ff_fc_momentum_x;
                sycl::double3 h_ff_fc_momentum_y;
                sycl::double3 h_ff_fc_momentum_z;
                sycl::double3 h_ff_fc_density_energy;
                compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_fc_momentum_x, h_ff_fc_momentum_y, h_ff_fc_momentum_z, h_ff_fc_density_energy);

		// copy far field conditions to the gpu
                /*
                DPCT1003:80: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_variable.get_ptr(), h_ff_variable,
                                 NVAR * sizeof(double))
                         .wait(),
                     0));
                /*
                DPCT1003:81: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_fc_momentum_x.get_ptr(),
                                 &h_ff_fc_momentum_x, sizeof(sycl::double3))
                         .wait(),
                     0));
                /*
                DPCT1003:82: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_fc_momentum_y.get_ptr(),
                                 &h_ff_fc_momentum_y, sizeof(sycl::double3))
                         .wait(),
                     0));
                /*
                DPCT1003:83: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_fc_momentum_z.get_ptr(),
                                 &h_ff_fc_momentum_z, sizeof(sycl::double3))
                         .wait(),
                     0));

                /*
                DPCT1003:84: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_fc_density_energy.get_ptr(),
                                 &h_ff_fc_density_energy, sizeof(sycl::double3))
                         .wait(),
                     0));
        }
	int nel;
	int nelr;
	
	// read in domain geometry
	double* areas;
	int* elements_surrounding_elements;
	double* normals;
	{
		std::ifstream file(data_file_name);
	
		file >> nel;
                nelr = block_length *
                       ((nel / block_length) + std::min(1, nel % block_length));

                double* h_areas = new double[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		double* h_normals = new double[nelr*NDIM*NNB];

				
		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> h_areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> h_elements_surrounding_elements[i + j*nelr];
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
				{
					file >> h_normals[i + (j + k*NNB)*nelr];
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		areas = alloc<double>(nelr);
		upload<double>(areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<double>(nelr*NDIM*NNB);
		upload<double>(normals, h_normals, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

	// Create arrays and set initial conditions
	double* variables = alloc<double>(nelr*NVAR);
	initialize_variables(nelr, variables);

	double* old_variables = alloc<double>(nelr*NVAR);   	
	double* fluxes = alloc<double>(nelr*NVAR);
	double* step_factors = alloc<double>(nelr); 
	double* fc_momentum_x = alloc<double>(nelr*NDIM); 
	double* fc_momentum_y = alloc<double>(nelr*NDIM);
	double* fc_momentum_z = alloc<double>(nelr*NDIM);
	double* fc_density_energy = alloc<double>(nelr*NDIM);

	// make sure all memory is doublely allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
        dpct::get_default_queue()
            .memset((void *)step_factors, 0, sizeof(double) * nelr)
            .wait();
        // make sure CUDA isn't still doing something before we start timing
        dpct::get_current_device().queues_wait_and_throw();

        // these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

	StopWatchInterface *timer = NULL;
	sdkCreateTimer( &timer);
	sdkStartTimer( &timer);

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy<double>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
		
		for(int j = 0; j < RK; j++)
		{
			compute_flux_contributions(nelr, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy);
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy, fluxes);
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
		}
	}

        dpct::get_current_device().queues_wait_and_throw();
        sdkStopTimer(&timer);  

	std::cout  << (sdkGetAverageTimerValue(&timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	
	std::cout << "Cleaning up..." << std::endl;
	dealloc<double>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<double>(normals);
	
	dealloc<double>(variables);
	dealloc<double>(old_variables);
	dealloc<double>(fluxes);
	dealloc<double>(step_factors);
	dealloc<double>(fc_momentum_x); 
	dealloc<double>(fc_momentum_y);
	dealloc<double>(fc_momentum_z);
	dealloc<double>(fc_density_energy);

	std::cout << "Done..." << std::endl;

	return 0;
}
