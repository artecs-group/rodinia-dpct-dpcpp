// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

// #include <cutil.h>
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

#ifdef TIME_IT
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

template <typename T>
#ifdef TIME_IT
T* alloc(int N, long long &time)
#else
T* alloc(int N)
#endif
{
	T* t;
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1003:17: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((t = (T *)sycl::malloc_device(
                             sizeof(T) * N, dpct::get_default_queue()),
                         0));
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        time = time1-time0;
        #endif
        return t;
}

template <typename T>
#ifdef TIME_IT
long long dealloc(T* array)
#else
void dealloc(T* array)
#endif
{
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1003:18: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((sycl::free((void *)array, dpct::get_default_queue()), 0));
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        return time1-time0;
        #endif
}

template <typename T>
#ifdef TIME_IT
long long copy(T* dst, T* src, int N)
#else
void copy(T* dst, T* src, int N)
#endif
{
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1003:19: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
        #ifdef TIME_IT
        time1 = get_time();
        return time1-time0;
        #endif
}

template <typename T>
#ifdef TIME_IT
long long upload(T* dst, T* src, int N)
#else
void upload(T* dst, T* src, int N)
#endif
{
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1003:20: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
        #ifdef TIME_IT
        time1 = get_time();
        return time1-time0;
        #endif
}

template <typename T>
#ifdef TIME_IT
long long download(T* dst, T* src, int N)
#else
void download(T* dst, T* src, int N)
#endif
{
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1003:21: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::get_default_queue()
                             .memcpy((void *)dst, (void *)src, N * sizeof(T))
                             .wait(),
                         0));
        #ifdef TIME_IT
        time1 = get_time();
        return time1-time0;
        #endif
}
#ifdef TIME_IT
long long dump(double* variables, int nel, int nelr)
#else
void dump(double* variables, int nel, int nelr)
#endif
{
	double* h_variables = new double[nelr*NVAR];
        #ifdef TIME_IT
        long long time =
        #endif
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

        #ifdef TIME_IT
        return time;
        #endif
}

/*
 * Element-based Cell-centered FVM solver functions
 */
dpct::constant_memory<double, 1> ff_variable(NVAR);
dpct::constant_memory<sycl::double3, 1> ff_flux_contribution_momentum_x(1);
dpct::constant_memory<sycl::double3, 1> ff_flux_contribution_momentum_y(1);
dpct::constant_memory<sycl::double3, 1> ff_flux_contribution_momentum_z(1);
dpct::constant_memory<sycl::double3, 1> ff_flux_contribution_density_energy(1);

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

#ifdef TIME_IT
long long initialize_variables(int nelr, double* variables)
#else
void initialize_variables(int nelr, double* variables)
#endif
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);

        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1049:22: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<double, 1> ff_variable;

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
        DPCT1010:23: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        return time1-time0;
        #endif
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

#ifdef TIME_IT
long long compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
#else
void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
#endif
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1049:25: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         cuda_compute_step_factor(
                                             nelr, variables, areas,
                                             step_factors, item_ct1);
                                 });
        });
        /*
        DPCT1010:26: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        return time1-time0;
        #endif
}

/*
 *
 *
*/
void cuda_compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes,
                       sycl::nd_item<3> item_ct1, double *ff_variable,
                       sycl::double3 *ff_flux_contribution_momentum_x,
                       sycl::double3 *ff_flux_contribution_momentum_y,
                       sycl::double3 *ff_flux_contribution_momentum_z,
                       sycl::double3 *ff_flux_contribution_density_energy)
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
        sycl::double3 flux_contribution_i_momentum_x,
            flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
        sycl::double3 flux_contribution_i_density_energy;
        compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	double flux_i_density = double(0.0);
        sycl::double3 flux_i_momentum;
        flux_i_momentum.x() = double(0.0);
        flux_i_momentum.y() = double(0.0);
        flux_i_momentum.z() = double(0.0);
        double flux_i_density_energy = double(0.0);

        sycl::double3 velocity_nb;
        double density_nb, density_energy_nb;
        sycl::double3 momentum_nb;
        sycl::double3 flux_contribution_nb_momentum_x,
            flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
        sycl::double3 flux_contribution_nb_density_energy;
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
			                                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
			
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
                            factor * (flux_contribution_nb_density_energy.x() +
                                      flux_contribution_i_density_energy.x());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.x() +
                                      flux_contribution_i_momentum_x.x());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.x() +
                                      flux_contribution_i_momentum_y.x());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.x() +
                                      flux_contribution_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density += factor * (momentum_nb.y() + momentum_i.y());
                        flux_i_density_energy +=
                            factor * (flux_contribution_nb_density_energy.y() +
                                      flux_contribution_i_density_energy.y());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.y() +
                                      flux_contribution_i_momentum_x.y());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.y() +
                                      flux_contribution_i_momentum_y.y());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.y() +
                                      flux_contribution_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density += factor * (momentum_nb.z() + momentum_i.z());
                        flux_i_density_energy +=
                            factor * (flux_contribution_nb_density_energy.z() +
                                      flux_contribution_i_density_energy.z());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.z() +
                                      flux_contribution_i_momentum_x.z());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.z() +
                                      flux_contribution_i_momentum_y.z());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.z() +
                                      flux_contribution_i_momentum_z.z());
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
                            factor *
                            (ff_flux_contribution_density_energy[0].x() +
                             flux_contribution_i_density_energy.x());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].x() +
                                      flux_contribution_i_momentum_x.x());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].x() +
                                      flux_contribution_i_momentum_y.x());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].x() +
                                      flux_contribution_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y());
                        flux_i_density_energy +=
                            factor *
                            (ff_flux_contribution_density_energy[0].y() +
                             flux_contribution_i_density_energy.y());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].y() +
                                      flux_contribution_i_momentum_x.y());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].y() +
                                      flux_contribution_i_momentum_y.y());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].y() +
                                      flux_contribution_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z());
                        flux_i_density_energy +=
                            factor *
                            (ff_flux_contribution_density_energy[0].z() +
                             flux_contribution_i_density_energy.z());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].z() +
                                      flux_contribution_i_momentum_x.z());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].z() +
                                      flux_contribution_i_momentum_y.z());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].z() +
                                      flux_contribution_i_momentum_z.z());
                }
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
        fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x();
        fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y();
        fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z();
        fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}

#ifdef TIME_IT
long long compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
#else
void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
#endif
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif

        /*
        DPCT1049:28: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                ff_variable.init();
                ff_flux_contribution_momentum_x.init();
                ff_flux_contribution_momentum_y.init();
                ff_flux_contribution_momentum_z.init();
                ff_flux_contribution_density_energy.init();

                auto ff_variable_ptr_ct1 = ff_variable.get_ptr();
                auto ff_flux_contribution_momentum_x_ptr_ct1 =
                    ff_flux_contribution_momentum_x.get_ptr();
                auto ff_flux_contribution_momentum_y_ptr_ct1 =
                    ff_flux_contribution_momentum_y.get_ptr();
                auto ff_flux_contribution_momentum_z_ptr_ct1 =
                    ff_flux_contribution_momentum_z.get_ptr();
                auto ff_flux_contribution_density_energy_ptr_ct1 =
                    ff_flux_contribution_density_energy.get_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(Dg * Db, Db),
                    [=](sycl::nd_item<3> item_ct1) {
                            cuda_compute_flux(
                                nelr, elements_surrounding_elements, normals,
                                variables, fluxes, item_ct1,
                                ff_variable_ptr_ct1,
                                ff_flux_contribution_momentum_x_ptr_ct1,
                                ff_flux_contribution_momentum_y_ptr_ct1,
                                ff_flux_contribution_momentum_z_ptr_ct1,
                                ff_flux_contribution_density_energy_ptr_ct1);
                    });
        });
        /*
        DPCT1010:29: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        return time1-time0;
        #endif
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
#ifdef TIME_IT
long long time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
#else
void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
#endif
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);

        #ifdef TIME_IT
  	long long time1;
	long long time0 = get_time();
        #endif
        /*
        DPCT1049:31: The workgroup size passed to the SYCL kernel may exceed the
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
        DPCT1010:32: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;
        #ifdef TIME_IT
        dpct::get_current_device().queues_wait_and_throw();
        time1 = get_time();
        return time1-time0;
        #endif
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
    #ifdef TIME_IT
    long long initTime = 0;
    long long alocTime = 0;
    long long cpInTime = 0;
    long long kernTime = 0;
    long long cpOtTime = 0;
    long long freeTime = 0;
    long long auxTime1 = 0;
    long long auxTime2 = 0;
    #endif
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];

        dpct::device_info prop;
        int dev;
	
	// CUDA_SAFE_CALL(cudaSetDevice(0));
	// CUDA_SAFE_CALL(cudaGetDevice(&dev));
	// CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));

        /*
        DPCT1003:34: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        #ifdef TIME_IT
        auxTime1 = get_time();
        checkCudaErrors((dpct::dev_mgr::instance().select_device(0), 0));
        checkCudaErrors(dev = dpct::dev_mgr::instance().current_device_id());
        auxTime2 = get_time();
        initTime = auxTime2-auxTime1;
        #else
        checkCudaErrors((dpct::dev_mgr::instance().select_device(0), 0));
        checkCudaErrors(dev = dpct::dev_mgr::instance().current_device_id());
        #endif
        /*
        DPCT1003:35: Migrated API does not return error code. (*, 0) is
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
                sycl::double3 h_ff_flux_contribution_momentum_x;
                sycl::double3 h_ff_flux_contribution_momentum_y;
                sycl::double3 h_ff_flux_contribution_momentum_z;
                sycl::double3 h_ff_flux_contribution_density_energy;
                compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
                /*
                DPCT1003:36: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                #ifdef TIME_IT
                auxTime1 = get_time();
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_variable.get_ptr(), h_ff_variable,
                                 NVAR * sizeof(double))
                         .wait(),
                     0));
                auxTime2 = get_time();
                cpInTime += auxTime2-auxTime1;

                auxTime1 = get_time();
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_x.get_ptr(),
                                 &h_ff_flux_contribution_momentum_x,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                auxTime2 = get_time();
                cpInTime += auxTime2-auxTime1;

                auxTime1 = get_time();
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_y.get_ptr(),
                                 &h_ff_flux_contribution_momentum_y,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                auxTime2 = get_time();
                cpInTime += auxTime2-auxTime1;

                auxTime1 = get_time();
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_z.get_ptr(),
                                 &h_ff_flux_contribution_momentum_z,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                auxTime2 = get_time();
                cpInTime += auxTime2-auxTime1;

                auxTime1 = get_time();
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_density_energy.get_ptr(),
                                 &h_ff_flux_contribution_density_energy,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                auxTime2 = get_time();
                cpInTime += auxTime2-auxTime1;
                #else


                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_variable.get_ptr(), h_ff_variable,
                                 NVAR * sizeof(double))
                         .wait(),
                     0));
                /*
                DPCT1003:37: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_x.get_ptr(),
                                 &h_ff_flux_contribution_momentum_x,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                /*
                DPCT1003:38: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_y.get_ptr(),
                                 &h_ff_flux_contribution_momentum_y,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                /*
                DPCT1003:39: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_momentum_z.get_ptr(),
                                 &h_ff_flux_contribution_momentum_z,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));

                /*
                DPCT1003:40: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(ff_flux_contribution_density_energy.get_ptr(),
                                 &h_ff_flux_contribution_density_energy,
                                 sizeof(sycl::double3))
                         .wait(),
                     0));
                #endif
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
		#ifdef TIME_IT
		areas = alloc<double>(nelr, auxTime1);
                alocTime += auxTime1;
        
                cpInTime += upload<double>(areas, h_areas, nelr);

                elements_surrounding_elements = alloc<int>(nelr*NNB, auxTime1);
                alocTime += auxTime1;
		cpInTime += upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<double>(nelr*NDIM*NNB, auxTime1);
                alocTime += auxTime1;
		cpInTime += upload<double>(normals, h_normals, nelr*NDIM*NNB);
                #else
		areas = alloc<double>(nelr);
		upload<double>(areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<double>(nelr*NDIM*NNB);
		upload<double>(normals, h_normals, nelr*NDIM*NNB);
                #endif

		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

        #ifdef TIME_IT
        double* variables = alloc<double>(nelr*NVAR, auxTime1);
        alocTime += auxTime1;

        kernTime += initialize_variables(nelr, variables);

        double* old_variables = alloc<double>(nelr*NVAR, auxTime1); 
        alocTime += auxTime1;

                double* fluxes = alloc<double>(nelr*NVAR, auxTime1);
        alocTime += auxTime1;

                double* step_factors = alloc<double>(nelr, auxTime1);
        alocTime += auxTime1;

        kernTime += initialize_variables(nelr, old_variables);
        kernTime += initialize_variables(nelr, fluxes);

        dpct::get_default_queue()
                .memset((void *)step_factors, 0, sizeof(double) * nelr)
                .wait();
        // make sure CUDA isn't still doing something before we start timing
        dpct::get_current_device().queues_wait_and_throw();

        #else
	// Create arrays and set initial conditions
	double* variables = alloc<double>(nelr*NVAR);
	initialize_variables(nelr, variables);

	double* old_variables = alloc<double>(nelr*NVAR);   	
	double* fluxes = alloc<double>(nelr*NVAR);
	double* step_factors = alloc<double>(nelr); 

	// make sure all memory is doublely allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
        dpct::get_default_queue()
            .memset((void *)step_factors, 0, sizeof(double) * nelr)
            .wait();
        // make sure CUDA isn't still doing something before we start timing
        dpct::get_current_device().queues_wait_and_throw();

        #endif

        // these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

        int error;
        StopWatchInterface *timer = NULL;

	sdkCreateTimer( &timer);
	sdkStartTimer( &timer);

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
                #ifdef TIME_IT
                copy<double>(old_variables, variables, nelr*NVAR);
                kernTime += compute_step_factor(nelr, variables, areas, step_factors);

                for(int j = 0; j < RK; j++)
                        {
                                kernTime += compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
                                kernTime += time_step(j, nelr, old_variables, variables, step_factors, fluxes);
                        }
                #else
		copy<double>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
                /*
                DPCT1010:41: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                error = 0;

                for(int j = 0; j < RK; j++)
		  {
		    compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
                    /*
                    DPCT1010:43: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    error = 0;

                    time_step(j, nelr, old_variables, variables, step_factors, fluxes);
                    /*
                    DPCT1010:45: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    error = 0;
                  }
                  #endif
	}

        dpct::get_current_device().queues_wait_and_throw();
        sdkStopTimer(&timer);  

	std::cout  << (sdkGetAverageTimerValue(&timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;

	std::cout << "Saving solution..." << std::endl;
        #ifdef TIME_IT
	cpOtTime += 
        #endif
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	
	std::cout << "Cleaning up..." << std::endl;
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(areas);
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<int>(elements_surrounding_elements);
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(normals);
	
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(variables);
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(old_variables);
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(fluxes);
        #ifdef TIME_IT
	freeTime += 
        #endif
	dealloc<double>(step_factors);

	std::cout << "Done..." << std::endl;

        #ifdef TIME_IT
        long long totalTime = initTime + alocTime + cpInTime + kernTime + cpOtTime + freeTime;
	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) initTime / 1000000, (float) initTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) alocTime / 1000000, (float) alocTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) cpInTime / 1000000, (float) cpInTime / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) kernTime / 1000000, (float) kernTime / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) cpOtTime / 1000000, (float) cpOtTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) freeTime / 1000000, (float) freeTime / (float) totalTime * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) totalTime / 1000000);
	#endif

	return 0;
}
