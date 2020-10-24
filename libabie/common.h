#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <math.h>
#else
#include <tgmath.h>
#endif
#include <float.h>

#ifndef real
    #ifdef LONGDOUBLE
        #define real long double
    #else
        #define real double
    #endif
#endif

#ifdef WIN32
#ifdef LIBABIE_EXPORTS
#define ABIELIBRARY_API __declspec(dllexport)
#else
#define ABIELIBRARY_API __declspec(dllimport)
#endif
#else
#define ABIELIBRARY_API
#endif

#ifndef real
#ifdef LONGDOUBLE
#define real long double
#else
#define real double
#endif
#endif

#define USE_PARALLEL 256

// Common variables
real *pos_global;  // the position state vector specified by the users
real *vel_global;  // the velocity state vector specified by the users
real *m_vec_global; // masses specified by the users
real *r_vec_global; // radii specified by the users

real *ext_acc_global; // externally calculated acceleration terms for each body
// extern real *y00;  // the position state vector used internally by the integrator
// extern real *dy0;  // the velocity state vector used internally by the integrator
// extern real *masses; // masses used internally by the integrator

size_t N_global; // number of particles
// extern size_t dim; // usually 3 * N
// extern size_t N_active; // number of massive particles
real G_global; // gravitational constant
real C_global; // speed of light constant (for PN calculations)
// extern real *eps; // per-particle softening parameter
// extern real h; // the internal time step used by the integrator
// extern real h_min; // minimum time step
real t_global;
// extern real t_end;
extern double dt; // the time step specified by the user
real close_encounter_distance;  // if 0, ignore close encounters
size_t n_close_encounters; // number of close encounters
size_t n_collisions; // number of collision events

// constants and flags
size_t MAX_N_CE; // number of close encounters before it stops integrating (will lead the Python framework to generate an exception)
size_t MAX_N_COLLISIONS; // number of close encounters before it stops integrating (will lead the Python framework to generate an exception)
size_t EXIT_MAX_N_CE_EXCEEDED;
size_t EXIT_MAX_N_COLLISIONS_EXCEEDED;
size_t EXIT_NORMAL;
size_t ENABLE_EXT_ACC; // enable the externally calculated accelerations

// buffer for storing close encounter events and collision events
// format: [time1, id1_event1, id2_event1, distance_event1, time2, id1_event2, id2_event2, distance_event2, ...]
real *buf_ce_events;
real *buf_collision_events;

// Getters/Setters
ABIELIBRARY_API void set_state(double *pos_vec, double *vel_vec, double *m_vec, double *r_vec, int N, double G, double C);
ABIELIBRARY_API int get_state(double *pos_vec, double *vel_vec, double *m_vec, double *r_vec);
ABIELIBRARY_API double get_model_time();
ABIELIBRARY_API void set_close_encounter_distance(double d);
ABIELIBRARY_API double get_close_encounter_distance();
ABIELIBRARY_API void set_close_encounter_buffer(double *buf_ce, int max_n_ce);
ABIELIBRARY_API void set_collision_buffer(double *buf_collision, int max_n_collision);
ABIELIBRARY_API void get_collision_buffer(double* buf_collision);

ABIELIBRARY_API void reset_close_encounter_buffer(); // should be called after the python interface finishes handling a close encounter exception
ABIELIBRARY_API void reset_collision_buffer(); // should be called after the python interface finishes handling a collision exception

// set the addtional forces calculated by external routines (e.g., in the python interface)
ABIELIBRARY_API size_t set_additional_forces(int N, double ext_acc[]);

// Utility functions
size_t ode_n_body_first_order(real *pos, size_t N, real G, const real *masses, real *dxdt);
size_t ode_n_body_second_order(const real *pos, size_t N, real G, const real *masses, const real *radii, real *acc);
// size_t ode_n_body_second_order_sapporo(const real *pos, size_t N, real G, const real *masses, const real *radii, real *acc);
size_t calculate_accelerations(const real pos[], const real vel[], size_t N, real G, const real masses[], const real radii[], real acc[]);

// Additonal forces
size_t calculate_additional_forces(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]);
size_t calculate_post_newtonian(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]);

#ifdef GPU
size_t ode_n_body_second_order_gpu(const real *vec, size_t N, real G, const real *masses, const real *radii, real *acc);
void gpu_init(int N);
void gpu_finalize();
#endif

size_t check_collisions_close_encounters(const real *vec, const real radii[], size_t N, real t);
real *vec_scalar_op(const real *vec, real scalar, size_t N, char op);
real *vec_vec_op(const real *vec1, real *vec2, size_t N, char op);
real vector_max_abs(const real *vec, size_t N);
real vector_norm(const real *vec, size_t N);
real sign(real x);
real cross_norm(const real *vec1, const real *vec2); // the magnitude of the cross product of two 3D vectors
real dot(const real *vec1, const real *vec2); // the dot product of two 3D vectors

// Integrator functions
ABIELIBRARY_API int initialize_code(double _G, double _C, int _N_MAX, int _MAX_N_CE, int _MAX_N_COLLISIONS);
ABIELIBRARY_API int finalize_code();

ABIELIBRARY_API size_t integrator_gauss_radau15(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real _G, real _t, real _t_end, real _dt);
ABIELIBRARY_API int integrator_gr(double t, double t_end, double dt);
ABIELIBRARY_API void integrator_runge_kutta(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real G, double _t, double _t_end, double _dt);
ABIELIBRARY_API int integrator_rk(double t, double t_end, double dt);
ABIELIBRARY_API void integrator_wisdom_holman(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real _G, real _t, real _t_end, real _dt);
ABIELIBRARY_API int integrator_wh(double t, double t_end, double dt);

ABIELIBRARY_API double calculate_energy();
ABIELIBRARY_API double calculate_energy_supplied(double* pos_vec, double* vel_vec, double* m_vec, int N, double G);

#endif
