#ifndef _SCTL_ODE_SOLVER_HPP_
#define _SCTL_ODE_SOLVER_HPP_

#include <functional>       // for function

#include "sctl/common.hpp"  // for Integer, sctl
#include "sctl/comm.hpp"    // for Comm
#include "sctl/comm.txx"    // for Comm::Self
#include "sctl/matrix.hpp"  // for Matrix
#include "sctl/vector.hpp"  // for Vector

namespace sctl {

/**
 * Implements spectral deferred correction (SDC) solver for ordinary differential equations (ODEs).
 */
template <class Real> class SDC {
  public:

    /// The function type to specify the RHS of the ODE.
    using Fn0 = std::function<void(Vector<Real>* dudt, const Vector<Real>& u, const Integer correction_idx, const Integer substep_idx)>;

    /// The function type to specify the RHS of the ODE.
    using Fn1 = std::function<void(Vector<Real>* dudt, const Vector<Real>& u)>;

    /// Callback function type.
    using MonitorFn = std::function<void(Real t, Real dt, const Vector<Real>& u)>;

    /**
     * Constructor
     *
     * @param[in] order the order of the method.
     * @param[in] comm the communicator.
     */
    explicit SDC(const Integer order, const Comm& comm = Comm::Self());

    /**
     * @return order of the method.
     */
    Integer Order() const;

    /**
     * Apply one step of spectral deferred correction (SDC).
     * Compute: \f$ u = u_0 + \int_0^{dt} F(u) \f$
     *
     * @param[out] u the solution
     * @param[in] dt the step size
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] N_picard the maximum number of Picard iterations
     * @param[in] tol_picard the tolerance for stopping Picard iterations
     * @param[out] error_interp an estimate of the truncation error of the solution interpolant
     * @param[out] error_picard the Picard iteration error
     * @param[out] norm_dudt maximum norm of du/dt
     * @param[out] iter_count number of Picard iterations
     */
    void operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn0& F, Integer N_picard = -1, const Real tol_picard = 0, Real* error_interp = nullptr, Real* error_picard = nullptr, Integer* iter_count = nullptr, Matrix<Real>* u_substep = nullptr) const;

    /**
     * Apply one step of spectral deferred correction (SDC).
     * Compute: \f$ u = u_0 + \int_0^{dt} F(u) \f$
     *
     * @param[out] u the solution
     * @param[in] dt the step size
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] N_picard the maximum number of Picard iterations
     * @param[in] tol_picard the tolerance for stopping Picard iterations
     * @param[out] error_interp an estimate of the truncation error of the solution interpolant
     * @param[out] error_picard the Picard iteration error
     * @param[out] norm_dudt maximum norm of du/dt
     * @param[out] iter_count number of Picard iterations on exit (or -1 if terminated)
     */
    void operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn1& F, Integer N_picard = -1, const Real tol_picard = 0, Real* error_interp = nullptr, Real* error_picard = nullptr, Integer* iter_count = nullptr, Matrix<Real>* u_substep = nullptr) const;

    /**
     * Solve ODE adaptively to required tolerance.
     * Compute: \f$ u = u_0 + \int_0^{T} F(u) \f$
     *
     * @param[out] u the final solution
     * @param[in] dt the initial step size guess
     * @param[in] T the final time
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] tol the required solution tolerance
     * @param[in] monitor_callback a callback function called after each accepted time-step
     * @param[in] continue_with_errors tries to compute the best solution even if the required tolerance cannot be satisfied.
     * @param[out] error estimate of the final output error
     *
     * @return the final time (should equal T if no errors)
     */
    Real AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn0& F, Real tol, const MonitorFn* monitor_callback = nullptr, bool continue_with_errors = false, Real* error = nullptr) const;

    /**
     * Solve ODE adaptively to required tolerance.
     * Compute: \f$ u = u_0 + \int_0^{T} F(u) \f$
     *
     * @param[out] u the final solution
     * @param[in] dt the initial step size guess
     * @param[in] T the final time
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] tol the required solution tolerance
     * @param[in] monitor_callback a callback function called after each accepted time-step
     * @param[in] continue_with_errors tries to compute the best solution even if the required tolerance cannot be satisfied.
     * @param[out] error estimate of the final output error
     *
     * @return the final time (should equal T if no errors)
     */
    Real AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn1& F, Real tol, const MonitorFn* monitor_callback = nullptr, bool continue_with_errors = false, Real* error = nullptr) const;

    /**
     * This is an example for how to use the SDC class.
     */
    static void test_one_step(const Integer Order = 5);

    /**
     * This example shows adaptive time-stepping with the SDC class.
     */
    static void test_adaptive_solve(const Integer Order = 5, const Real tol = 1e-5);

  private:

    template <class Container> Real max_norm(const Container& M) const;

    Matrix<Real> M_time_step, M_error, M_error_half;
    Vector<Real> nds;
    Integer order;
    Comm comm;
};

}

#endif // _SCTL_ODE_SOLVER_HPP_
