#pragma once

#include <pinocchio/fwd.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <aerial_robot_control/trajectory/trajectory_reference/polynomial.hpp>
#include <vector>

namespace motion_planning
{
void posTrajectoryGenerator(const Eigen::Vector3d x_init, const Eigen::Vector3d x_final,

                            std::vector<Eigen::Vector3d>& x_traj, double duration, double dt = 0.01)
{
  agi::Polynomial<Eigen::HouseholderQR<agi::Matrix<>>> poly_x(11, agi::Vector<3>(0, 0, 1), 2);  // min-jerk
  agi::Polynomial<Eigen::HouseholderQR<agi::Matrix<>>> poly_y(11, agi::Vector<3>(0, 0, 1), 2);  // min-jerk
  agi::Polynomial<Eigen::HouseholderQR<agi::Matrix<>>> poly_z(11, agi::Vector<3>(0, 0, 1), 2);  // min-jerk

  int num_samples = static_cast<int>(duration / dt);
  x_traj.resize(num_samples);
  Eigen::VectorXd times(num_samples);
  for (int i = 0; i < num_samples; ++i)
    times(i) = i * duration / (double)(num_samples - 1);

  poly_x.scale(0.0, duration);
  poly_y.scale(0.0, duration);
  poly_z.scale(0.0, duration);

  // initial state. position, velocity and acceleration
  poly_x.addConstraint(0, agi::Vector<3>(x_init(0), 0.0, 0.0));
  poly_y.addConstraint(0, agi::Vector<3>(x_init(1), 0.0, 0.0));
  poly_z.addConstraint(0, agi::Vector<3>(x_init(2), 0.0, 0.0));

  // final state. position, velocity and acceleration
  poly_x.addConstraint(duration, agi::Vector<3>(x_final(0), 0.0, 0.0));
  poly_y.addConstraint(duration, agi::Vector<3>(x_final(1), 0.0, 0.0));
  poly_z.addConstraint(duration, agi::Vector<3>(x_final(2), 0.0, 0.0));

  // evaluate
  poly_x.solve();
  poly_y.solve();
  poly_z.solve();

  for (int i = 0; i < num_samples; ++i)
  {
    double t = times(i);
    Eigen::Vector3d x, y, z;
    poly_x.eval(t, x);
    poly_y.eval(t, y);
    poly_z.eval(t, z);
    x_traj.at(i) << x(0), y(0), z(0);
  }
}
}  // namespace motion_planning
