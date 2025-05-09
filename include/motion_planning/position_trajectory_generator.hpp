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

void minJerkInterpolationPosVelAcc(const std::vector<Eigen::VectorXd> x_traj, double duration,
                                   std::vector<Eigen::VectorXd>& pos_traj, std::vector<Eigen::VectorXd>& vel_traj,
                                   std::vector<Eigen::VectorXd>& acc_traj)
{
  int x_size = x_traj.at(0).size();
  int num_samples = x_traj.size();

  if (num_samples < 2)
    return;

  std::vector<double> times(num_samples);
  for (int i = 0; i < num_samples; ++i)
    times[i] = i * duration / (num_samples - 1);

  pos_traj.resize(num_samples, Eigen::VectorXd(x_size));
  vel_traj.resize(num_samples, Eigen::VectorXd(x_size));
  acc_traj.resize(num_samples, Eigen::VectorXd(x_size));

  for (int i = 0; i < x_size; i++)
  {
    agi::Polynomial<Eigen::HouseholderQR<agi::Matrix<>>> poly(11, agi::Vector<3>(0, 0, 1), 2);  // min-jerk

    poly.scale(0, duration);

    // initial state
    poly.addConstraint(0, agi::Vector<3>(x_traj.at(0)(i), 0.0, 0.0));

    // mid states
    for (int j = 1; j < num_samples - 1; j++)
      poly.addConstraint(times.at(j), agi::Vector<1>(x_traj.at(j)(i)));

    // final state
    poly.addConstraint(duration, agi::Vector<3>(x_traj.at(num_samples - 1)(i), 0.0, 0.0));

    poly.solve();

    for (int j = 0; j < num_samples; j++)
    {
      double t = times.at(j);
      Eigen::Vector3d res;
      poly.eval(t, res);

      pos_traj.at(j)(i) = res(0);
      vel_traj.at(j)(i) = res(1);
      acc_traj.at(j)(i) = res(2);
    }
  }
}
}  // namespace motion_planning
