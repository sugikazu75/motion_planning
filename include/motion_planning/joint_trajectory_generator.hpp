#pragma once

#include <pinocchio/fwd.hpp>
#include <motion_planning/inverse_kinematics_3d.hpp>
#include <vector>

namespace motion_planning
{
bool jointTrajectory(const pinocchio::Model& model, pinocchio::Data& data, const pinocchio::FrameIndex frame_id,
                     const std::vector<Eigen::Vector3d>& x_des, std::vector<Eigen::VectorXd>& q_traj,
                     const Eigen::VectorXd& q_init, bool debug = false)
{
  Eigen::VectorXd q = q_init;
  q_traj.resize(x_des.size());

  bool success = true;
  for (size_t i = 0; i < x_des.size(); ++i)
  {
    const Eigen::Vector3d& target = x_des.at(i);
    success = motion_planning::solveIK(model, data, frame_id, target, q, q_traj.at(i), false, 1000, 1e-4);

    if (!success)
    {
      if (debug)
        std::cout << "IK solution not found for target: " << target.transpose() << std::endl;
      success = false;
    }
    else
    {
      if (debug)
        std::cout << "IK solution found for target: " << target.transpose() << " -> " << q_traj.at(i).transpose()
                  << std::endl;
    }
    q = q_traj.at(i);  // Update q for the next iteration
  }

  return success;  // All IK solutions found successfully
}
}  // namespace motion_planning
