#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <Eigen/Dense>
#include <iostream>

namespace motion_planning
{
bool solveIK(const pinocchio::Model& model, pinocchio::Data& data, const pinocchio::FrameIndex frame_id,
             const Eigen::Vector3d& x_des, const Eigen::VectorXd& q_init, Eigen::VectorXd& q_out, int max_iter = 100,
             double eps = 1e-4)
{
  const double DT = 1.0;
  const double damp = 1e-12;
  Eigen::VectorXd q = q_init;
  for (int i = 0; i < max_iter; ++i)
  {
    pinocchio::framesForwardKinematics(model, data, q);
    Eigen::Vector3d x_curr = data.oMf[frame_id].translation();
    Eigen::Vector3d err = x_des - x_curr;  // in world frame
    // std::cout << "Iteration " << i << std::endl;
    // std::cout << "q = " << std::endl;
    // std::cout << q.transpose() << std::endl;
    // std::cout << x_curr.transpose() << std::endl;
    // std::cout << "error:" << std::endl;
    // std::cout << err.transpose() << std::endl;
    // std::cout << std::endl;

    if (err.norm() < eps)
    {
      q_out = q;
      return true;
    }

    Eigen::MatrixXd J6 = Eigen::MatrixXd::Zero(6, model.nv);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::WORLD, J6);  // world frame
    Eigen::MatrixXd J = J6.topRows(3).rightCols(model.njoints - 2);                   // position. universe and root
    // std::cout << "jacobian:" << std::endl;
    // std::cout << J << std::endl;
    Eigen::MatrixXd JJt;
    JJt.noalias() = J * J.transpose() + damp * Eigen::MatrixXd::Identity(3, 3);
    // std::cout << "JJt" << std::endl;
    // std::cout << JJt << std::endl;
    Eigen::MatrixXd JJt_inv = JJt.inverse();
    // std::cout << "JJt_inv" << std::endl;
    // std::cout << JJt_inv << std::endl;

    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    v.tail(model.njoints - 2) = J.transpose() * JJt.ldlt().solve(err);
    // std::cout << "v:" << std::endl;
    // std::cout << v.transpose() << std::endl;
    q = pinocchio::integrate(model, q, v * DT);

    // clamp joint angles to limits
    for (int i = 0; i < model.nq; ++i)
    {
      q(i) = std::max(model.lowerPositionLimit(i), std::min(q(i), model.upperPositionLimit(i)));
    }

    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
  }
  return false;
}

}  // namespace motion_planning
