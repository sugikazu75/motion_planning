#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <Eigen/Dense>
#include <iostream>

namespace motion_planning
{
inline bool solveIK(const pinocchio::Model& model, pinocchio::Data& data, const pinocchio::FrameIndex frame_id,
                    const Eigen::Vector3d& x_des, const Eigen::VectorXd& q_init, Eigen::VectorXd& q_out,
                    bool debug = false, int max_iter = 100, double eps = 1e-4)
{
  const double damp = 1e-12;
  Eigen::VectorXd q = q_init;
  q_out = q_init;

  const pinocchio::SE3 oMdes(Eigen::Matrix3d::Identity(), x_des);

  bool is_floating_base = model.joints[1].shortname() == "JointModelFreeFlyer" ? true : false;
  if (debug)
  {
    std::cout << "is_floating_base: " << is_floating_base << std::endl;
    std::cout << "q_init:" << std::endl;
    std::cout << q.transpose() << std::endl;
  }

  // IK iteration
  for (int i = 0; i < max_iter; ++i)
  {
    pinocchio::framesForwardKinematics(model, data, q);
    const pinocchio::SE3 iMd = data.oMf[frame_id].actInv(oMdes);
    Eigen::Vector3d err = iMd.translation();  // in joint frame

    Eigen::Vector3d x_curr = data.oMf[frame_id].translation();

    if (debug)
    {
      std::cout << "Iteration " << i << std::endl;
      std::cout << "q = " << std::endl;
      std::cout << q.transpose() << std::endl;
      std::cout << "x_curr:" << std::endl;
      std::cout << x_curr.transpose() << std::endl;
      std::cout << "error:" << std::endl;
      std::cout << err.transpose() << std::endl;
    }

    if (err.norm() < eps)
    {
      q_out = q;
      return true;
    }

    Eigen::MatrixXd J6 = Eigen::MatrixXd::Zero(6, model.nv);
    pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL, J6);  // local frame
    Eigen::MatrixXd J;
    int n_joint = model.nv;
    if (is_floating_base)
      n_joint -= 6;
    J = J6.topRows(3).rightCols(n_joint);  // position. joint part

    // calculate v without weight
    Eigen::MatrixXd JJt;
    JJt.noalias() = J * J.transpose() + damp * Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    v.tail(J.cols()) = J.transpose() * JJt.ldlt().solve(err);

    // calculate weight matrix to avoid joint limit
    Eigen::VectorXd upper_limit = model.upperPositionLimit;
    Eigen::VectorXd lower_limit = model.lowerPositionLimit;
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n_joint);
    int q_offset = is_floating_base ? 7 : 0;
    int v_offset = is_floating_base ? 6 : 0;
    double w_weight = 25.0;
    for (int i = 0; i < n_joint; i++)
    {
      double q_i = q(i + q_offset);
      double v_i = v(i + v_offset);
      double q_mid = 0.5 * (lower_limit(i + q_offset) + upper_limit(i + q_offset));
      double range = upper_limit(i + q_offset) - lower_limit(i + q_offset);
      double norm = 2.0 * (q_i - q_mid) / range;
      if ((q_mid < q_i && 0 < v_i) || (q_i < q_mid && v_i < 0))
        w(i) = 1.0 + w_weight * norm * norm;
    }
    Eigen::MatrixXd W_inv = w.cwiseInverse().asDiagonal();

    // calculate weighted v
    Eigen::MatrixXd JWJt = J * W_inv * J.transpose() + damp * Eigen::MatrixXd::Identity(3, 3);
    v.tail(J.cols()) = W_inv * J.transpose() * JWJt.ldlt().solve(err);

    double alpha = std::min(1.0, 1.0 / err.norm());
    q = pinocchio::integrate(model, q, v * alpha);

    if (debug)
    {
      std::cout << "jacobian:" << std::endl;
      std::cout << J << std::endl;
      std::cout << "JJt" << std::endl;
      std::cout << JJt << std::endl;
      std::cout << "W_inv" << std::endl;
      std::cout << W_inv << std::endl;
      std::cout << "JWJt" << std::endl;
      std::cout << JWJt << std::endl;
      std::cout << "v:" << std::endl;
      std::cout << v.transpose() << std::endl;
      std::cout << "\n------------------------------\n" << std::endl;
    }

    // clamp joint angles to limits
    q = q.cwiseMax(model.lowerPositionLimit).cwiseMin(model.upperPositionLimit);
  }
  return false;
}

inline void computeJointVelocityAndAcceleration(const pinocchio::Model& model, pinocchio::Data& data,
                                                const Eigen::VectorXd& q, const Eigen::Vector3d& xdot_des,
                                                const Eigen::Vector3d& xddot_des, const pinocchio::FrameIndex frame_id,
                                                Eigen::VectorXd& dq_out, Eigen::VectorXd& ddq_out)
{
  Eigen::MatrixXd J6 = Eigen::MatrixXd::Zero(6, model.nv);
  pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::WORLD, J6);  // world frame

  Eigen::MatrixXd J = J6.topRows(3).rightCols(model.njoints - 2);  // position. universe and root
  Eigen::MatrixXd JJt = J * J.transpose() + 1e-12 * Eigen::MatrixXd::Identity(3, 3);

  Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(xdot_des);
  dq_out.tail(model.njoints - 2) = dq;

  pinocchio::forwardKinematics(model, data, q, dq_out);
  pinocchio::computeJointJacobiansTimeVariation(model, data, q, dq_out);

  Eigen::MatrixXd Jdot6 = Eigen::MatrixXd::Zero(6, model.nv);
  pinocchio::getFrameJacobianTimeVariation(model, data, frame_id, pinocchio::WORLD, Jdot6);
  Eigen::MatrixXd Jdot = Jdot6.topRows(3).rightCols(model.njoints - 2);  // position. universe and root

  Eigen::VectorXd ddq = J.transpose() * JJt.ldlt().solve(xddot_des - Jdot * dq_out);
  ddq_out = ddq.tail(model.njoints - 2);
}

}  // namespace motion_planning
