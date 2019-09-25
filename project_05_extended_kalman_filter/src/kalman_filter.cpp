#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_; 
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
    VectorXd y;
    y = z - H_ * x_;
    MatrixXd Ht= H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Sin = S.inverse();
    MatrixXd K = P_ * Ht * Sin;
  
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    // new state
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
   //recover state parameters
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);
  
    //Checking the value is not zero
  	if(px == 0. && py == 0.)
    	return;
  
    // Equations for h_func below
    double h1 = sqrt(px * px + py * py);
    //check division by zero
    if (h1 < .00001) {
		h1 = .00001;
    }
    double h2 = atan2(py, px);
    double h3 = (px*vx + py * vy) / h1;
 
    //Feed in equations above
    VectorXd hx(3);
    hx << h1, h2, h3;
 
    VectorXd y = z - hx;
    // Normalize the angle
    while (y(1) > M_PI) {
        y(1) -= 2 * M_PI;
    }
    while (y(1) < -M_PI) {
        y(1) += 2 * M_PI;
    }
  
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;
  
    x_ = x_ + K*y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I-K*H_) * P_ ;   
}
