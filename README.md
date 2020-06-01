# Quadrotor_Trajectory_Tracking
### Quadrotor trajectory tracking controllers and simulator
These scripts implements linear model predictive control for quadrotor tracking trajectories or being stabilized to the origin. The controller is based on the linear state space model linearized from the equibilium point.

### To use the controller, following the steps:
1. Install the dependencies: numpy, scipy, matplotlib, osqp
    To install osqp: pip install osqp
2. python main.py

### Choose the controllers:
1. Track the Trajectory(True) or Stabilize to Origion(Fase)
    TRACKING = True
2. Choose Controller from 'MPC', 'LQR', 'PID'
    CONTROLLER = 'MPC' 
    Note that LQR and PID are under debugging.
