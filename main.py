#!/usr/bin/env python3
import numpy as np
from quad import Quad
from mpc import mpc
from mpl_toolkits import mplot3d
from lqr import FiniteHorizonLQR
import scipy.sparse as sparse
import scipy.signal as ss
import os
import math
import random
import argparse
import json
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from task import Task
np.set_printoptions(precision=4, suppress=True)

## Track the Trajectory(True) or Stabilize to Origion(Fase)
TRACKING = True
## Choose Controller from 'MPC', 'LQR', 'PID'
CONTROLLER = 'MPC' 


M = 0.044 # mass of Crazyflie 2.1
G = 9.8   # gravity
PI = 3.14
dt = 0.02 # discrete time
dl = 0.02 # discrete length of trajectory
N_IND_SEARCH = 10
GOAL_DIS = 0.1  # [m] goal distance, consider approach
STOP_SPEED = 0.1  # [m/s] stop speed

## Choose Trajectory to Track
curve_pos = lambda x: np.array([-np.sin(0.5 * x), 5 + np.cos(0.5 * x),1.*x])
curve_vel = lambda x: np.array([-0.5*np.cos(0.5*x), -0.5*np.sin(0.5*x), x/x])

# curve_pos = lambda x: np.array([-np.sin(30 * x), 5 + np.sin(0.2 * x),3.*x])
# curve_vel = lambda x: np.array([-30*np.cos(30*x), 0.2*np.cos(0.2*x), 3. * x/x])

# curve_pos = lambda x: np.array([-np.sin(2 * x), 5 + np.cos(0.3 * x),1.2*x])
# curve_vel = lambda x: np.array([-2*np.cos(2*x), -0.3*np.sin(0.3*x), 1.2*x/x])

# [a,b,c] = np.random.random([3,])
# curve_pos = lambda t: np.array([-np.sin((a+b)*t), np.sin((c+b)*t), (a+c)*t*np.sin(t)+0.05*t])
# curve_vel = lambda t: np.array([-(a+b)*np.cos((a+b)*t), (c+b)*np.cos((c+b)*t), (a+c)*t/t+(a+c)*t*np.cos(t)+0.05*t/t])

# [a,b,c] = np.random.random([3,])
# curve_pos = lambda t: np.array([-np.sin((a+b)*t) * np.sin(a*t), np.sin((c+b)*t), 0.05*t])
# curve_vel = lambda t: np.array([-(a+b)*np.cos((a+b)*t)*np.sin(a*t) - np.sin((a+b)*t)*np.cos(a*t)*a, (c+b)*np.cos((c+b)*t), 0.05*t/t])

def main():

    simulation_time = 20

    ## prepare the trajectory(course) to track
    idx = np.arange(0, 4 * simulation_time, dl)
    course_pos = curve_pos(idx)
    course_vel = curve_vel(idx)
    speed = 0.5 # m/s
    course_vel = course_vel / np.linalg.norm(course_vel, axis=0) * (speed + np.random.rand())  # normalize with the same speed, but not direction.
    course_vel[:, 0] = 0  # avoid nan
    course_vel[:, -1] = 0  # stop
    course = np.concatenate((course_pos, course_vel), axis=0) # desired state with dim 6, shape [dim, timeslot]

    # init state and target state
    init_state = course[:, 0] + np.array([1., 2., 0., 0., 0., 0.])#np.array([-1., 1., 0., 0., 0., 0.]) # 
    state = init_state
    nearest, _ = calc_nearest_index(state, course, 0)
    target_states = course[:, nearest]

    # quadrotor object
    quad = Quad(init_state)

    # dimension
    nu = 4
    nx = init_state.shape[0] # 6

    # controller parameters setting for both lqr and linear mpc
    u = np.zeros(nu)
    horizon = 20  # predict and control horizon in mpc
    Q = np.array([20. , 20.,  20., 8, 8, 8])
    QN = Q
    # Q = np.array([8. , 8.,  8., 0.8, 0.8, 0.8])
    # QN = np.array([10., 10., 10., 1.0, 1.0, 1.0])
    R = np.array([6., 6., 6.])
    Ad, Bd = linearized_model(state) # actually use linear time-invariant model linearized from the equilibrium point
    
    # record state
    labels = ['x', 'y', 'z', 'vx', 'vy', 'vz'] # convenient for plot
    state_control = {x: [] for x in labels}
    target_chosen = {x: [] for x in labels}
    for ii in range(6):
        target_chosen[labels[ii]].append(target_states[ii])
    err = [] # tracking error

    # visualization
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # control loop
    timestamp = 0.0
    while timestamp <= simulation_time:
        print("timestamp", timestamp)
        if TRACKING:
            target_states, nearest, chosen_idxs = calc_ref_trajectory(state, course, nearest, horizon)
        else:
            target_states = np.array([0, 0, 10, 0, 0, 0])

        if CONTROLLER=='MPC':
            mpc_policy = mpc(state, target_states, Ad, Bd, horizon, Q, QN, R, TRACKING)
            nominal_action, actions, nominal_state = mpc_policy.solve()
            action = nominal_action #- np.dot(K,(target_states - nominal_state))
            roll_r, pitch_r, thrust_r = action
            u[0] = - pitch_r  
            u[1] = - roll_r
            u[2] = 0.0
            u[3] = thrust_r + M * G

        elif CONTROLLER=='LQR':
            lqr_policy = FiniteHorizonLQR(Ad, Bd, Q, R, QN, horizon=horizon)  # instantiate a lqr controller
            lqr_policy.set_target_state(target_states)  ## set target state to koopman observable state
            lqr_policy.sat_val = np.array([PI/8., PI/8., 0.75])
            K, ustar = lqr_policy(state)
            
            u[0] = - ustar[1]
            u[1] = - ustar[0]
            u[2] = 0.0
            u[3] = ustar[2] + M * G
            print("K", K)

        elif CONTROLLER=='PID':
            # Compute control errors
            x, y, z, dx, dy, dz = state
            x_r, y_r, z_r, dx_r, dy_r, dz_r = target_states
            ex = x - x_r
            ey = y - y_r
            ez = z - z_r
            dex = dx - dx_r
            dey = dy - dy_r
            dez = dz - dz_r
            xi = 1.2
            wn = 3.0
            
            Kp = - wn * wn
            Kd = - 2 * wn * xi
            Kxp = 1.2 * Kp
            Kxd = 1.2 * Kd
            Kyp = Kp
            Kyd = Kd
            Kzp = 0.8 * Kp
            Kzd = 0.8 * Kd
            
            pitch_r = Kxp * ex + Kxd * dex
            roll_r = Kyp * ey + Kyd * dey
            thrust_r = (Kzp * ez + Kzd * dez + 9.8) * 0.44
            
            u[0] = pitch_r
            u[1] = roll_r
            u[2] = 0.
            u[3] = thrust_r + M * G



        next_state = quad.step(state, u)
        state = next_state

        timestamp = timestamp + dt


        if TRACKING:
            err.append(state[:3] - target_states[:3, -1])
        else:
            err.append(state[:3] - target_states[:3])
        
        # record
        for ii in range(len(labels)):
            state_control[labels[ii]].append(state[ii])
            if TRACKING:
                target_chosen[labels[ii]].append(target_states[ii, -1])
            else:
                target_chosen[labels[ii]].append(target_states[ii])

        ## plot
        if True: #timestamp > dt:
            # # plot in real time
            plt.cla()
            x = np.append(init_state[0], state_control['x'])
            y = np.append(init_state[1], state_control['y'])
            z = np.append(init_state[2], state_control['z'])
            final_idx = len(x) - 1
            ax.plot3D(x, y, z, label='Quadcopter flight trajectory')
            ax.plot3D([init_state[0]], [init_state[1]], [init_state[2]], label='Initial position', color='blue', marker='x')
            ax.plot3D([x[final_idx]], [y[final_idx]], [z[final_idx]], label='Final position', color='green', marker='o')
            if TRACKING:
                plt_len = chosen_idxs[-1] + 100
                ax.plot3D(course[0, :plt_len], course[1, :plt_len], course[2, :plt_len])
                # ax.scatter3D([course[0, chosen_idxs[0]]], [course[1, chosen_idxs[0]]], [course[2, chosen_idxs[0]]], color='red', label='chosen target flight trajectory',
                #              marker='o')
                for i in range(horizon):
                    ax.scatter3D([course[0, chosen_idxs[i]]], [course[1, chosen_idxs[i]]], [course[2, chosen_idxs[i]]], color='red', marker='o')

            ax.scatter3D([target_states[0]], [target_states[1]], [target_states[2]], color='red', marker='o')
            plt.legend()
            plt.pause(0.01)
    err = np.stack(err)
    
    plt.figure()
    plt.plot(err[:,0], label='x')
    plt.plot(err[:,1], label='y')
    plt.plot(err[:,2], label='z')
    plt.legend()
    plt.show()


def linearized_model(state):
    g = 9.8
    m = 0.044
    # linear continuous system
    Ac = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    Bc = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [g, 0, 0],
        [0, - g, 0],
        [0, 0, 1.0/m]
    ])
    Cc = np.eye(Ac.shape[0])  # Full state observed
    Dc = np.zeros((Bc.shape[0], Bc.shape[1]))
    sysd = ss.cont2discrete((Ac, Bc, Cc, Dc), dt)  # Get the discrete time system
    Ad = sysd[0]
    Bd = sysd[1]
    return Ad, Bd

def calc_ref_trajectory(state, course, pind, horizon):
    ncourse = course.shape[1]

    xref = np.zeros((6, horizon + 1))

    nearest, _ = calc_nearest_index(state, course, pind)

    if pind >= nearest:
        nearest = pind
    ind = nearest + 2
    xref[:, 0] = course[:, ind]

    travel = 0.0
    chosen_idxs= [ind]
    # ori_traj_slot = 0.5 * dt
    for i in range(horizon):
        v = np.linalg.norm(state[3:])
        travel += abs(v) * dt
        dind = int(round(travel / dl)) #+ 1
        # print(ind + dind)
        if (ind + dind) < ncourse:
            xref[:, i+1] = course[:, ind + dind]
        else:
            xref[:, i+1] = course[:, ncourse - 1]
        chosen_idxs.append(ind + dind)
    return xref, nearest, chosen_idxs


def calc_nearest_index(state, course, pind):

    dx = [state[0] - icx for icx in course[0, pind:(pind + N_IND_SEARCH)]]
    dy = [state[1] - icy for icy in course[1, pind:(pind + N_IND_SEARCH)]]
    dz = [state[2] - icz for icz in course[2, pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 + idy ** 2 for (idx, idy, idz) in zip(dx, dy, dz)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    return ind, mind

def iterative_linear_mpc_control(zrefs, z0, actions, Ad, Bd, horizon, max_iter, KA, KB):
    if actions is None:
        actions = np.zeros((3, horizon))

    for i in range(max_iter):
        pactions = actions
        # mpc_policy = mpc(x0, xrefs, horizon)
        mpc_policy = mpc(z0, zrefs, Ad, Bd, horizon, KA, KB, TRACKING)
        first_action, actions, res = mpc_policy.solve()
        du = np.linalg.norm(actions - pactions)
        if du <= 0.1:
            break
    else:
        print("Iterative is max iter")
    return first_action, actions, res

def random_act():
    new_thrust = random.gauss(0.5, 0.1)
    roll = random.gauss(0, np.pi / 6)
    pitch = random.gauss(0, np.pi / 6)
    yaw = 0 # random.gauss(0, np.pi / 6)
    return np.array([roll, pitch, yaw, new_thrust + random.gauss(0., 0.02)])
    # return np.array([roll, pitch, new_thrust + random.gauss(0., 0.02)])

def simulate_with_actions(state, N, action_schedule):
    quad_dev = Quad(state)
    trajectory = [state.copy()]
    action_schedule[:, 2] += M * G
    for i in range(N-1):
        action = action_schedule[i]
        state, _ = quad_dev.step(state, action)
        trajectory.append(state.copy())
    return np.array(trajectory)

def acquire_action_schedule(horizon):
    action_schedule = [random_act() for i in range(horizon)]
    return np.array(action_schedule)

if __name__=='__main__':
    main()
