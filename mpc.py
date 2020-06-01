import osqp
import numpy as np
# import scipy as sp
import scipy.sparse as sparse
# from c2d import c2d


class mpc(object):

    def __init__(self, x0, xr, Ad, Bd, horizon, Q, QN, R, Tracking=True):
        
        pi = 3.14
        self.nx = Bd.shape[0]
        self.nu = Bd.shape[1]

        self.Ad = Ad
        self.Bd = Bd

        self.umin = np.array([-pi/8., -pi/8.,-0.75])  
        self.umax = np.array([ pi/8.,  pi/8., 0.75])

        self.xmin = np.array([-np.inf]*self.nx)
        self.xmax = np.array([ np.inf]*self.nx)

        self.Q = sparse.diags(Q)
        self.QN = sparse.diags(QN)
        self.R = sparse.diags(R)

        self.x0 = np.array(x0)
        self.xr = np.array(xr)

        self.N = horizon
        self.Tracking = Tracking


    def solve(self):
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))

        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
                               sparse.kron(sparse.eye(self.N), self.R)]).tocsc()
        if self.Tracking:
            pass
            proc = np.array([-self.Q.dot(xr_i) for xr_i in self.xr.T])[:-1].flatten()
            q = np.hstack([proc, -self.QN.dot(self.xr[:, -1]), np.zeros(self.N * self.nu)])
        else:
            q = np.hstack([np.kron(np.ones(self.N), -self.Q.dot(self.xr)), -self.QN.dot(self.xr),
                       np.zeros(self.N * self.nu)])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-self.x0, np.zeros(self.N * self.nx)])
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        lineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        uineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmax), np.kron(np.ones(self.N), self.umax)])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq]).tocsc()
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-self.N * self.nu:-(self.N - 1) * self.nu]
        ctrls = res.x[-self.N * self.nu:].reshape(-1, 4).T
        # print("ctrl: ", ctrl)
        nominal_state = res.x[self.nx:2*self.nx]
        return ctrl, ctrls, nominal_state

