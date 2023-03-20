import argparse

import numpy as np

import taichi as ti


@ti.data_oriented
class Mass_spring:
    def __init__(self, NV, NE, dt, Kl, mass):
        self.NV = NV
        self.NE = NE
        self.dt = dt

        self.Kl = Kl

        self.pos = ti.Vector.field(3, ti.f32, self.NV)
        self.prev_pos = ti.Vector.field(3, ti.f32, self.NV)
        self.vel = ti.Vector.field(3, ti.f32, self.NV)
        self.e2v = ti.Vector.field(2, int, self.NE)  
        self.mass = mass
        #explicit only
        self.damping = 14.5

        self.l_i = ti.field(ti.f32, self.NE)

        self.gravity = ti.Vector([0, 0, 9.8])

        self.F_f = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)

        self.F_f_mid = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)
        self.pos_mid = ti.Vector.field(3, ti.f32, self.NV)

        self.F_b = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)

        self.A_builder = ti.linalg.SparseMatrixBuilder(3 * NV,
                                                3 * NV,
                                                max_num_triplets=2000000)
        self.U = ti.field(float, ())
        self.manipulate_acc = ti.Vector.field(3, ti.f32, ())
        self.solver = ti.linalg.SparseSolver(ti.f32, "LLT")
        self.F_b_ndarr = ti.ndarray(dtype=ti.f32, shape=3 * NV)

    @ti.kernel
    def compute_energy(self):
        self.U[None] = 0
        
        self.U[None] += self.mass * -self.manipulate_acc[None].dot(self.pos[0]) * self.dt**2
        for i in range(self.NV):
            self.U[None] += self.pos[i].dot(self.gravity) * self.mass * self.dt**2

        for i in range(self.NV):
            X = self.pos[i] - self.prev_pos[i] - self.vel[i]*self.dt
            self.U[None] += 0.5 * self.mass * X.dot(X)

        for i in range(self.NE):
            ia, ib = self.e2v[i]
            a, b = self.pos[ia], self.pos[ib]

            base_len = self.l_i[i]
            delta = a - b
            l_tau = delta.norm()
            self.U[None] += (l_tau - base_len)**2 * self.Kl * self.dt**2

    @ti.func
    def compute_dl(self, l_tau, l_base):
        return -self.Kl * 2.0 * (l_base - l_tau)

    @ti.func
    def compute_dl2(self, l_base):
        return self.Kl * 2.0

    @ti.func
    def compute_l_dx2(self, p1, p2, l_tau, dim):
        return (l_tau**2 - (p1[dim] - p2[dim])**2) / (l_tau**3)

    @ti.func
    def compute_l_dxy(self, p1, p2, l_tau, dim, d1):
        return (p1[dim] - p2[dim]) * (p1[d1] - p2[d1]) / (l_tau**3)


    @ti.kernel
    def compute_Hessian(self, H: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            for j in range(3):
                H[3 * i + j, 3 * i + j] += self.mass
        dt = self.dt
        for i in range(self.NE):
            xx, yy = self.e2v[i]
            # a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            # membrane_energy_edge
            # print("compute edge")
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    delta = self.pos[xx] - self.pos[yy]
                    a, b = self.pos[xx], self.pos[yy]
                    l_tau = delta.norm()
                    dldx = delta / l_tau
                    dldy = -dldx
                    base_len = self.l_i[i]
                    if(j==k):
                        H[xx * 3 + j, xx * 3 + k] += dt**2 * (self.compute_dl(l_tau, base_len)*self.compute_l_dx2(a, b, l_tau, j) + self.compute_dl2(base_len)*dldx[j]*dldx[k])
                        H[yy * 3 + j, yy * 3 + k] += dt**2 * (self.compute_dl(l_tau, base_len)*self.compute_l_dx2(b, a, l_tau, j) + self.compute_dl2(base_len)*dldy[j]*dldy[k])
                        H[xx * 3 + j, yy * 3 + k] += dt**2 * (-self.compute_dl(l_tau, base_len)*self.compute_l_dx2(a, b, l_tau, j) + self.compute_dl2(base_len)*dldx[j]*dldx[k])
                        H[yy * 3 + j, xx * 3 + k] += dt**2 * (-self.compute_dl(l_tau, base_len)*self.compute_l_dx2(b, a, l_tau, j) + self.compute_dl2(base_len)*dldy[j]*dldy[k])
                    
                    else:
                        H[xx * 3 + j, xx * 3 + k] += dt**2 * (self.compute_dl(l_tau, base_len)*self.compute_l_dxy(a, b, l_tau, j, k) + self.compute_dl2(base_len)*dldx[j]*dldx[k])
                        H[yy * 3 + j, yy * 3 + k] += dt**2 * (self.compute_dl(l_tau, base_len)*self.compute_l_dxy(b, a, l_tau, j, k) + self.compute_dl2(base_len)*dldy[j]*dldy[k])
                        H[xx * 3 + j, yy * 3 + k] += dt**2 * (-self.compute_dl(l_tau, base_len)*self.compute_l_dxy(a, b, l_tau, j, k) + self.compute_dl2(base_len)*dldx[j]*dldx[k])
                        H[yy * 3 + j, xx * 3 + k] += dt**2 * (-self.compute_dl(l_tau, base_len)*self.compute_l_dxy(b, a, l_tau, j, k) + self.compute_dl2(base_len)*dldy[j]*dldy[k])


    @ti.kernel
    def compute_residual_newton(self):
        for i in self.F_b:
            self.F_b[i] = self.mass * self.gravity * self.dt**2
        # self.F_b.fill(0)
        dt = self.dt

        self.F_b[0] -= self.mass * self.manipulate_acc[None] * dt**2
        for i in range(self.NV):
            self.F_b[3 * i] += self.rho * (self.dx ** 2) * (self.pos[i] - self.prev_pos[i] - self.vel[i]*dt)
        # print("mass", rho * dx ** 2 * (pos[1] - prev_pos[1] - vel[1]*dt))
        for i in range(self.NE):
            xx, yy = self.e2v[i]

            base_len = self.l_i[i]
            delta = self.pos[xx] - self.pos[yy]
            l_tau = delta.norm()
            self.F_b[xx] += delta * self.compute_dl(l_tau, base_len) / l_tau * (dt**2) 
            self.F_b[yy] += -delta * self.compute_dl(l_tau, base_len) / l_tau * (dt**2)

    @ti.kernel
    def compute_residual(self):
        for i in self.F_b:
            self.F_b[i] = self.mass * self.vel[i] + self.dt * self.F_f[i]

    @ti.kernel
    def compute_force(self):
        self.F_f.fill(0)
        for i in self.F_f:
            self.F_f[i] -= self.mass * self.gravity
        dt = self.dt

        self.F_f[0] += self.mass * self.manipulate_acc[None]

        for i in range(self.NE):
            xx, yy = self.e2v[i]

            base_len = self.l_i[i]
            delta = self.pos[xx] - self.pos[yy]
            l_tau = delta.norm()
            self.F_f[xx] -= delta * self.compute_dl(l_tau, base_len) / l_tau
            self.F_f[yy] -= -delta * self.compute_dl(l_tau, base_len) / l_tau
    
    @ti.kernel
    def compute_force_mid(self):
        self.F_f_mid.fill(0)
        for i in self.F_f:
            self.F_f_mid[i] -= self.mass * self.gravity
        dt = self.dt

        self.F_f_mid[0] += self.mass * self.manipulate_acc[None]

        for i in range(self.NE):
            xx, yy = self.e2v[i]

            base_len = self.l_i[i]
            delta = self.pos_mid[xx] - self.pos_mid[yy]
            l_tau = delta.norm()
            self.F_f_mid[xx] -= delta * self.compute_dl(l_tau, base_len) / l_tau
            self.F_f_mid[yy] -= -delta * self.compute_dl(l_tau, base_len) / l_tau

    @ti.kernel
    def update_vel(self):
        for i in range(self.NV):
            for j in range(3):
                self.vel[i][j] = (self.pos[i][j] - self.prev_pos[i][j]) / self.dt

    @ti.kernel
    def get_prev_pos(self):
        for i in range(self.NV):
            for j in range(3):
                self.prev_pos[i][j] = self.pos[i][j]

    @ti.kernel
    def step_explicit(self):
        for i in range(self.NV):
            cond = (self.pos[i] <= 0) & (self.vel[i] <= 0)
            for j in ti.static(range(self.pos.n)):
                if cond[j]:
                    self.vel[i][j] = 0
            self.pos[i] += self.vel[i] * self.dt
            self.vel[i] += self.F_f[i] / self.mass * self.dt
            self.vel[i] *= ti.exp(-self.dt * self.damping)

    @ti.kernel
    def step_symplectic(self):
        for i in range(self.NV):
            self.vel[i] += self.F_f[i] / self.mass * self.dt
        for i in range(self.NV):
            cond = (self.pos[i] <= 0) & (self.vel[i] <= 0)
            for j in ti.static(range(self.pos.n)):
                if cond[j]:
                    self.vel[i][j] = 0
            self.pos[i] += self.dt * self.vel[i]
            self.vel[i] *= ti.exp(-self.dt * self.damping)

    def step_implicit(self):
        self.compute_force()
        self.compute_residual()
        self.compute_Hessian(self.A_builder)
        Hessian = self.A_builder.build()
        self.flatten(self.F_b_ndarr, self.F_b)
        self.solver.analyze_pattern(Hessian)
        self.solver.factorize(Hessian)
        # Solve the linear system

        vel = self.solver.solve(self.F_b_ndarr)
        
        self.aggragate(self.vel, vel)
        self.step_implicit_pos()

    @ti.kernel
    def step_implicit_pos(self):
        for i in range(self.NV):
            cond = (self.pos[i] <= 0) & (self.vel[i] <= 0)
            for j in ti.static(range(self.pos.n)):
                if cond[j]:
                    self.vel[i][j] = 0
            self.pos[i] += self.dt * self.vel[i]

    @ti.kernel
    def flatten(self, dest: ti.types.ndarray(), src: ti.template()):
        for i in range(self.NV):
            for j in range(3):
                dest[3 * i + j] = src[i][j]

    @ti.kernel
    def aggragate(self, dest: ti.template(), src: ti.types.ndarray()):
        # print(step)
        for i in range(self.NV):
            for j in range(3):
                dest[i][j] = src[3 * i + j]
    
    @ti.kernel
    def step_midpoint_pos(self):

        for i in range(self.NV):
            cond = (self.pos[i] <= 0) & (self.vel[i] <= 0)
            for j in ti.static(range(self.pos.n)):
                if cond[j]:
                    self.vel[i][j] = 0
            v_mid = self.vel[i] + self.F_f[i] / self.mass * self.dt * 0.5
            self.pos[i] += self.vel[i] * self.dt
            
            self.pos_mid[i] = self.pos[i] + self.vel[i] * self.dt * 0.5

    
    @ti.kernel
    def step_midpoint_vel(self):
        for i in range(self.NV):
            self.vel[i] += self.F_f_mid[i] * self.dt
            self.vel[i] *= ti.exp(-self.dt * self.damping)

    def step(self, s):
        if s == "explicit":
            self.compute_force()
            self.step_explicit()
        elif s == "symplectic":
            self.compute_force()
            self.step_symplectic()
        elif s == "midpoint":
            self.compute_force()
            self.step_midpoint_pos()
            self.compute_force_mid()
            self.step_midpoint_vel()
        elif s == "implicit":
            self.step_implicit()
        else:
            print("not implemented")

    