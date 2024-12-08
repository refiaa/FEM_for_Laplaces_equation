import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from typing import Tuple

class Material:
    def __init__(self, young_modulus: float, poisson_ratio: float):
        self.e = young_modulus
        self.nu = poisson_ratio
        self.d_mat = self._compute_d_matrix()

    def _compute_d_matrix(self) -> np.ndarray:
        factor = self.e / (1.0 - self.nu**2)
        d_mat = factor * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0]
        ])
        return d_mat

class Mesh:
    def __init__(self, width: float, height: float, nx: int, ny: int):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.nnx = nx + 1
        self.nny = ny + 1
        self.coords = self._generate_coordinates()
        self.connectivity = self._generate_connectivity()

    def _generate_coordinates(self) -> np.ndarray:
        x_coords = np.linspace(0.0, self.width, self.nnx)
        y_coords = np.linspace(0.0, self.height, self.nny)
        xx, yy = np.meshgrid(x_coords, y_coords)
        coords = np.column_stack((xx.ravel(), yy.ravel()))
        return coords

    def _generate_connectivity(self) -> np.ndarray:
        conn = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * self.nnx + i
                n2 = n1 + 1
                n3 = (j + 1) * self.nnx + i
                n4 = n3 + 1
                conn.append([n1, n2, n4, n3])
        return np.array(conn, dtype=int)

class Element:
    def __init__(self, d_mat: np.ndarray):
        self.d_mat = d_mat
        self.gauss_points, self.weights = self._gauss_quadrature()

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])

        dn_dxi = 0.25 * np.array([
            -(1 - eta),
            (1 - eta),
            (1 + eta),
            -(1 + eta)
        ])

        dn_deta = 0.25 * np.array([
            -(1 - xi),
            -(1 + xi),
            (1 + xi),
            (1 - xi)
        ])

        return n, dn_dxi, dn_deta

    @staticmethod
    def _gauss_quadrature():
        gp = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        gauss_points = [(xi, eta) for xi in gp for eta in gp]
        weights = [1.0 * 1.0] * 4 
        return gauss_points, weights

    def element_stiffness(self, xe: np.ndarray, ye: np.ndarray) -> np.ndarray:
        ke = np.zeros((8, 8))
        for (xi, eta), w in zip(self.gauss_points, self.weights):
            _, dn_dxi, dn_deta = self._shape_functions(xi, eta)

            j11 = np.dot(dn_dxi, xe)
            j12 = np.dot(dn_dxi, ye)
            j21 = np.dot(dn_deta, xe)
            j22 = np.dot(dn_deta, ye)
            jac = j11 * j22 - j12 * j21

            inv_jac = np.array([[j22, -j12], [-j21, j11]]) / jac
            dn_dx = inv_jac[0, 0] * dn_dxi + inv_jac[0, 1] * dn_deta
            dn_dy = inv_jac[1, 0] * dn_dxi + inv_jac[1, 1] * dn_deta

            b_mat = np.zeros((3, 8))
            for i_node in range(4):
                b_mat[0, 2*i_node]   = dn_dx[i_node]
                b_mat[1, 2*i_node+1] = dn_dy[i_node]
                b_mat[2, 2*i_node]   = dn_dy[i_node]
                b_mat[2, 2*i_node+1] = dn_dx[i_node]

            ke += b_mat.T @ self.d_mat @ b_mat * jac * w
        return ke

class FEMSolver:
    def __init__(self, mesh: Mesh, material: Material):
        self.mesh = mesh
        self.material = material
        self.element = Element(self.material.d_mat)

        self.ndof = self.mesh.coords.shape[0] * 2
        self.K = sp.lil_matrix((self.ndof, self.ndof), dtype=float)
        self.F = np.zeros(self.ndof, dtype=float)

        self._apply_boundary_conditions = False
        self.fixed_dofs = []
        self.fixed_values = []
        self.traction_forces = None

    def assemble_system(self):
        for e_id, conn in enumerate(self.mesh.connectivity):
            xe = self.mesh.coords[conn, 0]
            ye = self.mesh.coords[conn, 1]
            ke = self.element.element_stiffness(xe, ye)

            edof = []
            for n_id in conn:
                edof.extend([2 * n_id, 2 * n_id + 1])
            edof = np.array(edof)

            for i_local, i_glob in enumerate(edof):
                for j_local, j_glob in enumerate(edof):
                    self.K[i_glob, j_glob] += ke[i_local, j_local]

        sigma_x = 100.0
        right_nodes = np.where(np.isclose(self.mesh.coords[:, 0], self.mesh.width))[0]
        right_nodes = right_nodes[np.argsort(self.mesh.coords[right_nodes, 1])]

        for i in range(len(right_nodes) - 1):
            n1 = right_nodes[i]
            n2 = right_nodes[i + 1]
            y1 = self.mesh.coords[n1, 1]
            y2 = self.mesh.coords[n2, 1]
            length = y2 - y1

            f_line = sigma_x * length
            self.F[2 * n1] += 0.5 * f_line
            self.F[2 * n2] += 0.5 * f_line

    def apply_boundary_conditions(self):
        left_nodes = np.where(np.isclose(self.mesh.coords[:, 0], 0.0))[0]
        for ln in left_nodes:
            self._fix_dof(2 * ln, 0.0)

        bottom_left = np.where((np.isclose(self.mesh.coords[:, 0], 0.0)) &
                               (np.isclose(self.mesh.coords[:, 1], 0.0)))[0]
        if len(bottom_left) == 1:
            self._fix_dof(2 * bottom_left[0] + 1, 0.0)
        self._apply_boundary_conditions = True

    def _fix_dof(self, dof: int, value: float):
        self.fixed_dofs.append(dof)
        self.fixed_values.append(value)

    def solve(self) -> np.ndarray:
        if not self._apply_boundary_conditions:
            raise RuntimeError()

        self.K = self.K.tocsr()

        large_val = 1e20
        for dof, val in zip(self.fixed_dofs, self.fixed_values):
            self.K[dof, :] = 0.0
            self.K[:, dof] = 0.0
            self.K[dof, dof] = large_val
            self.F[dof] = val * large_val

        u = spla.spsolve(self.K, self.F)
        return u

    def compute_stresses(self, u: np.ndarray) -> np.ndarray:
        stresses = []
        for e_id, conn in enumerate(self.mesh.connectivity):
            xe = self.mesh.coords[conn, 0]
            ye = self.mesh.coords[conn, 1]

            edof = []
            for n_id in conn:
                edof.extend([2 * n_id, 2 * n_id + 1])
            edof = np.array(edof)
            u_e = u[edof]

            _, dn_dxi, dn_deta = self.element._shape_functions(0.0, 0.0)

            j11 = np.dot(dn_dxi, xe)
            j12 = np.dot(dn_dxi, ye)
            j21 = np.dot(dn_deta, xe)
            j22 = np.dot(dn_deta, ye)
            jac = j11 * j22 - j12 * j21
            inv_jac = np.array([[j22, -j12], [-j21, j11]]) / jac
            dn_dx = inv_jac[0, 0] * dn_dxi + inv_jac[0, 1] * dn_deta
            dn_dy = inv_jac[1, 0] * dn_dxi + inv_jac[1, 1] * dn_deta

            b_mat = np.zeros((3, 8))
            for i_node in range(4):
                b_mat[0, 2 * i_node]   = dn_dx[i_node]
                b_mat[1, 2 * i_node + 1] = dn_dy[i_node]
                b_mat[2, 2 * i_node]   = dn_dy[i_node]
                b_mat[2, 2 * i_node + 1] = dn_dx[i_node]

            eps = b_mat @ u_e
            sigma = self.material.d_mat @ eps
            stresses.append(sigma)
        return np.array(stresses)

def main():
    width = 1.0
    height = 1.0
    nx = 4
    ny = 4
    young_modulus = 1.92e4
    poisson_ratio = 0.2

    material = Material(young_modulus, poisson_ratio)
    mesh = Mesh(width, height, nx, ny)

    solver = FEMSolver(mesh, material)

    solver.assemble_system()

    solver.apply_boundary_conditions()

    u = solver.solve()

    stresses = solver.compute_stresses(u)

    print("Nodal Displacements (u_x, u_y) in meters:")
    for i_node in range(mesh.coords.shape[0]):
        ux = u[2 * i_node]
        uy = u[2 * i_node + 1]
        print(f"Node {i_node}: Ux={ux:.6e}, Uy={uy:.6e}")

    print("\nElement Stresses [sigma_x, sigma_y, tau_xy] in Pa:")
    for e_id, sigma in enumerate(stresses):
        print(f"Element {e_id}: sigma_x={sigma[0]:.6e}, sigma_y={sigma[1]:.6e}, tau_xy={sigma[2]:.6e}")

if __name__ == "__main__":
    main()
