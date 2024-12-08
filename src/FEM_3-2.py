import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from typing import Tuple, List

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
    def __init__(self, width: float, height: float, nx: int, ny: int,
                 hole_center: Tuple[float, float], hole_radius: float):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.nnx = nx + 1
        self.nny = ny + 1
        self.hole_center = hole_center
        self.hole_radius = hole_radius
        self.coords = self._generate_coordinates()
        self.connectivity = self._generate_connectivity()
        self._remove_hole_elements()
        self._condense_mesh()

    def _generate_coordinates(self) -> np.ndarray:
        x_coords = np.linspace(0.0, self.width, self.nnx)
        y_coords = np.linspace(0.0, self.height, self.nny)
        xx, yy = np.meshgrid(x_coords, y_coords)
        return np.column_stack((xx.ravel(), yy.ravel()))

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

    def _remove_hole_elements(self):
        xc, yc = self.hole_center
        new_conn = []
        for elem in self.connectivity:
            xe = self.coords[elem, 0]
            ye = self.coords[elem, 1]
            x_cen = np.mean(xe)
            y_cen = np.mean(ye)
            dist = np.sqrt((x_cen - xc)**2 + (y_cen - yc)**2)
            if dist > self.hole_radius:
                new_conn.append(elem)
        self.connectivity = np.array(new_conn, dtype=int)

    def _condense_mesh(self):
        if len(self.connectivity) == 0:
            raise RuntimeError("No elements left after removing hole.")

        used_nodes = np.unique(self.connectivity)
        new_index = -np.ones(self.coords.shape[0], dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        
        new_conn = []
        for elem in self.connectivity:
            new_conn.append([new_index[n] for n in elem])
        self.connectivity = np.array(new_conn, dtype=int)

        self.coords = self.coords[used_nodes, :]

class Element:
    def __init__(self, d_mat: np.ndarray):
        self.d_mat = d_mat
        self.gauss_points, self.weights = self._gauss_quadrature()

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = 0.25 * np.array([
            (1 - xi)*(1 - eta),
            (1 + xi)*(1 - eta),
            (1 + xi)*(1 + eta),
            (1 - xi)*(1 + eta)
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
        weights = [1.0]*4
        return gauss_points, weights

    def element_stiffness(self, xe: np.ndarray, ye: np.ndarray) -> np.ndarray:
        ke = np.zeros((8, 8))
        for (xi, eta), w in zip(self.gauss_points, self.weights):
            _, dn_dxi, dn_deta = self._shape_functions(xi, eta)
            j11 = np.dot(dn_dxi, xe)
            j12 = np.dot(dn_dxi, ye)
            j21 = np.dot(dn_deta, xe)
            j22 = np.dot(dn_deta, ye)
            jac = j11*j22 - j12*j21
            inv_jac = np.array([[j22, -j12], [-j21, j11]]) / jac
            dn_dx = inv_jac[0,0]*dn_dxi + inv_jac[0,1]*dn_deta
            dn_dy = inv_jac[1,0]*dn_dxi + inv_jac[1,1]*dn_deta

            b_mat = np.zeros((3, 8))
            for i_node in range(4):
                b_mat[0, 2*i_node]   = dn_dx[i_node]
                b_mat[1, 2*i_node+1] = dn_dy[i_node]
                b_mat[2, 2*i_node]   = dn_dy[i_node]
                b_mat[2, 2*i_node+1] = dn_dx[i_node]

            ke += b_mat.T @ self.d_mat @ b_mat * jac * w
        return ke

    def element_strain_stress(self, xe: np.ndarray, ye: np.ndarray, u_e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, dn_dxi, dn_deta = self._shape_functions(0.0, 0.0)
        j11 = np.dot(dn_dxi, xe)
        j12 = np.dot(dn_dxi, ye)
        j21 = np.dot(dn_deta, xe)
        j22 = np.dot(dn_deta, ye)
        jac = j11*j22 - j12*j21
        inv_jac = np.array([[j22, -j12], [-j21, j11]]) / jac
        dn_dx = inv_jac[0,0]*dn_dxi + inv_jac[0,1]*dn_deta
        dn_dy = inv_jac[1,0]*dn_dxi + inv_jac[1,1]*dn_deta

        b_mat = np.zeros((3, 8))
        for i_node in range(4):
            b_mat[0, 2*i_node]   = dn_dx[i_node]
            b_mat[1, 2*i_node+1] = dn_dy[i_node]
            b_mat[2, 2*i_node]   = dn_dy[i_node]
            b_mat[2, 2*i_node+1] = dn_dx[i_node]

        eps = b_mat @ u_e
        sigma = self.d_mat @ eps
        return eps, sigma

class FEMSolver:
    def __init__(self, mesh: Mesh, material: Material, sigma_applied=100.0):
        self.mesh = mesh
        self.material = material
        self.element = Element(material.d_mat)
        self.ndof = self.mesh.coords.shape[0]*2
        self.K = sp.lil_matrix((self.ndof, self.ndof), dtype=float)
        self.F = np.zeros(self.ndof, dtype=float)
        self.fixed_dofs = []
        self.fixed_values = []
        self._bc_applied = False
        self.sigma_applied = sigma_applied

    def assemble_system(self):
        for conn in self.mesh.connectivity:
            xe = self.mesh.coords[conn, 0]
            ye = self.mesh.coords[conn, 1]
            ke = self.element.element_stiffness(xe, ye)
            edof = []
            for n_id in conn:
                edof.extend([2*n_id, 2*n_id+1])
            edof = np.array(edof)
            for i_local, i_glob in enumerate(edof):
                for j_local, j_glob in enumerate(edof):
                    self.K[i_glob, j_glob] += ke[i_local, j_local]

        width = self.mesh.width
        right_nodes = np.where(np.isclose(self.mesh.coords[:,0], width))[0]
        right_nodes = right_nodes[np.argsort(self.mesh.coords[right_nodes,1])]
        for i in range(len(right_nodes)-1):
            n1 = right_nodes[i]
            n2 = right_nodes[i+1]
            y1 = self.mesh.coords[n1, 1]
            y2 = self.mesh.coords[n2, 1]
            length = y2 - y1
            f_line = self.sigma_applied * length
            self.F[2*n1] += 0.5 * f_line
            self.F[2*n2] += 0.5 * f_line

    def apply_boundary_conditions(self):
        left_nodes = np.where(np.isclose(self.mesh.coords[:,0], 0.0))[0]
        if len(left_nodes) == 0:
            raise RuntimeError("No nodes on the left boundary to apply boundary conditions.")

        min_y = np.min(self.mesh.coords[left_nodes, 1])
        bottom_left_nodes = left_nodes[np.isclose(self.mesh.coords[left_nodes, 1], min_y)]
        
        if len(bottom_left_nodes) == 0:
            raise RuntimeError("No bottom-left node found to fix y displacement.")
        
        bottom_left_node = bottom_left_nodes[0]
        
        for ln in left_nodes:
            self._fix_dof(2*ln, 0.0)
        
        self._fix_dof(2*bottom_left_node + 1, 0.0)
        
        self._bc_applied = True

    def _fix_dof(self, dof: int, value: float):
        self.fixed_dofs.append(dof)
        self.fixed_values.append(value)

    def solve(self) -> np.ndarray:
        if not self._bc_applied:
            raise RuntimeError("Boundary conditions not applied.")
        self.K = self.K.tocsr()
        large_val = 1e20

        for dof, val in zip(self.fixed_dofs, self.fixed_values):
            for j in range(self.ndof):
                self.K[dof, j] = 0.0
            for i in range(self.ndof):
                self.K[i, dof] = 0.0
            self.K[dof, dof] = large_val
            self.F[dof] = val * large_val

        u = spla.spsolve(self.K, self.F)
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            raise RuntimeError("Solution contains NaN or Inf values.")
        return u

    def compute_element_stress(self, u: np.ndarray) -> np.ndarray:
        stresses = []
        for conn in self.mesh.connectivity:
            edof = []
            for n_id in conn:
                edof.extend([2*n_id, 2*n_id+1])
            edof = np.array(edof)
            u_e = u[edof]
            xe = self.mesh.coords[conn,0]
            ye = self.mesh.coords[conn,1]
            _, sigma = self.element.element_strain_stress(xe, ye, u_e)
            stresses.append(sigma)
        return np.array(stresses)

    def compute_nodal_values(self, u: np.ndarray, var: str = "ux") -> np.ndarray:
        values = np.zeros(self.mesh.coords.shape[0])
        if var == "ux":
            for i in range(self.mesh.coords.shape[0]):
                values[i] = u[2*i]
        elif var == "uy":
            for i in range(self.mesh.coords.shape[0]):
                values[i] = u[2*i+1]
        else:
            raise ValueError("Invalid variable for nodal values.")
        return values

    def interpolate_to_nodes(self, u: np.ndarray, var: str = "sigmax") -> np.ndarray:
        if var not in ["sigmax","sigmay","tauxy"]:
            raise ValueError("Invalid variable for stress interpolation.")
        stresses = self.compute_element_stress(u)
        idx = {"sigmax":0, "sigmay":1, "tauxy":2}[var]
        node_stress = np.zeros(self.mesh.coords.shape[0])
        count = np.zeros(self.mesh.coords.shape[0])
        for e_id, conn in enumerate(self.mesh.connectivity):
            val = stresses[e_id, idx]
            for n in conn:
                node_stress[n] += val
                count[n] += 1.0
        node_stress[count>0] /= count[count>0]
        return node_stress

def main():
    E = 1.92e4
    nu = 0.2
    width = 2.0
    height = 4.0

    hole_center = (0, 0)
    hole_radius = 1
    nx = 200
    ny = 400
    sigma_applied = 100.0

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    material = Material(E, nu)
    mesh = Mesh(width, height, nx, ny, hole_center, hole_radius)

    solver = FEMSolver(mesh, material, sigma_applied)
    solver.assemble_system()
    solver.apply_boundary_conditions()
    u = solver.solve()

    sigmax_nodal = solver.interpolate_to_nodes(u, "sigmax")
    ux_nodal = solver.compute_nodal_values(u, "ux")

    tri_conn = []
    for conn in mesh.connectivity:
        tri_conn.append([conn[0], conn[1], conn[2]])
        tri_conn.append([conn[0], conn[2], conn[3]])
    tri_conn = np.array(tri_conn)

    plt.figure(figsize=(6,12))
    plt.tricontourf(mesh.coords[:,0], mesh.coords[:,1], tri_conn, sigmax_nodal, levels=20, cmap='viridis')
    plt.colorbar(label='sigma_x [Pa]')
    plt.title('Sigma_x Contour')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(os.path.join(output_dir, "problem2_sigma_x_contour.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,12))
    plt.tricontourf(mesh.coords[:,0], mesh.coords[:,1], tri_conn, ux_nodal, levels=20, cmap='viridis')
    plt.colorbar(label='u_x [m]')
    plt.title('Ux Contour')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(os.path.join(output_dir, "problem2_ux_contour.png"), dpi=300)
    plt.close()

    line_x = hole_center[0]
    tol = 1e-8
    line_nodes = np.where(np.abs(mesh.coords[:,0]-line_x)<tol)[0]
    line_nodes = line_nodes[np.argsort(mesh.coords[line_nodes,1])]
    dist = mesh.coords[line_nodes,1] - (hole_center[1] + hole_radius)
    sigma_x_line = sigmax_nodal[line_nodes]

    plt.figure()
    plt.plot(dist, sigma_x_line, 'o-', label='FEM sigma_x')
    plt.axvline(x=0.0, color='r', linestyle='--', label='Hole boundary')
    plt.xlabel('Distance from hole boundary [m]')
    plt.ylabel('sigma_x [Pa]')
    plt.title('Sigma_x Distribution')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "problem2_sigma_x_lineplot.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
