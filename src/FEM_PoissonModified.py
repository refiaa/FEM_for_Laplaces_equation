import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import matplotlib.gridspec as gridspec
from typing import Tuple, List
from scipy.sparse import csr_matrix, linalg
from scipy.sparse.linalg import spsolve
from matplotlib.ticker import AutoMinorLocator

"""
ほぼ全部書き直した 

domainの違いかNeumann境界での問題かも知れないけど解の誤差がむしろ増加している。
正直わからん
"""

class Node:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.value = 0.0

class Element:
    def __init__(self, id: int, nodes: List[Node]):
        self.id = id
        self.nodes = nodes

class PoissonFEMSolver:
    def __init__(self, Lx: float = 0.5, Ly: float = 0.5):
        self.Lx = Lx
        self.Ly = Ly

    def create_mesh(self, nx: int, ny: int) -> Tuple[List[Node], List[Element]]:
        nodes = []
        node_id = 0
        
        for j in range(ny):
            for i in range(nx):
                x = i * self.Lx / (nx - 1)
                y = j * self.Ly / (ny - 1)
                nodes.append(Node(node_id, x, y))
                node_id += 1

        elements = []
        for j in range(ny-1):
            for i in range(nx-1):
                node_ids = [
                    i + j*nx,
                    i + 1 + j*nx,
                    i + 1 + (j+1)*nx,
                    i + (j+1)*nx
                ]
                element_nodes = [nodes[id] for id in node_ids]
                elements.append(Element(len(elements), element_nodes))

        return nodes, elements

    def get_element_matrices(self, element: Element) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array([[node.x, node.y] for node in element.nodes])
        
        # element dim
        dx = abs(coords[1,0] - coords[0,0])
        dy = abs(coords[3,1] - coords[0,1])
        
        gauss_points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)],
            [ 1/np.sqrt(3), -1/np.sqrt(3)],
            [ 1/np.sqrt(3),  1/np.sqrt(3)],
            [-1/np.sqrt(3),  1/np.sqrt(3)]
        ])
        weights = np.ones(4)
        
        ke = np.zeros((4, 4))
        fe = np.zeros(4)
        
        for i in range(4):
            xi, eta = gauss_points[i]
            w = weights[i]
            
            dN_dxi = np.array([
                -(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4
            ])
            dN_deta = np.array([
                -(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4
            ])
            
            J = np.array([
                [dx/2, 0],
                [0, dy/2]
            ])
            J_inv = np.linalg.inv(J)
            det_J = np.linalg.det(J)
            
            B = np.vstack([
                J_inv[0,0] * dN_dxi + J_inv[0,1] * dN_deta,
                J_inv[1,0] * dN_dxi + J_inv[1,1] * dN_deta
            ])
            
            N = np.array([
                (1-xi)*(1-eta)/4,
                (1+xi)*(1-eta)/4,
                (1+xi)*(1+eta)/4,
                (1-xi)*(1+eta)/4
            ])
            
            ke += w * det_J * (B.T @ B)
            fe += w * det_J * N
        
        return ke, fe

    def assemble_system(self, nodes: List[Node], elements: List[Element]) -> Tuple[csr_matrix, np.ndarray]:
        n_nodes = len(nodes)
        K = np.zeros((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        for element in elements:
            ke, fe = self.get_element_matrices(element)
            
            for i, node_i in enumerate(element.nodes):
                F[node_i.id] += fe[i]
                for j, node_j in enumerate(element.nodes):
                    K[node_i.id, node_j.id] += ke[i,j]
        
        for node in nodes:
            if abs(node.x - self.Lx) < 1e-10 or abs(node.y - self.Ly) < 1e-10:
                K[node.id, :] = 0
                K[node.id, node.id] = 1
                F[node.id] = 0
        
        return csr_matrix(K), F

    def solve(self, nx: int, ny: int) -> Tuple[np.ndarray, List[Node], List[Element]]:
        nodes, elements = self.create_mesh(nx, ny)
        K, F = self.assemble_system(nodes, elements)
        u = spsolve(K, F)
        
        for i, node in enumerate(nodes):
            node.value = u[i]
            
        return u, nodes, elements

class Visualization:
    @staticmethod
    def plot_results(nodes: List[Node], nx: int, ny: int, case_name: str):
        if not os.path.exists('./img_ps'):
            os.makedirs('./img_ps')
            
        X = np.array([node.x for node in nodes]).reshape(ny, nx)
        Y = np.array([node.y for node in nodes]).reshape(ny, nx)
        U = np.array([node.value for node in nodes]).reshape(ny, nx)
        
        fig = plt.figure(figsize=(20, 12))
        
        ax1 = plt.subplot(221)
        cont = ax1.contourf(X, Y, U, 20, cmap='viridis')
        plt.colorbar(cont, ax=ax1)
        ax1.set_title('Solution Contours')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        ax2 = plt.subplot(222, projection='3d')
        surf = ax2.plot_surface(X, Y, U, cmap='viridis')
        plt.colorbar(surf, ax=ax2)
        ax2.set_title('3D Surface')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        
        ax3 = plt.subplot(212)
        
        x_plot = []
        u_plot = []

        for node in nodes:
            if abs(node.y) < 1e-10:
                x_plot.append(node.x)
                u_plot.append(node.value)
        
        idx = np.argsort(x_plot)
        x_plot = np.array(x_plot)[idx]
        u_plot = np.array(u_plot)[idx]
        
        x_exact = np.linspace(0, 0.5, 100)
        u_exact = 0.07367 * (1 - (2*x_exact/0.5)**2)
        
        ax3.plot(x_exact*2, u_exact, 'k-', label='Exact Solution', linewidth=2)
        ax3.plot(x_plot*2, u_plot, 'bo-', label='FEM Solution', markersize=6)
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('u')
        ax3.set_title('Solution Comparison along y=0')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 0.1)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'./img_ps/{case_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    solver = PoissonFEMSolver()
    
    cases = {
        '2-element': (3, 3),
        '8-element': (5, 5),
        '18-element': (7, 7)
    }
    
    for case_name, (nx, ny) in cases.items():
        print(f"\nSolving {case_name} case:")
        u, nodes, elements = solver.solve(nx, ny)
        
        center_idx = len(nodes) // 2
        center_value = u[center_idx]
        print(f"Center point value: {center_value:.6f}")
        print(f"Error vs exact: {abs(center_value - 0.07367)/0.07367*100:.2f}%")
        
        Visualization.plot_results(nodes, nx, ny, case_name)

if __name__ == "__main__":
    main()