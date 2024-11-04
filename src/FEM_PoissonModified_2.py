import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import matplotlib.gridspec as gridspec

from typing import Tuple, List, Dict
from scipy.sparse import csr_matrix, linalg
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from enum import Enum

"""
ほぼ全部書き直した #2
"""

class BoundaryType(Enum):
    DIRICHLET = 1
    NEUMANN = 2

@dataclass
class Node:
    id: int
    x: float
    y: float
    value: float = 0.0
    boundary_type: BoundaryType = None

@dataclass
class Element:
    id: int
    nodes: List[Node]
    
class GaussQuadrature:
    def __init__(self, order: int = 2):
        if order == 2:
            self.points = np.array([
                [-1/np.sqrt(3), -1/np.sqrt(3)],
                [ 1/np.sqrt(3), -1/np.sqrt(3)],
                [ 1/np.sqrt(3),  1/np.sqrt(3)],
                [-1/np.sqrt(3),  1/np.sqrt(3)]
            ])
            self.weights = np.ones(4)
        elif order == 3:
            r = np.sqrt(0.6)
            w1 = 5/9
            w2 = 8/9
            self.points = np.array([
                [-r, -r], [0, -r], [r, -r],
                [-r,  0], [0,  0], [r,  0],
                [-r,  r], [0,  r], [r,  r]
            ])
            self.weights = np.array([
                w1*w1, w1*w2, w1*w1,
                w1*w2, w2*w2, w1*w2,
                w1*w1, w1*w2, w1*w1
            ])

class ImprovedPoissonFEMSolver:
    def __init__(self, Lx: float = 0.5, Ly: float = 0.5):
        self.Lx = Lx
        self.Ly = Ly
        self.gauss = GaussQuadrature(order=3)
        self.logger = logging.getLogger(__name__)
        
    def create_mesh(self, nx: int, ny: int) -> Tuple[List[Node], List[Element]]:
        nodes = []
        node_id = 0
        
        for j in range(ny):
            for i in range(nx):
                x = i * self.Lx / (nx - 1)
                y = j * self.Ly / (ny - 1)
                
                boundary_type = None
                if abs(x - self.Lx) < 1e-10 or abs(y - self.Ly) < 1e-10:
                    boundary_type = BoundaryType.DIRICHLET
                elif abs(x) < 1e-10 or abs(y) < 1e-10:
                    boundary_type = BoundaryType.NEUMANN
                
                nodes.append(Node(node_id, x, y, boundary_type=boundary_type))
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
    
    def get_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = np.array([
            (1-xi)*(1-eta)/4,
            (1+xi)*(1-eta)/4,
            (1+xi)*(1+eta)/4,
            (1-xi)*(1+eta)/4
        ])
        
        dN_dxi = np.array([
            -(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4
        ])
        
        dN_deta = np.array([
            -(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4
        ])
        
        return N, dN_dxi, dN_deta

    def get_element_matrices(self, element: Element) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array([[node.x, node.y] for node in element.nodes])
        
        dx = abs(coords[1,0] - coords[0,0])
        dy = abs(coords[3,1] - coords[0,1])
        
        ke = np.zeros((4, 4))
        fe = np.zeros(4)
        
        for i, (xi, eta) in enumerate(self.gauss.points):
            w = self.gauss.weights[i]
            
            N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)
            
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
            
            stabilization = 1e-10 * np.eye(4)
            ke += w * det_J * (B.T @ B + stabilization)
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
            if node.boundary_type == BoundaryType.DIRICHLET:
                K[node.id, :] = 0
                K[node.id, node.id] = 1
                F[node.id] = 0
            elif node.boundary_type == BoundaryType.NEUMANN:
                pass
        
        return csr_matrix(K), F
    
    def solve(self, nx: int, ny: int) -> Tuple[np.ndarray, List[Node], List[Element]]:
        nodes, elements = self.create_mesh(nx, ny)
        K, F = self.assemble_system(nodes, elements)
        u = spsolve(K, F)
        
        for i, node in enumerate(nodes):
            node.value = u[i]
            
        return u, nodes, elements

    def compute_error(self, u: np.ndarray, nodes: List[Node]) -> Dict[str, float]:
        exact_center = 0.07367
        
        center_idx = len(nodes) // 2
        center_value = u[center_idx]
        
        rel_error = abs(center_value - exact_center) / exact_center
        
        return {
            'center_value': center_value,
            'relative_error': rel_error,
            'l2_error': np.sqrt(np.mean((u - exact_center)**2))
        }

class EnhancedVisualization:
    @staticmethod
    def plot_comprehensive_results(nodes: List[Node], nx: int, ny: int, case_name: str,
                                 error_data: Dict[str, float]):
        """Create comprehensive visualization with enhanced graphics"""
        if not os.path.exists('./img_ps'):
            os.makedirs('./img_ps')
            
        X = np.array([node.x for node in nodes]).reshape(ny, nx)
        Y = np.array([node.y for node in nodes]).reshape(ny, nx)
        U = np.array([node.value for node in nodes]).reshape(ny, nx)
        
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        ax1 = plt.subplot(gs[0, 0])
        cont = ax1.contourf(X, Y, U, 30, cmap='viridis')
        plt.colorbar(cont, ax=ax1)
        ax1.set_title('Solution Contours')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        ax2 = plt.subplot(gs[0, 1], projection='3d')
        surf = ax2.plot_surface(X, Y, U, cmap='viridis', antialiased=True)
        plt.colorbar(surf, ax=ax2)
        ax2.set_title('3D Surface')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.view_init(elev=30, azim=45)
        
        ax3 = plt.subplot(gs[1, :])
        
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
        
        ax3.plot(x_exact, u_exact, 'k-', label='Exact Solution', linewidth=2)
        ax3.plot(x_plot, u_plot, 'bo-', label='FEM Solution', markersize=6)
        
        error_text = f'Center Value: {error_data["center_value"]:.6f}\n'
        error_text += f'Relative Error: {error_data["relative_error"]*100:.2f}%\n'
        error_text += f'L2 Error: {error_data["l2_error"]:.6f}'
        ax3.text(0.02, 0.95, error_text, transform=ax3.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('u')
        ax3.set_title('Solution Comparison along y=0')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'./img_ps/{case_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    solver = ImprovedPoissonFEMSolver()
    
    cases = {
        '2-element': (3, 3),
        '8-element': (5, 5),
        '18-element': (7, 7)
    }
    
    convergence_data = {}
    
    for case_name, (nx, ny) in cases.items():
        logger.info(f"\nSolving {case_name} case:")
        
        u, nodes, elements = solver.solve(nx, ny)
        
        error_data = solver.compute_error(u, nodes)
        convergence_data[nx] = error_data
        
        logger.info(f"Center point value: {error_data['center_value']:.6f}")
        logger.info(f"Relative error: {error_data['relative_error']*100:.2f}%")
        
        EnhancedVisualization.plot_comprehensive_results(
            nodes, nx, ny, case_name, error_data)
    
    mesh_sizes = list(convergence_data.keys())
    errors = [data['relative_error'] for data in convergence_data.values()]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(mesh_sizes, errors, 'bo-', label='Convergence Rate')
    plt.grid(True)
    plt.xlabel('Mesh Size (N)')
    plt.ylabel('Relative Error')
    plt.title('Convergence Study')
    plt.savefig('./img_ps/convergence_study.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()