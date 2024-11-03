import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from dataclasses import dataclass
from typing import Callable, Tuple, Dict
from matplotlib.ticker import ScalarFormatter

@dataclass
class BoundaryCond:
    type: str
    value: Callable[[float, float], float]

class LaplaceSolver:
    def __init__(self, nx: int, ny: int, boundary_conditions: dict, max_iter: int = 20000, tol: float = 1e-6):
        self.nx = nx
        self.ny = ny
        self.boundary_conditions = boundary_conditions
        self.max_iter = max_iter
        self.tol = tol
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self.u = np.zeros((ny, nx))
        self.convergence_history = []
        self.setupBoundaryConditions()
        self.initialize_solution()
    
    def initialize_solution(self):
        y = np.linspace(0, 1, self.ny)
        x = np.linspace(0, 1, self.nx)
        X, Y = np.meshgrid(x, y)
        
        if 'left' in self.boundary_conditions and self.boundary_conditions['left'].type == 'Dirichlet':
            for j in range(self.ny):
                y_val = j * self.dy
                if abs(y_val - 1.0) < self.dy/2:
                    self.u[j, :] = 1.0 * (1 - X[j,:])
                else:
                    self.u[j, :] = 0.0
                    
        if 'bottom' in self.boundary_conditions and self.boundary_conditions['bottom'].type == 'Dirichlet':
            for i in range(self.nx):
                self.u[:, i] *= (1 - Y[:,i])
    
    def setupBoundaryConditions(self):
        for boundary, condition in self.boundary_conditions.items():
            if condition.type == 'Dirichlet':
                if boundary == 'left':
                    for j in range(self.ny):
                        y = j * self.dy
                        if abs(y - 1.0) < self.dy/2:
                            self.u[j, 0] = 1.0
                        else:
                            self.u[j, 0] = 0.0
                elif boundary == 'right':
                    for j in range(self.ny):
                        y = j * self.dy
                        self.u[j, -1] = condition.value(1, y)
                elif boundary == 'bottom':
                    for i in range(self.nx):
                        x = i * self.dx
                        self.u[0, i] = condition.value(x, 0)
                elif boundary == 'top':
                    for i in range(self.nx):
                        x = i * self.dx
                        self.u[-1, i] = condition.value(x, 1)

    def solve(self) -> Tuple[np.ndarray, list]:
        h = min(self.dx, self.dy)
        omega = 2.0 / (1 + np.sin(np.pi * h))
        
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            max_diff = 0.0
            
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    u_new = 0.25 * (self.u[j+1, i] + self.u[j-1, i] +
                                  self.u[j, i+1] + self.u[j, i-1])
                    self.u[j, i] = (1 - omega) * u_old[j, i] + omega * u_new
            
            self.setupBoundaryConditions()
            self.applyNeumannConditions()
            
            diff = np.linalg.norm(self.u - u_old) / (np.linalg.norm(u_old) + 1e-10)
            self.convergence_history.append(diff)
            
            if diff < self.tol:
                print(f'Converged after {iteration+1} iterations. Final residual: {diff:.2e}')
                return self.u, self.convergence_history
            
            if not np.isfinite(diff):
                print("Solution diverged")
                return self.u, self.convergence_history
        
        print(f'Warning: Did not converge within {self.max_iter} iterations. Final residual: {diff:.2e}')
        return self.u, self.convergence_history

    def applyNeumannConditions(self):
        for boundary, condition in self.boundary_conditions.items():
            if condition.type == 'Neumann':
                if boundary == 'left':
                    self.u[:, 0] = self.u[:, 1]
                elif boundary == 'right':
                    self.u[:, -1] = self.u[:, -2]
                elif boundary == 'bottom':
                    self.u[0, :] = self.u[1, :]
                elif boundary == 'top':
                    self.u[-1, :] = self.u[-2, :]

    def checkNeumannCondition(self, boundary: str, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        if boundary == 'left':
            return np.allclose(self.u[:, 0], self.u[:, 1], rtol=rtol, atol=atol)
        elif boundary == 'right':
            return np.allclose(self.u[:, -1], self.u[:, -2], rtol=rtol, atol=atol)
        elif boundary == 'bottom':
            return np.allclose(self.u[0, :], self.u[1, :], rtol=rtol, atol=atol)
        elif boundary == 'top':
            return np.allclose(self.u[-1, :], self.u[-2, :], rtol=rtol, atol=atol)
        return False

    def getSolution(self) -> np.ndarray:
        return self.u

    def getConvergenceHistory(self) -> list:
        return self.convergence_history

class Visualizer:
    @staticmethod
    def plot_solution(u: np.ndarray, title: str = 'Laplace Equation Solution', filename: str = 'solution.png'):
        ny, nx = u.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        plt.figure(figsize=(20, 6))
        
        plt.subplots_adjust(wspace=0.7)
        
        plt.subplot(131)
        contour = plt.contourf(X, Y, u, 50, cmap='viridis')
        plt.colorbar(contour)
        plt.title(f'Contour Plot\n{title}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax = plt.subplot(132, projection='3d')
        surf = ax.plot_surface(X, Y, u, cmap='viridis')
        
        pos = ax.get_position()
        colorbar_ax = plt.gcf().add_axes([pos.x1 + 0.05, pos.y0, 0.01, pos.height])
        plt.colorbar(surf, cax=colorbar_ax)
        
        ax.view_init(elev=30, azim=45)
        ax.set_title(f'Surface Plot\n{title}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        ax.dist = 12
        
        ax_cross = plt.subplot(133)
        mid_x = nx // 2
        mid_y = ny // 2
        plt.plot(x, u[mid_y, :], 'b-', label='y=0.5 cross-section')
        plt.plot(y, u[:, mid_x], 'r--', label='x=0.5 cross-section')
        plt.title('Cross-sections')
        plt.xlabel('Coordinate')
        plt.ylabel('u')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        pos_cross = ax_cross.get_position()
        ax_cross.set_position([pos_cross.x0 + 0.05, pos_cross.y0, pos_cross.width, pos_cross.height])
        
        os.makedirs('./img', exist_ok=True)
        filepath = os.path.join('./img', filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    @staticmethod
    def plotConvergence(convergence_histories: Dict[int, list], title: str = 'Convergence', filename: str = 'convergence.png'):
        plt.figure(figsize=(10, 6))
        
        for nx, history in convergence_histories.items():
            plt.semilogy(range(1, len(history)+1), history, 
                      label=f'Mesh {nx}x{nx}', marker='o', markersize=2)
        
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Relative Error')
        plt.grid(True)
        plt.legend()
        
        os.makedirs('./img', exist_ok=True)
        filepath = os.path.join('./img', filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plotConvergenceAnalysis(mesh_sizes: list, errors: list, filename: str = 'convergence_analysis.png'):
        plt.figure(figsize=(8, 6))
        
        log_h = np.log([1.0/n for n in mesh_sizes])
        log_err = np.log(errors)
        
        valid_indices = np.isfinite(log_h) & np.isfinite(log_err)
        if np.any(valid_indices):
            slope, intercept = np.polyfit(log_h[valid_indices], log_err[valid_indices], 1)
        else:
            slope = np.nan
            
        plt.loglog(mesh_sizes, errors, 'bo-', label='Numerical Error')
        
        if np.isfinite(slope):
            h_ref = np.array([min(mesh_sizes), max(mesh_sizes)])
            plt.loglog(h_ref, np.exp(intercept) * (1.0/h_ref)**slope, 'r--', 
                      label=f'Slope: {slope:.2f}')
        
        plt.title('Convergence Analysis')
        plt.xlabel('Mesh Size (N)')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        
        os.makedirs('./img', exist_ok=True)
        filepath = os.path.join('./img', filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close()
        
        return slope

def defineBoundaryConditions_case1() -> dict:
    return {
        'left': BoundaryCond(
            type='Dirichlet',
            value=lambda x, y: 1.0 if abs(y - 1.0) < 1e-10 else 0.0
        ),
        'right': BoundaryCond(
            type='Neumann',
            value=lambda x, y: 0
        ),
        'bottom': BoundaryCond(
            type='Neumann',
            value=lambda x, y: 0
        ),
        'top': BoundaryCond(
            type='Neumann',
            value=lambda x, y: 0
        )
    }

def defineBoundaryConditions_case2() -> dict:
    return {
        'left': BoundaryCond(
            type='Dirichlet',
            value=lambda x, y: 1.0 if abs(y - 1.0) < 1e-10 else 0.0
        ),
        'right': BoundaryCond(
            type='Neumann',
            value=lambda x, y: 0
        ),
        'bottom': BoundaryCond(
            type='Dirichlet',
            value=lambda x, y: 0
        ),
        'top': BoundaryCond(
            type='Neumann',
            value=lambda x, y: 0
        )
    }

def checkConditions_case1(solver: LaplaceSolver, rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, Dict[str, bool]]:
    u = solver.getSolution()
    
    dirichlet_satisfied = np.isclose(u[-1, 0], 1.0, rtol=rtol, atol=atol)
    
    neumann_checks = {
        'right': solver.checkNeumannCondition('right', rtol, atol),
        'bottom': solver.checkNeumannCondition('bottom', rtol, atol),
        'top': solver.checkNeumannCondition('top', rtol, atol)
    }
    
    return dirichlet_satisfied, neumann_checks

def checkConditions_case2(solver: LaplaceSolver, rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, Dict[str, bool]]:
    u = solver.getSolution()
    
    dirichlet_satisfied = (
        np.all(np.isclose(u[0, :], 0, rtol=rtol, atol=atol)) and
        np.isclose(u[-1, 0], 1.0, rtol=rtol, atol=atol)
    )
    
    neumann_checks = {
        'right': solver.checkNeumannCondition('right', rtol, atol),
        'top': solver.checkNeumannCondition('top', rtol, atol)
    }
    
    return dirichlet_satisfied, neumann_checks

def analyzeConvergence(solutions: Dict[int, np.ndarray]) -> Tuple[list, list]:
    mesh_sizes = sorted(solutions.keys())
    errors = []
    
    for i in range(len(mesh_sizes)-1):
        nx_coarse = mesh_sizes[i]
        nx_fine = mesh_sizes[i+1]
        
        u_coarse = solutions[nx_coarse]
        u_fine = solutions[nx_fine]
        
        error = computeError(u_coarse[::2,::2], u_fine)
        errors.append(error)
    
    return mesh_sizes[:-1], errors

def computeError(u1: np.ndarray, u2: np.ndarray) -> float:
    if u1.shape != u2.shape:
        raise ValueError(f"Solutions have different shapes: {u1.shape} vs {u2.shape}")
    
    error = np.sqrt(np.mean((u1 - u2)**2))

    if np.all(u2 == 0):
        return error
    return error / np.sqrt(np.mean(u2**2))

def main():
    mesh_sizes = [10, 20, 40, 80, 160]
    
    solutions_case1 = {}
    solutions_case2 = {}
    convergence_histories_case1 = {}
    convergence_histories_case2 = {}
    
    print("\nSolving Case 1")
    print("==============")
    for nx in mesh_sizes:
        ny = nx
        
        solver = LaplaceSolver(
            nx=nx,
            ny=ny,
            boundary_conditions=defineBoundaryConditions_case1()
        )
        u, convergence_history = solver.solve()
        solutions_case1[nx] = u
        convergence_histories_case1[nx] = convergence_history
        
        dirichlet_satisfied, neumann_checks = checkConditions_case1(solver)
        print(f'\nMesh {nx}x{nx}:')
        print(f'Dirichlet conditions satisfied: {dirichlet_satisfied}')
        print('Neumann conditions satisfied:')
        for boundary, satisfied in neumann_checks.items():
            print(f'  {boundary}: {satisfied}')
        
        filename = f'case1_mesh_{nx}x{ny}.png'
        Visualizer.plot_solution(u, title=f'Case 1 - Mesh {nx}x{ny}', filename=filename)
    
    if len(solutions_case1) > 1:
        mesh_sizes_analysis, errors = analyzeConvergence(solutions_case1)
        convergence_rate = Visualizer.plotConvergenceAnalysis(
            mesh_sizes_analysis, 
            errors, 
            filename='case1_convergence_analysis.png'
        )
        print(f'\nCase 1 Convergence Rate: {convergence_rate:.2f}')
        
        Visualizer.plotConvergence(
            convergence_histories_case1,
            title='Case 1 Convergence History',
            filename='case1_convergence_history.png'
        )
    
    print("\nSolving Case 2")
    print("==============")
    for nx in mesh_sizes:
        ny = nx
        solver = LaplaceSolver(
            nx=nx,
            ny=ny,
            boundary_conditions=defineBoundaryConditions_case2()
        )
        u, convergence_history = solver.solve()
        solutions_case2[nx] = u
        convergence_histories_case2[nx] = convergence_history
        
        dirichlet_satisfied, neumann_checks = checkConditions_case2(solver)
        print(f'\nMesh {nx}x{nx}:')
        print(f'Dirichlet conditions satisfied: {dirichlet_satisfied}')
        print('Neumann conditions satisfied:')
        for boundary, satisfied in neumann_checks.items():
            print(f'  {boundary}: {satisfied}')
        
        filename = f'case2_mesh_{nx}x{ny}.png'
        Visualizer.plot_solution(u, title=f'Case 2 - Mesh {nx}x{ny}', filename=filename)
    
    if len(solutions_case2) > 1:
        mesh_sizes_analysis, errors = analyzeConvergence(solutions_case2)
        convergence_rate = Visualizer.plotConvergenceAnalysis(
            mesh_sizes_analysis, 
            errors, 
            filename='case2_convergence_analysis.png'
        )
        print(f'\nCase 2 Convergence Rate: {convergence_rate:.2f}')
        
        Visualizer.plotConvergence(
            convergence_histories_case2,
            title='Case 2 Convergence History',
            filename='case2_convergence_history.png'
        )

if __name__ == "__main__":
    main()