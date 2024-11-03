import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from dataclasses import dataclass
from typing import Callable, Tuple, Dict
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import RectBivariateSpline

# TODO: Poisson's eq.もできるように修正する。

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
        
        for j in range(self.ny):
            for i in range(self.nx):
                if 'left' in self.boundary_conditions and self.boundary_conditions['left'].type == 'Dirichlet':
                    value = self.boundary_conditions['left'].value(0, j*self.dy)
                    self.u[j, i] = value * (1 - X[j,i])

    def setupBoundaryConditions(self):
        for boundary, condition in self.boundary_conditions.items():
            if condition.type == 'Dirichlet':
                if boundary == 'left':
                    for j in range(self.ny):
                        y = j * self.dy
                        self.u[j, 0] = condition.value(0, y)
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
        omega = 1.8
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            
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
                break
        else:
            print(f'Warning: Did not converge within {self.max_iter} iterations. Final residual: {diff:.2e}')
        
        return self.u, self.convergence_history

    def applyNeumannConditions(self):
        """∂u/∂n = 0"""
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
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        contour = plt.contourf(X, Y, u, 50, cmap='viridis')
        plt.colorbar(contour)
        plt.title(f'Contour Plot\n{title}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax = plt.subplot(132, projection='3d')
        surf = ax.plot_surface(X, Y, u, cmap='viridis')
        plt.colorbar(surf)
        ax.set_title(f'Surface Plot\n{title}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        
        plt.subplot(133)
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
        os.makedirs('./img', exist_ok=True)
        filepath = os.path.join('./img', filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight')
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
        
        valid_indices = np.isfinite(errors)
        if np.any(valid_indices):
            log_h = np.log([1/n for n in mesh_sizes])
            log_err = np.log(errors)
            slope, intercept = np.polyfit(log_h[valid_indices], log_err[valid_indices], 1)
            
            plt.loglog(mesh_sizes, errors, 'bo-', label='Numerical Error')
            h = np.array([min(mesh_sizes), max(mesh_sizes)])
            plt.loglog(h, np.exp(intercept) * (1/h)**slope, 'r--', 
                      label=f'Slope: {slope:.2f}')
        else:
            slope = np.nan
            plt.loglog(mesh_sizes, errors, 'bo-', label='Numerical Error')
        
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
    
    u_ref = solutions[mesh_sizes[-1]]
    ny_ref, nx_ref = u_ref.shape
    
    for nx in mesh_sizes[:-1]:
        u = solutions[nx]
        ny, nx = u.shape
        
        x_coarse = np.linspace(0, 1, nx)
        y_coarse = np.linspace(0, 1, ny)
        x_fine = np.linspace(0, 1, nx_ref)
        y_fine = np.linspace(0, 1, ny_ref)
        
        interp = RectBivariateSpline(y_coarse, x_coarse, u)
        u_interp = interp(y_fine, x_fine)
        
        error = np.linalg.norm(u_interp - u_ref) / np.linalg.norm(u_ref)
        errors.append(error)
    
    return mesh_sizes[:-1], errors

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
        solutions_case1[nx] = u.copy()
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
        solutions_case2[nx] = u.copy()
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
        try:
            mesh_sizes_analysis, errors = analyzeConvergence(solutions_case2)
            convergence_rate = Visualizer.plotConvergenceAnalysis(
                mesh_sizes_analysis, 
                errors, 
                filename='case2_convergence_analysis.png'
            )
            print(f'\nCase 2 Convergence Rate: {convergence_rate:.2f}')
        except Exception as e:
            print(f'\nError in convergence analysis for Case 2: {str(e)}')
        
        Visualizer.plotConvergence(
            convergence_histories_case2,
            title='Case 2 Convergence History',
            filename='case2_convergence_history.png'
        )

if __name__ == "__main__":
    main()