import numpy as np
from scipy.optimize import linprog

class LabellingFunction:
    
    def __init__(self):
        pass
    
    def to_hyperplanes(self):
        pass
    
    def in_label(self):
        pass
    


class LabellingFunctionZonotope(LabellingFunction):
    def __init__(self, center, generators):
        self.center = center
        self.generators = generators
        self.inequalities = self.to_hyperplanes()
    
    def to_hyperplanes(self):
        """
        Convert the zonotope to a set of hyperplane inequalities.
        Each generator contributes two hyperplanes.
        """
        c = self.center
        G = np.array(self.generators)

        inequalities = []
        for g in G:
            # Create two inequalities for each generator
            inequalities.append((g, np.dot(g, c) + 1))  # Positive direction
            inequalities.append((-g, -np.dot(g, c) + 1)) # Negative direction
        return inequalities
    
    def in_label(self, y):
        """
        Check whether the numpy array `y` is contained within the zonotope using Linear Programming.
        """
        y = np.array(y, dtype=np.float32)
        G = np.array(self.generators)
        c = self.center
        
        # Number of generators
        num_generators = G.shape[0]
        
        # Objective: Minimize the auxiliary variable t
        # The variable vector x will have size (num_generators + 1) where the last element is t
        c_lp = np.zeros(num_generators + 1)
        c_lp[-1] = 1  # We want to minimize the last variable (t)
        
        # Constraints: y = Gx + c, and -t <= x_i <= t
        A_eq = np.hstack([G.T, np.zeros((G.shape[1], 1))])  # G * x = y - c, so A_eq is G and b_eq is y - c
        b_eq = y - c
        
        # Inequality constraints for the t variable (infinity norm)
        A_ub = np.vstack([np.hstack([np.eye(num_generators), -np.ones((num_generators, 1))]),
                          np.hstack([-np.eye(num_generators), -np.ones((num_generators, 1))])])
        b_ub = np.ones(2 * num_generators)
        
        # Bounds: x_i has no explicit bounds; t >= 0
        bounds = [(None, None)] * num_generators + [(0, None)]
        
        # Solve the LP problem
        res = linprog(c_lp, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='highs')
        
        # Check if the solution is feasible and if t <= 1
        if res.success and res.x[-1] <= 1:
            return True
        else:
            return False
            
            
            
    
# class LabellingFunctionBox:
#     def __init__(self, lower, upper):
#         self.lower = lower
#         self.upper = upper
        
#     def in_box
    
    