import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescentVisualizer:
    def __init__(self, name, function=None): 
        self.name = name
        self.function = function
        self.path = []  # Store path of gradient descent
        self.minimum = None  # Store found minimum
        
    def assign_function(self, coefficients, x=None):
        """Takes a list of Coeffiecients to define a function"""
        self.coefficients = coefficients  #example: [1, -5, 25] means x^2 - 5x + 25

        """
        When you assign a function, you'll have the ability to define the coefficients and either
        A) define what independent value you want to evaluate the function at or B) say "None" and insert independent
        values at a later time
        """
        
        if x is None:
            # Just store the coefficients and create the function
            def poly_function(x):
                my_function = 0
                for i in range(len(self.coefficients)):
                    my_function += self.coefficients[i] * (x**(len(self.coefficients)-i-1))
                return my_function
            self.function = poly_function
            return None
        else:
            # Evaluate the function at x
            my_function = 0
            for i in range(len(self.coefficients)):
                my_function += self.coefficients[i] * (x**(len(self.coefficients)-i-1))
            # Example:
            # When i=0: 1 * x^2
            # When i=1: -5 * x^1
            # When i=2: 25 * x^0
            return my_function

    def highest_degree(self):
        """Returns highest power in the polynomial"""
        return len(self.coefficients) - 1
        
    def horner(self, x):
        """
        Horner method is the superior numerical method for evaluating polynomial derivatives, as it uses 
        nested multiplications making it very efficient.

        Horner's method to evaluate polynomial and its derivative
        Returns: (p, d) where p is polynomial value and d is derivative value
        """
        n = len(self.coefficients)
        p = self.coefficients[0]  
        d = 0  

        for i in range(1, n):
            d = p + x*d  
            p = self.coefficients[i] + x*p  
        
        return p, d

    def evaluate(self, x):
        """Evaluate the function at point x"""
        p, _ = self.horner(x)  # Only need polynomial value
        return p
    
    def gradient(self, x):
        """Calculate gradient (derivative) at point x"""
        _, d = self.horner(x)  # Only need derivative value
        return d
    
    def descent_step(self, x, learning_rate):
        """Take one step of gradient descent"""
        grad = self.gradient(x)
        new_x = x - learning_rate * grad
        self.path.append((x, self.evaluate(x)))
        return new_x
    
    def plot(self, start, end):
        """Plot function and gradient descent path"""
        x = np.linspace(start, end, 1000)
        y = [self.evaluate(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label=self.name)
        
        if self.path:
            path_x, path_y = zip(*self.path)
            plt.plot(path_x, path_y, 'ro-', label='Gradient Path')
            
        if self.minimum:
            plt.plot(self.minimum[0], self.minimum[1], 'g*', 
                    markersize=15, label='Minimum')
            
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'{self.name} with Gradient Descent Path')
        plt.legend()
        plt.show()

class GradientDescentVisualizer3D:
    def __init__(self, name):
        self.name = name
        self.path = []  # Will store (x, y, z) points of descent path
        self.minimum = None
        
    def evaluate(self, x, y):
        """Example function f(x,y) = x² + y²"""
        return x**2 + y**2
    
    def gradient(self, x, y):
        """Calculate gradient [∂f/∂x, ∂f/∂y]"""
        h = 1e-7
        dx = (self.evaluate(x + h, y) - self.evaluate(x, y)) / h
        dy = (self.evaluate(x, y + h) - self.evaluate(x, y)) / h
        return np.array([dx, dy])
    
    def descent_step(self, point, learning_rate):
        """Take one step of gradient descent"""
        x, y = point
        grad = self.gradient(x, y)
        new_point = point - learning_rate * grad
        self.path.append((x, y, self.evaluate(x, y)))
        return new_point
    
    def plot(self, x_range=(-5, 5), y_range=(-5, 5)):
        # Create grid of points
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(X, Y)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Plot gradient descent path if it exists
        if self.path:
            path = np.array(self.path)
            ax.plot3D(path[:,0], path[:,1], path[:,2], 'r.-', 
                     linewidth=2, markersize=10, label='Gradient Descent Path')
        
        # Plot minimum if found
        if self.minimum is not None:
            ax.scatter(*self.minimum, color='red', s=100, label='Minimum')
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{self.name} with Gradient Descent Path')
        
        # Add color bar
        fig.colorbar(surface)
        plt.legend()
        plt.show()

# Test:

quadratic_curve = GradientDescentVisualizer("Quadratic Curve", None)
quadratic_curve.assign_function([1, -5, 25, 0, 8], None)  # Just store the function
print(f"Degree of polynomial: {quadratic_curve.highest_degree()}")  # Should print 2
print(quadratic_curve.evaluate(4))
print(quadratic_curve.gradient(2))

#print(f"f(2) = {quadratic_curve.evaluate(2)}")  # Now we can evaluate at any x using evaluate

# Example usage:
if __name__ == '__main__':
    # Create visualizer
    gd_vis = GradientDescentVisualizer3D("Bowl Function")
    
    # Starting point
    point = np.array([4.0, 4.0])
    learning_rate = 0.1
    
    # Perform gradient descent
    for _ in range(30):
        point = gd_vis.descent_step(point, learning_rate)
    
    # Store minimum
    gd_vis.minimum = (*point, gd_vis.evaluate(*point))
    
    # Plot results
    gd_vis.plot()
