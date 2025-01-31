import numpy as np
import matplotlib.pyplot as plt

class GradientDescentVisualizer:
    def __init__(self, name, function=None): 
        self.name = name
        self.function = function
        self.path = []  # Store path of gradient descent
        self.minimum = None  # Store found minimum
        
    def assign_function(self, coefficients, x=None):
        """Takes a list of Coeffiecients to define a function"""
        self.coefficients = coefficients  #example: [1, -5, 25] means x^2 - 5x + 25
        
        if x is None:
            # Just store the coefficients and create the function
            def poly_function(x):
                my_function = 0
                for i in range(len(self.coefficients)):
                    my_function += self.coefficients[i] * (x**(len(self.coefficients)-1-i))
                return my_function
            self.function = poly_function
            return None
        else:
            # Evaluate the function at x
            my_function = 0
            for i in range(len(self.coefficients)):
                my_function += self.coefficients[i] * (x**(len(self.coefficients)-1-i))
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
        p = self.coefficients[0]  # Start with highest degree coefficient
        d = 0  # Derivative starts at 0

        for i in range(1, n):
            d = p + x*d  # Derivative calculation
            p = self.coefficients[i] + x*p  # Polynomial calculation
        
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

# Test:

quadratic_curve = GradientDescentVisualizer("Quadratic Curve", None)
quadratic_curve.assign_function([1, -5, 25, 0, 8], None)  # Just store the function
print(f"Degree of polynomial: {quadratic_curve.highest_degree()}")  # Should print 2
print(quadratic_curve.evaluate(4))
print(quadratic_curve.gradient(2))

#print(f"f(2) = {quadratic_curve.evaluate(2)}")  # Now we can evaluate at any x using evaluate

# Example usage:
# if __name__ == '__main__':
#     # Create visualizer for f(x) = x^2
#     gd_vis = GradientDescentVisualizer("Quadratic Function")
#     gd_vis.assign_function(lambda x: x**2)
    
#     # Perform gradient descent
#     x = 2.0  # Starting point
#     learning_rate = 0.1
    
#     for _ in range(20):
#         x = gd_vis.descent_step(x, learning_rate)
    
#     gd_vis.minimum = (x, gd_vis.evaluate(x))
    
#     # Plot results
#     gd_vis.plot(-3, 3)

# if __name__ == '__main__':
