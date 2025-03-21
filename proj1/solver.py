import sympy as sp

# Define the symbols
a, b, delta, theta = sp.symbols('a b delta theta')

# Define the expression inside arctan
tan_term = (a * sp.tan(delta)) / b

# Define the arctan term
arctan_term = sp.atan(tan_term)

# Left-hand side of the equation
lhs = sp.sin(arctan_term + theta)

# Right-hand side of the equation
rhs = sp.sin(arctan_term) / a

# Solve the equation
equation = sp.Eq(lhs, rhs)
solution = sp.solve(equation, theta)

print(solution)
solution = sp.solve(equation,delta)

print(solution)
