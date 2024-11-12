class SimpleLinearRegression:
    
    def __init__(self):
        # Dataset hardcoded
        self.x = [23, 26, 30, 34, 43, 48, 52, 57, 58]  # Advertising
        self.y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]  # Sales
        self.n = len(self.x)  
    
    #Sumas necesarias para la regresión lineal
    def calculate_sums(self):
        self.sum_x = sum(self.x)
        self.sum_y = sum(self.y)
        self.sum_xy = sum(xi * yi for xi, yi in zip(self.x, self.y))
        self.sum_x2 = sum(xi ** 2 for xi in self.x)

    #Calcular B0
    def calculate_b0(self, b1):
        numerator = (self.sum_x2 * self.sum_y) - (self.sum_x * self.sum_xy)
        denominator = (self.n * self.sum_x2) - (self.sum_x ** 2)
        return numerator / denominator

    #Calcular B1
    def calculate_b1(self):
        numerator = (self.n * self.sum_xy) - (self.sum_x * self.sum_y)
        denominator = (self.n * self.sum_x2) - (self.sum_x ** 2)
        return numerator / denominator

    # Calcular las sumas, luego calcular B1 y B0
    def fit(self):
        self.calculate_sums()
        b1 = self.calculate_b1()
        b0 = self.calculate_b0(b1)
        return b0, b1

    # Predicción para un nuevo valor de x 
    def predict(self, x_value, b0, b1):
        return b0 + b1 * x_value



if __name__ == "__main__":
    model = SimpleLinearRegression()
    b0, b1 = model.fit()

    # Ecuación de la regresión
    print(f"Ecuación de Regresión: y = {b0:.2f} + {b1:.2f} * x")

    # Solicitar valor de Advertising al usuario
    try:
        x_value = float(input("Ingrese un valor de Advertising para predecir las Sales: "))
        y_pred = model.predict(x_value, b0, b1)
        print(f"Predicción para Advertising = {x_value}: Sales ≈ {y_pred:.2f}")
    except ValueError:
        print("Por favor, ingrese un número válido.")
