class CostFunction():
    def __init__(self):
       self.name = "parabola"
    def run(self, x ):
        return sum([x[i]**2 for i in range(len(x))])       
    def size(self):
        return 2

