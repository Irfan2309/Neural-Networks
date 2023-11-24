import numpy as np

# Particle Class for PSO Algorithm
class Particle:
    # Constructor to initialize the particle
    def __init__(self):
        self.position = None
        self.velocity = None
        self.boundary = None
        self.pbest_position = None
        self.pbest_fitness = float('inf')
        self.fitness = 0
        self.informants = []

    # Function to set boundary
    def setBounds(self, boundary):
        self.boundary = boundary

    # Function to initialize position and velocity
    def initialize(self, weightBiasCount):
        # Setting the lower and upper bounds
        lowBound = self.boundary[0][0]
        highBound = self.boundary[0][1]
        
        # Initializing the position and velocity
        self.position = np.random.uniform(lowBound, highBound, weightBiasCount)
        self.pbest_position = self.position
        self.velocity = np.zeros(weightBiasCount)
        

    # Function to calculate velocity
    def updateVelocity(self, alpha, beta, gamma, delta, ibest_position, gbest_position):
        for i in range(len(self.velocity)):
            # Generating random numbers
            c1 = np.random.random()
            c2 = np.random.random()
            c3 = np.random.random()

            # Calculating the velocity
            cognitive = beta * c1 * (self.pbest_position[i] - self.position[i])
            social = gamma * c2 * (ibest_position[i] - self.position[i])
            global_ = delta * c3 * (gbest_position[i] - self.position[i])
            # Updating the velocity
            self.velocity[i] = alpha * self.velocity[i] + cognitive + social + global_
    
    # Function to update position
    def updatePosition(self):
        # Updating the position
        self.position = self.position + self.velocity

        # Checking if the position is within the bounds
        for i in range(len(self.position)):
            if self.position[i] < self.boundary[i][0]:
                # If not, set it to the lower bound
                self.position[i] = self.boundary[i][0]

            elif self.position[i] > self.boundary[i][1]:
                # If not, set it to the upper bound
                self.position[i] = self.boundary[i][1]
    
    # Function to set informants
    def setInformants(self,informantsList, numInformants):
        # Setting random informants
        self.informants = np.random.choice(informantsList, numInformants, replace=False)

    # Function to get the position
    def getPosition(self):
        return self.position
    
    # Function to get the fitness
    def getFitness(self):
        return self.fitness
    
    # Function to get the particle best position
    def getpBestPosition(self):
        return self.pbest_position
    
    # Function to get the particle best fitness
    def getpBestFitness(self):
        return self.pbest_fitness

    # Function to print the particle information
    def print(self):
        print("=====================================================")
        print("=           Particle Information                    =")
        print("=====================================================")
        print("Position        : ", self.position)
        print("Velocity        : ", self.velocity)
        print("Best Position   : ", self.pbest_position)
        print("Fitness         : ", self.fitness)
        print("Best Fitness    : ", self.pbest_fitness)
        print("Informants      : ", self.informants)
        print("=====================================================")
        return
    
