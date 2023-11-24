import numpy as np
from Particle import Particle

# Class for PSO Algorithm
class PSO:
    # Constructor
    def __init__(self, population, boundary, network, numInformants):
        self.population = population
        self.particles = [Particle() for i in range(population)]
        self.boundary = boundary
        self.network = network
        self.numInformants = numInformants
        self.gbest_fitness = float('inf')
        self.gbest_position = None
        self.gbest_accuracy = 0
    
    # Function to print the parameters
    def print(self):
        print("=====================================")
        print("PSO Parameters")
        print("=====================================")
        print("Population: ", self.population)
        print("Boundary: ", self.boundary)
        print("Number of Informants: ", self.numInformants)
        print("=====================================")
    

    # Function to initialize the population
    def initialize(self):
        # Getting the number of weights and biases
        weightBiasCount = self.network.weightBiasList().shape[0]
        # Looping through each particle
        for particle in self.particles:

            # Setting the boundary
            x = [[self.boundary[0], self.boundary[1]] for j in range(weightBiasCount)]
            particle.setBounds(x) 
            
            # Initializing the particle
            particle.initialize(weightBiasCount)

            # Set the informants
            informantsList = [j for j in self.particles if j != self.particles.index(particle)]
            particle.setInformants(informantsList, self.numInformants)
    
    # Function to train the network
    def train(self, x_train, y_train, loss_func, epochs, alpha, beta, gamma, delta):
        
        # Initializing the population
        self.initialize()
       
       # Looping through each epoch
        for epoch in range(epochs):

            # Looping through each particle
            for particle in self.particles:

                # Updating the network with the particle position
                self.network.update(particle.position)

                # Calculating the fitness
                accuracy, fitness = self.network.evaluate(x_train, y_train, loss_func)
                particle.fitness = fitness

                # Checking if the fitness is better than the best fitness
                if fitness < particle.pbest_fitness:
                    # Setting the best fitness
                    particle.pbest_fitness = fitness
                    # Setting the best position
                    particle.pbest_position = particle.position

                # Checking if the fitness is better than the global best fitness
                if fitness < self.gbest_fitness:
                    # Setting the global best fitness
                    self.gbest_fitness = fitness
                    # Setting the global best position
                    self.gbest_position = particle.position
                    self.gbest_accuracy = accuracy

                # Updating the velocity
                particle.updateVelocity(alpha, beta, gamma, delta, particle.informants[0].getpBestPosition(), self.gbest_position)

                # Updating the position
                particle.updatePosition()

                informantsList = [j for j in self.particles if j != particle]
                particle.setInformants(informantsList, self.numInformants)
            print (epoch + 1, '/', epochs, "gbest ===================>", self.gbest_fitness," , ", self.gbest_accuracy)
            # Updating the network with the global best position
            self.network.update(self.gbest_position)
        # Returning the global best fitness and position
        return fitness, accuracy


