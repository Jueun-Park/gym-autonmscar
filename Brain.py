import time
import pygame

from pygame.locals import *

class Brain:
    def __init__(self, database):
        self.database = database

    def run(self):
        while True:
            # Implement Your Algorithm HERE!!
            ''' 
            1. How can i get a lidar data?
                data = self.database.lidar.data

            2. How can i move a car?
                self.database.control.up()
                self.database.control.down()
                self.database.control.right()
                self.database.control.left()

            3. How can i get a car status data?
                self.database.car.direction
                self.database.car.speed
            '''
            print(self.database.car.speed, self.database.car.direction)
            self.database.control.up()
            self.database.control.right()

            time.sleep(0.01)
            if self.database.stop:
                break