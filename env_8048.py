import numpy as np
import pygame
import gymnasium as gym
import random
import time

# Automomous Electric Vehicle to find the passegnger, pickup and drop to the destination and maintian battery level

# 0.1% car battery discharge for taking any random action 
# 0 reward For taking any random action
# -1 reward if car position met obstacles position
# 5 points reward if car position met electric charging station position and reset battery capacity to 100%
# 10 points reward for pickup pessenger
# 20 points reward for droping pessenger to desitination 



class MyEnv(gym.Env):

    def __init__(self):
        super(MyEnv, self).__init__()

        self.grid_size = 10
        self.cell_size = 70
        self.channels = 3
        self.max_battery = 100  # Current Battery Status 100 % 
        self.observation_shape = (self.grid_size * self.cell_size, self.grid_size * self.cell_size, self.channels)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)

        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        pygame.display.set_caption("Autonomous Electrical Vehicle - sah8048@thi.de (Sanwal Hussain)")

        self.load_images()
        self.canvas = np.zeros(self.observation_shape, dtype=np.uint8)
        self.elements = []
        self.reset()

    # 1 .reset()
    # -------------    
    def reset(self):
        self.battery_left = self.max_battery
        self.reward = 0
        self.steps_taken = 0
        self.car_pos = self.random_position()

        while True:
            self.passenger_pos = self.random_position()
            if self.passenger_pos != self.car_pos:
                break

        while True:
            self.destination_pos = self.random_position()
            if self.destination_pos != self.car_pos and self.destination_pos != self.passenger_pos:
                break

        self.obstacles = []
        for _ in range(5):
            while True:
                obstacle_pos = self.random_position()
                if obstacle_pos != self.car_pos and obstacle_pos != self.passenger_pos and obstacle_pos != self.destination_pos:
                    self.obstacles.append(obstacle_pos)
                    break

        self.ev_charges = []
        num_ev_charges = 3
        for _ in range(num_ev_charges):
            while True:
                ev_charge_pos = self.random_position()
                if ev_charge_pos != self.car_pos and ev_charge_pos != self.passenger_pos and ev_charge_pos != self.destination_pos and ev_charge_pos not in self.obstacles:
                    self.ev_charges.append(ev_charge_pos)
                    break

        self.passenger_picked = False
        self.update_distance_to_goal()
        self.display_status()
        pygame.display.update()
        return np.zeros(self.observation_shape, dtype=np.uint8)
    
    # 2 .step()
    # -------------  
    def step(self, action):
        self.battery_left -= 0.1 # 0.1% car battery discharge for taking any random action
        reward = 0 # 0 reward For taking any random action
        done = False

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.car_pos = (self.car_pos[0] + moves[action][0], self.car_pos[1] + moves[action][1])
        self.car_pos = (max(0, min(self.grid_size - 1, self.car_pos[0])), max(0, min(self.grid_size - 1, self.car_pos[1])))

        self.steps_taken += 1  # Increment steps taken

        if self.car_pos in self.obstacles:
            reward = -1 # -1 point reward if car position met obstacles position
            done = False
        elif self.car_pos in self.ev_charges:
            self.battery_left = self.max_battery
            reward = 5 # 5 point reward if car position met fuel station position
            self.ev_charges.remove(self.car_pos)
        elif self.car_pos == self.passenger_pos and not self.passenger_picked:
            self.passenger_picked = True
            reward = 10 # 10 points reward for pickup pessenger
        elif self.car_pos == self.destination_pos and self.passenger_picked:
            reward = 20 # 20 points reward for droping pessenger to desitination 
            done = True

        self.reward += reward

        if self.battery_left == 0:
            done = True

        self.update_distance_to_goal()
        self.display_status()
        return np.zeros(self.observation_shape, dtype=np.uint8), reward, done, self.distance_to_goal
    
    # 3 .render()
    # -------------
    def render(self, mode="human"):
        if mode == "human":
            self.display_status()
            pygame.display.update()

    # 4 .close()
    # -------------        
    def close(self):
        time.sleep(2)
        pygame.quit()

    # 5 .random position()
    # -------------
    # Generate a random position for Vehicle, Pessenger, EV Charging station and Obstacles
    def random_position(self):
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
    
    # 6 .update distance()
    # -------------
    # Show the distance to goal from the car current position
    def update_distance_to_goal(self):
        self.distance_to_goal = np.linalg.norm(np.array(self.destination_pos) - np.array(self.car_pos))
    
    # 7 .display status()
    # -------------
    # Display the current status of the environment
    def display_status(self):
        self.screen.blit(self.map_image, (0, 0))
        self.screen.blit(self.car_icon, (self.car_pos[0] * self.cell_size, self.car_pos[1] * self.cell_size))
        
        if not self.passenger_picked:
            self.screen.blit(self.passenger_icon, (self.passenger_pos[0] * self.cell_size, self.passenger_pos[1] * self.cell_size))

        self.screen.blit(self.destination_icon, (self.destination_pos[0] * self.cell_size, self.destination_pos[1] * self.cell_size))

        for pos in self.obstacles:
            self.screen.blit(self.obstacle_icon, (pos[0] * self.cell_size, pos[1] * self.cell_size))

        for pos in self.ev_charges:
            self.screen.blit(self.ev_icon, (pos[0] * self.cell_size, pos[1] * self.cell_size))

        text = f'Battery: {self.battery_left:.2f}% | Reward: {self.reward:.2f} | Steps: {self.steps_taken} | Distance to Goal: {self.distance_to_goal:.2f}'

        font = pygame.font.SysFont(None, 24)
        img = font.render(text, True, (255, 255, 0))
        self.screen.blit(img, (10, 680))

        status_text = 'One person is waiting' if not self.passenger_picked else 'Person picked up'
        if self.passenger_picked and self.car_pos == self.destination_pos:
            status_text = 'Person dropped off'
        
        status_img = font.render(status_text, True, (255, 255, 0))
        self.screen.blit(status_img, (10, 650))

    # 8 .load images()
    # -------------
    # Load and scale images for the environment
    def load_images(self):
        self.map_image = pygame.image.load("media/ingolstadt-map.png")
        self.map_image = pygame.transform.scale(self.map_image, (self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        self.car_icon = pygame.image.load("media/electric-car.png")
        self.car_icon = pygame.transform.scale(self.car_icon, (self.cell_size, self.cell_size))
        self.passenger_icon = pygame.image.load("media/passenger.png")
        self.passenger_icon = pygame.transform.scale(self.passenger_icon, (self.cell_size, self.cell_size))
        self.destination_icon = pygame.image.load("media/destination.png")
        self.destination_icon = pygame.transform.scale(self.destination_icon, (self.cell_size, self.cell_size))
        self.obstacle_icon = pygame.image.load("media/obstacle.png")
        self.obstacle_icon = pygame.transform.scale(self.obstacle_icon, (self.cell_size, self.cell_size))
        self.ev_icon = pygame.image.load("media/ev-fuel.png")
        self.ev_icon = pygame.transform.scale(self.ev_icon, (self.cell_size, self.cell_size))

if __name__ == "__main__":
    env = MyEnv()
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)
        if done:
            break

    env.close()
