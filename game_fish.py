import pygame
import random
from collections import namedtuple
import numpy as np
from variables_n_utils import *
from fish import Fish
import math

pygame.init()

font = pygame.font.SysFont('arial', 25)


### SHARK AI ###
class Shark():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_fish_idx = 0
        self.id = -1  # not a fish
        self.size = 3  # size threshold
        self.target_reset_cooltime = 10 # cannot change target for 5 attempts
        self.target_reset_count = 0

    ######################### shark behavior functions ###############################

    def get_close(self, target_fish):  # target fish를 인자로 받도록 수정
        dx = target_fish.x - self.x
        dy = target_fish.y - self.y

        # 맵의 경계를 지나가는 것이 더 가까운 경우에 대한 처리
        if (abs(dx) > WIDTH // 2):
            dx = dx - WIDTH if dx > 0 else dx + WIDTH
        if (abs(dy) > HEIGHT // 2):
            dy = dy - HEIGHT if dy > 0 else dy + HEIGHT

        # 절대값을 보고 x, y의 스칼라 값만 먼저 결정
        move_x = SHARK_MOVE_STEP if abs(dx) > abs(dy) else BLOCK_SIZE if dx != 0 else 0
        move_y = SHARK_MOVE_STEP if abs(dy) > abs(dx) else BLOCK_SIZE if dy != 0 else 0

        # 대각선으로 움직여야 하지만 shark_move_step을 넘어가면 안됨 -> x, y의 값을 1로 조정
        if move_x + move_y > SHARK_MOVE_STEP:
            move_x = BLOCK_SIZE
            move_y = BLOCK_SIZE

        # x, y의 부호를 결정
        self.x += move_x * (1 if dx > 0 else -1)
        self.y += move_y * (1 if dy > 0 else -1)

        self.x, self.y = bound_less_domain(self.x, self.y)

    def check_target_alive(self, n):
        return self.target_fish_idx < n

    def move(self, fish_list):
        # move towards the target twice
        if self.check_target_alive(len(fish_list)) and self.measure_fish_size(fish_list) < self.size:  # target fish alive
            target_fish = fish_list[self.target_fish_idx]
            self.get_close(target_fish)
        else:
            # pass
            # randomly select target
            if self.target_reset_count == self.target_reset_cooltime:
                self.reset_target(fish_list)
                self.target_reset_count = 0
            else:
                self.target_reset_count += 1

    def measure_fish_size(self, fish_list):
        # RULE 1
        target_fish = fish_list[self.target_fish_idx]
        x,y = target_fish.x, target_fish.y
        # RULE 1
        fish_nearby = 0
        for idx in range(len(fish_list)):
            if idx == self.target_fish_idx:
                continue
            cur_fish = fish_list[idx]
            
            # distance = math.sqrt((x - cur_fish.x) ** 2 + (y - cur_fish.y) ** 2)
            # bound를 넘어갔을 때 처리
            dx = min(abs(x - cur_fish.x), WIDTH - abs(x - cur_fish.x))
            dy = min(abs(y - cur_fish.y), HEIGHT - abs(y - cur_fish.y))
            distance = math.sqrt(dx ** 2 + dy ** 2)
            
            if distance <= RULE_1_RADIUS:
                fish_nearby += 1
            # if abs(x - cur_fish.x) <= RULE_1_RADIUS and abs(y - cur_fish.y) <= RULE_1_RADIUS:
            #     fish_nearby += 1

        # RULE 2
        # for idx in range(len(fish_list)):
        #     if idx == self.target_fish_idx:
        #         continue
        return fish_nearby + 1

    def reset_target(self, fish_list):
        self.target_fish_idx = random.randint(0, len(fish_list) - 1)
        return self.target_fish_idx
        # print('target reset to ', self.target_fish_idx)
        
    def set_target(self, fish_list, target_id):
        for i in range(len(fish_list)):
            if fish_list[i].id == target_id:
                self.target_fish_idx = i
                return True
        # if not found
        self.target_fish_idx = 0
        return False

            ### SHARK AI ###

############################# SHARK AI (if needed) #############################

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT): #w=640, h=640
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Swarm')
        self.clock = pygame.time.Clock()
        # To use shark agent, we need a dummy shark
        self.shark = Shark()

        self.init_fish_num = INITIAL_FISH_NUM
        self.reset()

    def reset(self):
        # init game state
        self.shark.x=0
        self.shark.y=0

        # Random position initialization
        self.fish_list = [Fish(
            BLOCK_SIZE * random.randint(self.w // (4 * BLOCK_SIZE), 3 * self.w // (4 * BLOCK_SIZE)),
            BLOCK_SIZE * random.randint(self.h // (4 * BLOCK_SIZE), 3 * self.h // (4 * BLOCK_SIZE)),
            id = i
        ) for i in range(self.init_fish_num)]

        self.score = 0
        # self.food = None
        # self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y, 0)
        if self.food in self.fish_list:
            self._place_food()

    def play_step(self, actions): # get actions from the agent
        self.frame_iteration += 1
        # self.score = self.frame_iteration
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move fish
        self._move(actions)  # update the head

        # 3. move and update shark / check fish eaten
        self.shark.move(self.fish_list)
        # 잡아먹히면 보상 -1
        reward = 0  
        if self.check_eaten():
            reward = -1  # 물고기가 잡아먹혔을 때 보상은 -1

        # 4. check if game over
        # reward = REWARD_EVERY_STEP # +1 for every step
        game_over = False
        #if self.is_collision()[0] or self.frame_iteration > 1000: # nothing happens for too long
        if self.frame_iteration > 500: # survives long enough
            game_over = True
            # reward = REWARD_GET_EATEN
            return reward, game_over, self.score

        # food
        # # place new food or just move
        # if self.food in self.fish_list:
        #     reward = REWARD_FOOD
        #     self.frame_iteration = 0
        #     self._place_food()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        self.score += reward
        # 6. return game over and score
        return reward, game_over, self.score
    
    def check_eaten(self):
    #     for fish in self.fish_list:
    #         if fish.detect_collision(self.shark):
    #             fish.update_state(False)
    #             self.fish_list.remove(fish)
    #             return True
    #     return False
    # 잡히면 랜덤 리스폰
        for fish in self.fish_list:
            if fish.detect_collision(self.shark):
                fish.x = BLOCK_SIZE * random.randint(self.w // (4 * BLOCK_SIZE), 3 * self.w // (4 * BLOCK_SIZE))
                fish.y = BLOCK_SIZE * random.randint(self.h // (4 * BLOCK_SIZE), 3 * self.h // (4 * BLOCK_SIZE))
                fish.update_state(True)
                self.shark.reset_target(self.fish_list)
                return True
        return False

    def collision_fish(self, x, y):  # check collision among fish => no op
        for fish in self.fish_list:
            if fish.x == x and fish.y == y:  # collision
                return True
        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        # fishes
        for fish in self.fish_list:
            color = RED if fish.id == self.shark.target_fish_idx else GREEN1
            pygame.draw.rect(self.display, color, pygame.Rect(fish.x, fish.y, BLOCK_SIZE, BLOCK_SIZE))
            
        # food
        # pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # shark
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.shark.x, self.shark.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.shark.x+4, self.shark.y+4, 12, 12))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, actions):
        for fish_idx in range(len(self.fish_list)):
            current_fish = self.fish_list[fish_idx]
            x = current_fish.x
            y = current_fish.y
            # left down right up
            if np.array_equal(actions[fish_idx], [0,0,1,0]): # RIGHT
                x += BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [1,0,0,0]): # LEFT
                x -= BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [0,1,0,0]): # DOWN
                y += BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [0,0,0,1]): # UP
                y -= BLOCK_SIZE

            if self.collision_fish(x, y):  # new location collides with other fish
                continue

            x_new, y_new= bound_less_domain(x,y)
            current_fish.move(x_new - current_fish.x, y_new - current_fish.y)
