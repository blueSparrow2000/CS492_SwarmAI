from variables_n_utils import *

current_id = 0

class Fish:
    def __init__(self, x, y, size=1, id=None, color=None):
        self.x = x
        self.y = y
        self.size = size
        self.id = id if id is not None else self.generate_id()
        self.color = color if color is not None else (0, 255, 0)  # 기본 색상은 초록색
        self.alive = True  # 물고기가 살아있는지 여부를 나타내는 플래그

    def generate_id(self):
        global current_id
        id = current_id
        current_id += 1
        return id

    def move(self, dx, dy):
        if self.alive:
            self.x += dx
            self.y += dy
            self.x, self.y = bound_less_domain(self.x, self.y)

    def update_state(self, is_alive):
        self.alive = is_alive

    def detect_collision(self, other):
        if not self.alive:
            return False
        if self.x == other.x and self.y == other.y: # collision
            return True
