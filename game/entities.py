"""
game/entities.py
----------------
Core game classes for the 2D Adaptive Shooter.
Keeps the main loop clean by encapsulating update and rendering logic.
"""

import pygame
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Visuals
        self.image = pygame.Surface((40, 40))
        self.image.fill((0, 255, 150))  # Teal player
        
        # Physics
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        self.speed = 6

    def update(self, keys):
        # Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rect.x += self.speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.rect.y += self.speed
            
        # Bounds checking
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((8, 20))
        self.image.fill((255, 255, 0))  # Yellow laser
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = 12
        self.missed = False  # Flag to track if it leaves screen without hitting

    def update(self):
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.missed = True
            self.kill()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, difficulty_speed_multiplier: float):
        super().__init__()
        size = random.randint(25, 45)
        self.image = pygame.Surface((size, size))
        self.image.fill((255, 50, 50))  # Red enemy
        
        self.rect = self.image.get_rect(
            center=(random.randint(size, SCREEN_WIDTH - size), -50)
        )
        
        # Speed inherently scales with the game's difficulty variable
        base_speed = random.uniform(2.0, 4.0)
        self.speed = base_speed * difficulty_speed_multiplier

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()
