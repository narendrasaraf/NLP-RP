"""
game/main.py
------------
Main game loop for the 2D Adaptive Shooter.
Currently runs entirely locally with tracked stats mapped to the HUD.
"""

import pygame
import sys
import time
import random
import uuid

from game.entities import Player, Bullet, Enemy, SCREEN_WIDTH, SCREEN_HEIGHT
from game.api_client import GameAPIClient
from game.adaptation import calculate_new_speed

def main():
    # ── Pygame Init ──────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Adaptive 2D Shooter (Standalone Phase)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # ── Game State ───────────────────────────────────────────────────────────
    all_sprites = pygame.sprite.Group()
    bullets     = pygame.sprite.Group()
    enemies     = pygame.sprite.Group()

    player = Player()
    all_sprites.add(player)

    # Core Difficulty Variable 
    game_speed = 1.0  
    ai_state   = "Loading..."
    ai_cii     = 0.0

    # Create background ML thread connected to the new API port
    session_id = str(uuid.uuid4())[:8]
    api = GameAPIClient(session_id=session_id)

    # Trackers for Cognitive Math Engine
    kill_count  = 0
    death_count = 0
    miss_count  = 0
    score       = 0
    
    last_shot_time = time.time()
    reaction_times = []
    
    # Telemetry interval timer
    last_telemetry_time = time.time()
    TELEMETRY_INTERVAL  = 2.0  # Sends POST payload every 2 seconds
    
    # Spawn timers
    spawn_rate = 90  # frames between spawns (lower = harder)
    spawn_timer = 0  # accrues per frame

    # ── Main Loop ────────────────────────────────────────────────────────────
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Record reaction time (Time since last shot)
                    now = time.time()
                    reaction = (now - last_shot_time) * 1000 # convert to ms
                    if reaction < 5000: # Ignore breaks in playing
                        reaction_times.append(reaction)
                    last_shot_time = now
                    
                    # Spawn bullet
                    bullet = Bullet(player.rect.centerx, player.rect.top)
                    all_sprites.add(bullet)
                    bullets.add(bullet)

        # 2. Logic Update
        keys = pygame.key.get_pressed()
        player.update(keys)
        bullets.update()
        enemies.update()

        # Track missed bullets automatically based on 'missed' flag from entity
        for b in bullets:
            if b.missed:
                miss_count += 1
                b.missed = False # Prevent multi-counting before kill

        # 3. Dynamic Enemy Spawning
        spawn_timer += 1 
        if spawn_timer >= spawn_rate:
            spawn_timer = 0
            new_enemy = Enemy(difficulty_speed_multiplier=game_speed)
            all_sprites.add(new_enemy)
            enemies.add(new_enemy)

        # 4. Collision Detection
        
        # A. Bullets hit Enemies
        hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
        if hits:
            kill_count += len(hits)
            score += len(hits) * 10

        # B. Enemies hit Player
        player_hit = pygame.sprite.spritecollide(player, enemies, True)
        if player_hit:
            death_count += 1
            score -= 15
            player.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50)
            
            screen.fill((150, 0, 0))
            pygame.display.flip()
            time.sleep(0.05)

        # 5. Non-Blocking Telemetry Extraction
        now = time.time()
        if now - last_telemetry_time >= TELEMETRY_INTERVAL:
            last_telemetry_time = now
            
            avg_reaction = sum(reaction_times[-5:]) / 5 if len(reaction_times) >= 5 else 0.0
            
            # Simulate NLP features
            simulated_polarity = random.uniform(-1.0, 1.0)
            simulated_intensity = random.uniform(0.0, 1.0)
            
            # Build requested dictionary format
            telemetry_snapshot = {
                "telemetry": {
                    "kill_count": kill_count,
                    "death_count": death_count,
                    "miss_count": miss_count,
                    "reaction_time": round(avg_reaction, 2),
                    "score": score
                },
                "nlp": {
                    "polarity": round(simulated_polarity, 2),
                    "emotional_intensity": round(simulated_intensity, 2)
                }
            }
            
            # Dispatch payload seamlessly to backend thread (no game freezing!)
            api.send_telemetry(telemetry_snapshot)
            print(f"[DISPATCH] Sent raw telemetry for {session_id} to AI...")

        # 6. Check for AI Difficulty Updates
        ml_update = api.check_for_updates()
        if ml_update:
            # Re-scale game variables based on AI generic action/state
            new_speed = calculate_new_speed(game_speed, ml_update["state"])
            
            print(f"[{ml_update['action'].upper()}] CII: {ml_update['cii']} | State: {ml_update['state']}")
            print(f"   -> Adjusting Game Speed: {game_speed:.2f} -> {new_speed:.2f}")
            
            # Explicitly update BOTH core game physics tracking variables
            game_speed = new_speed
            
            # Faster game_speed = smaller spawn_rate (enemies spawn more frequently)
            # Default 1.0 speed = 90 frames. Max 3.0 speed = 30 frames.
            spawn_rate = max(30, int(90 / game_speed))
            
            ai_state   = ml_update['state']
            ai_cii     = ml_update['cii']

        # 7. Drawing & HUD
        screen.fill((20, 20, 30))  # Dark background
        all_sprites.draw(screen)

        avg_hud = sum(reaction_times[-5:]) / 5 if len(reaction_times) >= 5 else 0.0
        hud_lines = [
            f"Score: {score}  (Kills:{kill_count} D:{death_count} M:{miss_count})",
            "",
            f"CII Score: {ai_cii:+.3f}",
            f"Player State: {ai_state}",
            f"Difficulty Level: {game_speed:.2f}x"
        ]
        
        # Draw HUD block
        hud_bg = pygame.Surface((340, 160))
        hud_bg.set_alpha(150)
        screen.blit(hud_bg, (10, 10))
        
        for i, text in enumerate(hud_lines):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (20, 20 + (i * 25)))

        pygame.display.flip()
        clock.tick(60) # Lock to 60 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
