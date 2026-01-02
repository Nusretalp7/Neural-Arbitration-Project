# neural_arbitrator_pygame.py
import pygame
import numpy as np
import sys
from collections import deque

# ----------------------------
# Wong & Wang style attractor
# ----------------------------
class WongWangAttractor:
    def __init__(self, dt=0.002):
        # timestep for internal integration (s)
        self.dt = dt

        # parameters (tuned for interactive sim)
        self.tau_S = 0.1         # synaptic gating time constant
        self.a = 270.0
        self.b = 108.0
        self.d = 0.154
        self.J_11 = 0.2609
        self.J_22 = 0.2609
        self.J_12 = 0.0497
        self.J_21 = 0.0497
        self.I_0 = 0.3255
        self.gamma = 0.641

        # noise that enters synaptic gating update (scale)
        self.sigma_noise = 0.02

        # initial small asymmetry helps spontaneous decisions
        self.S1 = 0.1 + np.random.uniform(-0.005, 0.005)
        self.S2 = 0.1 + np.random.uniform(-0.005, 0.005)

        self.time = 0.0

    def H_function(self, x):
        # Rectified F-I curve (Wong & Wang style)
        y = self.a * x - self.b
        if y <= 0:
            return 0.0
        # safe denominator
        denom = 1.0 - np.exp(-self.d * y)
        if denom <= 1e-12:
            return y  # fallback
        return y / denom

    def step(self, I1, I2):
        # one internal Euler step (dt)
        # noise on gating variable (scaled by sqrt(dt))
        noise1 = self.sigma_noise * np.sqrt(self.dt) * np.random.randn()
        noise2 = self.sigma_noise * np.sqrt(self.dt) * np.random.randn()

        x1 = self.J_11 * self.S1 - self.J_12 * self.S2 + I1 + self.I_0
        x2 = self.J_22 * self.S2 - self.J_21 * self.S1 + I2 + self.I_0

        r1 = self.H_function(x1)
        r2 = self.H_function(x2)

        dS1 = (-self.S1 / self.tau_S + self.gamma * (1.0 - self.S1) * r1) * self.dt + noise1
        dS2 = (-self.S2 / self.tau_S + self.gamma * (1.0 - self.S2) * r2) * self.dt + noise2

        self.S1 = np.clip(self.S1 + dS1, 0.0, 1.0)
        self.S2 = np.clip(self.S2 + dS2, 0.0, 1.0)

        self.time += self.dt

        return r1, r2, self.S1, self.S2

    def reset(self):
        self.S1 = 0.1 + np.random.uniform(-0.005, 0.005)
        self.S2 = 0.1 + np.random.uniform(-0.005, 0.005)
        self.time = 0.0

# ----------------------------
# Human / BCI signal generator
# ----------------------------
class BCI_NoiseSource:
    def __init__(self, base_intent=1.0, switch_prob=0.01, input_sigma=0.25):
        # base_intent: +1 means bias to Right, -1 means Left
        self.base_intent = base_intent
        self.switch_prob = switch_prob
        self.input_sigma = input_sigma
        self.current_intent = base_intent

    def step(self):
        # occasional switching of underlying intent
        if np.random.rand() < self.switch_prob:
            self.current_intent *= -1.0
        # produce two noisy input currents I1 (Right), I2 (Left)
        # we map intent so that if current_intent = +1 => I_right slightly higher
        mu_right = 0.06 * (1.0 if self.current_intent > 0 else 0.0)
        mu_left  = 0.06 * (1.0 if self.current_intent < 0 else 0.0)
        noise_r = np.random.randn() * self.input_sigma
        noise_l = np.random.randn() * self.input_sigma
        I1 = mu_right + 0.01 * noise_r   # small direct drive into currents
        I2 = mu_left  + 0.01 * noise_l
        return I1, I2

# ----------------------------
# Pygame Simulation
# ----------------------------
def run_simulation():
    pygame.init()
    W, H = 1000, 480
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Neural Arbitrator — RAW vs WTA")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # instantiate model and source
    # We'll run the attractor with a small internal dt and multiple substeps per frame
    model = WongWangAttractor(dt=0.002)
    source = BCI_NoiseSource(base_intent=1.0, switch_prob=0.01, input_sigma=1.0)

    mode = "WTA"  # start with WTA; press 'r' to toggle RAW
    paused = False

    # cursor state
    x = W // 2
    vx = 0.0
    mass = 1.0      # inertia (increase = heavier / smoother)
    friction = 0.95  # damping per frame (0..1) - reduced damping for less resistance

    # params mapping
    raw_gain = 600.0   # map raw signal to velocity (increased for jitter visibility)
    wta_gain = 8.0     # map attractor output difference to velocity (much stronger for visible movement)

    # history for plotting small traces
    hist_len = 400
    r1_hist = deque([0.0]*hist_len, maxlen=hist_len)
    r2_hist = deque([0.0]*hist_len, maxlen=hist_len)
    s1_hist = deque([0.0]*hist_len, maxlen=hist_len)
    s2_hist = deque([0.0]*hist_len, maxlen=hist_len)
    
    # cursor trajectory history for visual smoothness comparison
    trajectory_len = 150
    trajectory = deque(maxlen=trajectory_len)
    
    # smoothness metrics
    vel_history = deque([0.0]*60, maxlen=60)  # 1 second at 60fps
    jitter_indicator = 0.0

    show_internals = True

    running = True
    frame = 0
    while running:
        dt_frame = clock.tick(60) / 1000.0  # seconds per frame (~0.0167)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    mode = "RAW" if mode == "WTA" else "WTA"
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    # change base intent (toggle left/right)
                    source.base_intent *= -1.0
                    source.current_intent = source.base_intent
                elif event.key == pygame.K_i:
                    # reset attractor
                    model.reset()
                elif event.key == pygame.K_UP:
                    # increase noise
                    source.input_sigma *= 1.2
                elif event.key == pygame.K_DOWN:
                    source.input_sigma /= 1.2
                elif event.key == pygame.K_h:
                    show_internals = not show_internals
                elif event.key == pygame.K_LEFT:
                    model.I_0 -= 0.001
                    print(f"I_0 decreased to {model.I_0:.4f}")
                elif event.key == pygame.K_RIGHT:
                    model.I_0 += 0.001
                    print(f"I_0 increased to {model.I_0:.4f}")

        if paused:
            continue

        # generate BCI noisy inputs (one sample per frame; can be fine-grained)
        I1_frame, I2_frame = source.step()

        # --- RAW MODE: directly map noisy difference to cursor acceleration ---
        if mode == "RAW":
            # direct instantaneous command (very jittery)
            cmd = (I1_frame - I2_frame)  # positive -> right, negative -> left
            # convert command to acceleration/velocity
            vx += raw_gain * cmd * dt_frame / mass
            # add tiny high-frequency jitter to emulate BCI tremor
            vx += np.random.randn() * 20.0 * (source.input_sigma * 0.2) * dt_frame

        # --- WTA MODE: feed inputs into attractor and use its output to command cursor ---
        else:
            # integrate model for several smaller substeps to keep stable dynamics
            substeps = max(1, int(0.02 / model.dt))  # ~10 substeps if dt=0.002
            r1, r2, S1, S2 = 0.0, 0.0, model.S1, model.S2
            for _ in range(substeps):
                r1, r2, S1, S2 = model.step(I1_frame, I2_frame)

            # compute command from firing rates difference (smoothed + bounded)
            diff = r1 - r2  # can be large; we normalize/scale carefully
            # Stronger gain: use tanh with larger coefficient for more responsive movement
            cmd = np.tanh(0.15 * diff)  # increased gain for decisive, visible action

            # map to velocity with inertia
            vx += wta_gain * cmd * dt_frame / mass

        # apply friction/damping
        vx *= friction

        # integrate position
        x += vx * dt_frame

        # clamp in screen
        if x < 20:
            x = 20
            vx = 0
        if x > W - 20:
            x = W - 20
            vx = 0

        # update histories
        r1_hist.append(r1)
        r2_hist.append(r2)
        s1_hist.append(S1)
        s2_hist.append(S2)
        
        # track trajectory and compute jitter
        trajectory.append(int(x))
        vel_history.append(abs(vx))
        
        # jitter metric: ONLY cursor smoothness (velocity variance)
        # In WTA mode, neural noise is internal - what matters is OUTPUT smoothness
        if len(vel_history) > 10:
            # compute velocity changes (acceleration jitter)
            vel_list = list(vel_history)
            vel_changes = [abs(vel_list[i] - vel_list[i-1]) for i in range(1, len(vel_list))]
            jitter_indicator = np.std(vel_changes) * 100  # acceleration jitter metric

        # --- Drawing ---
        screen.fill((18, 18, 20))

        # draw center line and targets
        pygame.draw.line(screen, (60,60,60), (W//2, 0), (W//2, H), 1)
        pygame.draw.rect(screen, (30, 80, 200), (W-60, H//2-40, 40, 80))  # right target
        pygame.draw.rect(screen, (200, 80, 30), (20, H//2-40, 40, 80))    # left target
        
        # draw cursor trajectory (trail showing smoothness)
        if len(trajectory) > 2:
            trail_pts = [(t, H//2) for t in trajectory]
            # fade alpha by drawing with varying line widths
            for i in range(1, len(trail_pts)):
                alpha_val = int(i / len(trail_pts) * 128)
                if mode == "RAW":
                    color = tuple(min(255, c + alpha_val//3) for c in (200, 80, 80))  # red-ish
                else:
                    color = tuple(min(255, c + alpha_val//3) for c in (80, 200, 80))  # green-ish
                width = max(1, int(1 + i / len(trail_pts) * 3))
                pygame.draw.line(screen, color, trail_pts[i-1], trail_pts[i], width)

        # draw cursor (plane)
        pygame.draw.circle(screen, (240,240,240), (int(x), H//2), 12)
        # velocity arrow
        vx_px = int(np.clip(vx*0.02, -40, 40))
        pygame.draw.line(screen, (200, 200, 0), (int(x), H//2), (int(x)+vx_px, H//2 - 30), 3)

        # draw small HUD
        hud_y = 10
        text = font.render(f"Mode: {mode}  |  RAW_GAIN={raw_gain:.0f}  WTA_GAIN={wta_gain:.2f}  Noiseσ={source.input_sigma:.2f}  I_0={model.I_0:.4f}", True, (220,220,220))
        screen.blit(text, (10, hud_y))

        text2 = font.render("Keys: R toggle RAW/WTA   SPACE pause   C toggle intent   I reset WTA   Left/Right adj I_0", True, (160,160,160))
        screen.blit(text2, (10, hud_y+22))

        # show cursor numeric info
        info = font.render(f"X={x:.1f}  Vx={vx:.2f}  Intent={'Right' if source.current_intent>0 else 'Left'}", True, (200,200,200))
        screen.blit(info, (10, hud_y+46))
        
        # show jitter/smoothness metric
        jitter_color = (255, 100, 100) if jitter_indicator > 15 else (100, 255, 100) if jitter_indicator < 5 else (255, 200, 100)
        jitter_text = font.render(f"Jitter/Smoothness: {jitter_indicator:.1f}  {'TREMOR!' if jitter_indicator > 15 else 'STABLE' if jitter_indicator < 5 else 'MODERATE'}", True, jitter_color)
        screen.blit(jitter_text, (10, hud_y+70))

        # draw internal variables plot area
        if show_internals:
            plot_w = 420
            plot_h = 110
            plot_x = 10
            plot_y = H - plot_h - 10

            # background rect
            pygame.draw.rect(screen, (28,28,32), (plot_x, plot_y, plot_w, plot_h))
            
            # mode indicator background (color coded)
            if mode == "RAW":
                mode_color = (60, 20, 20)  # dark red
                mode_label = "RAW MODE - Noisy Direct Drive"
            else:
                mode_color = (20, 60, 20)  # dark green
                mode_label = "WTA MODE - Neural Arbitration"
            pygame.draw.rect(screen, mode_color, (plot_x, plot_y - 25, plot_w, 22))
            mode_txt = font.render(mode_label, True, (220, 220, 100))
            screen.blit(mode_txt, (plot_x + 5, plot_y - 22))
            
            # r1/r2 traces
            pts_r1 = []
            pts_r2 = []
            N = len(r1_hist)
            for i, (a,b) in enumerate(zip(r1_hist, r2_hist)):
                px = plot_x + int(i * (plot_w / hist_len))
                # scale rates to fit
                y1 = plot_y + plot_h - int(np.clip(a / 50.0, 0.0, 1.0) * plot_h)
                y2 = plot_y + plot_h - int(np.clip(b / 50.0, 0.0, 1.0) * plot_h)
                pts_r1.append((px, y1))
                pts_r2.append((px, y2))
            if len(pts_r1) > 1:
                pygame.draw.lines(screen, (40,160,240), False, pts_r1, 2)
                pygame.draw.lines(screen, (240,120,80), False, pts_r2, 2)
            # S1/S2 bars
            bar_w = 16
            bx = plot_x + plot_w + 8
            by = plot_y
            pygame.draw.rect(screen, (40,40,40), (bx, by, 100, plot_h))
            # S1
            pygame.draw.rect(screen, (40,160,240), (bx+10, by + int((1.0 - s1_hist[-1]) * plot_h), bar_w, int(s1_hist[-1] * plot_h)))
            txt = font.render(f"S1={s1_hist[-1]:.3f}", True, (180,180,180))
            screen.blit(txt, (bx+10, by + plot_h + 2))
            # S2
            pygame.draw.rect(screen, (240,120,80), (bx+40, by + int((1.0 - s2_hist[-1]) * plot_h), bar_w, int(s2_hist[-1] * plot_h)))
            txt2 = font.render(f"S2={s2_hist[-1]:.3f}", True, (180,180,180))
            screen.blit(txt2, (bx+40, by + plot_h + 2))

        pygame.display.flip()

        frame += 1

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()
