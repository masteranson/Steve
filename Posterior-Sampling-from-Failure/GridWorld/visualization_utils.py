from collections import defaultdict
import pygame
import math
import os

try:
    from .ValueIteration import ValueIteration
except ImportError:
    from ValueIteration import ValueIteration

# How many pixels to pull the arrow away from each cell center.
_ARROW_INSET_FRAC = 0.15   # 15 percent of half-cell.
# Sequential palette for successive policies (extend / tweak as you like)
# Twelve visually distinct RGB colors for successive policies
_POLICY_COLORS = [
    (  0,   0,   0),   # 1  black       - initial policy
    ( 40, 120, 240),   # 2  blue
    (230, 150,  35),   # 3  orange
    (160,  45, 180),   # 4  purple
    ( 30, 180,  75),   # 5  green
    (200,  40,  55),   # 6  red
    ( 80, 180, 210),   # 7  cyan
    (230, 230,  60),   # 8  yellow
    (110,  60,   0),   # 9  brown
    (255, 105, 180),   # 10 pink
    ( 80,  80,  80),   # 11 gray
    (  0, 150, 130),   # 12 teal
]
# small pixel offsets to apply for successive policies (cycle if > len)
# (dx_scale, dy_scale) are *multipliers* of a base OFFSET_PX
_POLICY_OFFSETS = [
    ( 0.0,  0.0),   # 1st policy: no shift
    (-1.0,  0.0),   # 2nd: 1 px left of perpendicular
    (+1.0,  0.0),   # 3rd: 1 px right of perpendicular
    ( 0.0, -1.0),   # 4th: 1 px up of perpendicular
    ( 0.0, +1.0),   # 5th: 1 px down of perpendicular
    (-2.0,  0.0),   # 6th: 2 px left
    (+2.0,  0.0),   # 7th: 2 px right
    ( 0.0, -2.0),   # 8th: 2 px up
    ( 0.0, +2.0),   # 9th: 2 px down
    (-3.0,  0.0),   # 10th
    (+3.0,  0.0),   # 11th
    ( 0.0, -3.0),   # 12th
]
_OFFSET_PX = 2                    # base pixels for one unit of shift

# --- Color mapping from TD error to RGB ---
def val_to_color(val, bad_col=(249, 193, 169)):
    val = max(min(val, 0.0), -1.0)  # clip to [-1, 0]
    # return tuple([255 * (1 + val) + (col * -val) for col in bad_col])
    return bad_col

# --- Draw the GridWorld + agent path + td errors ---
def draw_state(screen,
               grid_mdp,
               state,
               policy=None,
               action_char_dict={},
               show_value=True,
               agent=None,
               draw_statics=False,
               agent_shape=None,
               trajectory=None,
               switches=None,
               early_stop_idx=None,
               candidate_goals=None,
               policy_candidates=None):
    """
    Extended version of draw_state with TD error overlays.

    Inputs:
        trajectory: list of ((x, y), td_error)
        switches: list of trajectory indices where recovery triggered
        early_stop_idx: int index of early stop (optional)
    """
    sx, sy = state[:2]
    val_text_dict = defaultdict(lambda : defaultdict(float))
    if show_value:
        if agent is not None:
            for s in agent.q_func.keys():
                val_text_dict[s.x][s.y] = agent.get_value(s)
        else:
            vi = ValueIteration(grid_mdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.x][s.y] = vi.get_value(s)

    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        vi = ValueIteration(grid_mdp)
        vi.run_vi()
        for s in vi.get_states():
            policy_dict[s.x][s.y] = policy(s)

    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0)

    # Keep the most negative TD error observed in every visited cell.
    cell_min_td = {}
    visited_cells = set()
    if trajectory:
        for (tx, ty, *_), td in trajectory:
            key = (tx, ty)
            visited_cells.add(key)
            cell_min_td[key] = min(td, cell_min_td.get(key, 0.0))

    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = { (g[0], g[1]) for g in grid_mdp.get_goal_locs() }
    if trajectory:
        stop_coords = trajectory[-1][0][:2]
    else:
        stop_coords = None
    goal_reached  = (stop_coords in goal_locs) if stop_coords else False
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    if draw_statics:
        for i in range(grid_mdp.width):
            for j in range(grid_mdp.height):
                cell_x, cell_y = i, j
                top_left = width_buffer + cell_width*cell_y, height_buffer + cell_height*cell_x
                
                if candidate_goals and (cell_x, cell_y) in candidate_goals:
                    pale_green = (20, 30, 20)
                    pygame.draw.rect(screen, pale_green, top_left + (cell_width, cell_height), 0)

                # Color visited states with the worst (most-negative) TD seen there.
                if (cell_x, cell_y) in visited_cells:
                    td_val = cell_min_td.get((cell_x, cell_y), 0.0)
                    if td_val < 0:
                        color = val_to_color(td_val)
                    else:
                        color = (255, 255, 255)
                    pygame.draw.rect(screen, color, top_left + (cell_width, cell_height), 0)

                # Cell border
                pygame.draw.rect(screen, (46, 49, 49), top_left + (cell_width, cell_height), 3)

                if policy and not grid_mdp.is_wall(cell_x, cell_y):
                    a = policy_dict[cell_x][cell_y]
                    text_a = action_char_dict.get(a, a)
                    text_center = int(top_left[0] + cell_width/2.0 - 10), int(top_left[1] + cell_height/3.0)
                    text_rendered = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered, text_center)

                if show_value and not grid_mdp.is_wall(cell_x, cell_y):
                    val = val_text_dict[cell_x][cell_y]
                    text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_pos = int(top_left[0] + cell_width/2.0 - 10), int(top_left[1] + cell_height/3.0)
                    screen.blit(text, text_pos)

                if grid_mdp.is_wall(cell_x, cell_y):
                    padded = top_left[0] + 5, top_left[1] + 5
                    pygame.draw.rect(screen, (94, 99, 99), padded + (cell_width-10, cell_height-10), 0)

                if (cell_x, cell_y) in goal_locs:
                    center = int(top_left[0] + cell_width/2.0), int(top_left[1] + cell_height/2.0)
                    pygame.draw.rect(screen, (154, 255, 157), top_left + (cell_width, cell_height), 0)

                if show_value and (cell_x, cell_y) == (sx, sy) and agent_shape is None:
                    center = int(top_left[0] + cell_width/2.0), int(top_left[1] + cell_height/2.0)
                    agent_shape = draw_agent(center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)
        
        # Draw one color-coded circle in each policy candidate goal cell.
        if policy_candidates:
            # Palette indices: 1 (blue) for first policy, then 2, 3, etc.
            # if the run actually reached the goal, overwrite the *last*
            # The initial policy uses black (index 0).
            start_idx   = 1
            palette     = _POLICY_COLORS[start_idx:start_idx + len(policy_candidates)]
            if goal_reached and palette:
                palette[-1] = _POLICY_COLORS[0]        # black

            for (gx, gy), colour in zip(policy_candidates, palette):
                if colour == _POLICY_COLORS[0]:
                    continue                           # skip black dot
                top_left = (
                    width_buffer + cell_width  * gy,
                    height_buffer + cell_height * gx,
                )
                centre = (
                    int(top_left[0] + cell_width  / 2),
                    int(top_left[1] + cell_height / 2),
                )
                radius = int(0.25 * min(cell_width, cell_height))
                pygame.draw.circle(screen, colour, centre, radius)


        # Draw arrows; each policy gets a perpendicular offset to avoid overlap
                # Draw arrows; each policy gets a perpendicular offset to avoid overlap
        if trajectory and len(trajectory) > 1:
            inset         = _ARROW_INSET_FRAC * min(cell_width, cell_height)
            switches_set  = set(switches or [])
            incoming_segs = {s - 1 for s in switches_set if s > 0}
            segments      = list(zip(trajectory[:-1], trajectory[1:]))
            num_segments  = len(segments)

            # Build a color-index list for every segment (chronological).
            seg_color_idx = []
            policy_idx    = 1                         # blue for the first policy
            for k in range(num_segments):
                if k in switches_set:                 # recovery triggers new policy
                    policy_idx += 1
                seg_color_idx.append(policy_idx)

            # If the goal was reached, render the final policy segments in black
            if goal_reached and seg_color_idx:
                last_policy = seg_color_idx[-1]
                seg_color_idx = [
                    0 if c == last_policy else c for c in seg_color_idx
                ]

            # Draw from oldest to newest so later policies sit on top
            for seg_idx in reversed(range(num_segments)):
                # Safe, wrapped palette and offset lookup.
                pal_i  = seg_color_idx[seg_idx] % len(_POLICY_COLORS)
                off_i  = seg_color_idx[seg_idx] % len(_POLICY_OFFSETS)
                colour = _POLICY_COLORS[pal_i]
                ox, oy = _POLICY_OFFSETS[off_i]
                #

                (x1, y1, *_), _ = segments[seg_idx][0]
                (x2, y2, *_), _ = segments[seg_idx][1]
                if (x1, y1) == (x2, y2):
                    continue

                # Cell-center coordinates.
                c1 = (
                    width_buffer + cell_width  * y1 + cell_width  / 2,
                    height_buffer + cell_height * x1 + cell_height / 2,
                )
                c2 = (
                    width_buffer + cell_width  * y2 + cell_width  / 2,
                    height_buffer + cell_height * x2 + cell_height / 2,
                )

                # direction + perpendicular
                dx, dy = c2[0] - c1[0], c2[1] - c1[1]
                dist   = math.hypot(dx, dy) or 1.0
                ux, uy = dx / dist, dy / dist
                px, py = -uy, ux

                # perpendicular offset for this policy index
                off_x  = (px * ox + ux * oy) * _OFFSET_PX
                off_y  = (py * ox + uy * oy) * _OFFSET_PX

                # inset + offset start/end points
                start = (c1[0] + ux * inset + off_x, c1[1] + uy * inset + off_y)
                end   = (c2[0] - ux * inset + off_x, c2[1] - uy * inset + off_y)

                is_last   = seg_idx == num_segments - 1
                draw_head = (seg_idx not in incoming_segs) and not (goal_reached and is_last)
                _draw_arrow(
                    screen,
                    start,
                    end,
                    color=colour,
                    width=2,
                    head_len=6,
                    draw_head=draw_head,
                )


    if agent_shape is not None:
        # Clear and draw new agent
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left = (
            width_buffer + cell_width * (sx - 1),
            height_buffer + cell_height * (grid_mdp.height - sy),
        )
        center = (
            int(top_left[0] + cell_width / 2.0),
            int(top_left[1] + cell_height / 2.0),
        )
        agent_shape = draw_agent(center, screen, base_size=min(cell_width, cell_height) / 2.5 - 8)

    return agent_shape

def draw_agent(center_point, screen, base_size=20):
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)
    return pygame.draw.polygon(screen, tri_color, tri)

# Helper: draw a line and optional arrow head.
def _draw_arrow(screen, start, end, color=(0, 0, 0), width=2, head_len=6, draw_head=True):
    """
    Draws an arrow from `start` to `end`.
    `start`, `end` are pixel (x,y) tuples.
    """
    pygame.draw.line(screen, color, start, end, width)
    if draw_head:
        angle = math.atan2(start[1] - end[1], start[0] - end[0])
        for sign in (+1, -1):
            theta = angle + sign * math.pi / 6      # +/- 30 degrees
            hx = end[0] + head_len * math.cos(theta)
            hy = end[1] + head_len * math.sin(theta)
            pygame.draw.line(screen, color, end, (hx, hy), width)

def save_episode_visualization(
    grid_mdp,
    trajectory,
    switches=None,
    early_stop_idx=None,
    save_path="trajectory.png",
    screen_size=(720, 720),
    title=None,
    show_legend=True,
    candidate_goals=None,
    policy_candidates=None
):

    """
    Uses draw_state to render trajectory + td errors via pygame and saves image.
    """
    from visualization_utils import draw_state  # avoid circular import

    pygame.init()
    screen = pygame.Surface(screen_size)
    clock = pygame.time.Clock()
    
    # draw_state draws full path with coloring
    draw_state(
        screen,
        grid_mdp,
        state=trajectory[-1][0],
        show_value=False,
        draw_statics=True,
        trajectory=trajectory,
        switches=switches,
        early_stop_idx=early_stop_idx,
        candidate_goals=candidate_goals,
        policy_candidates=policy_candidates
    )

        # Draw optional title
    if title:
        max_width = screen_size[0] - 40
        font_size = 28
        while font_size > 10:
            title_font = pygame.font.SysFont("Arial", font_size, bold=True)
            title_surf = title_font.render(title, True, (255, 255, 255))
            if title_surf.get_width() <= max_width:
                break
            font_size -= 1
        screen.blit(title_surf, (screen_size[0] // 2 - title_surf.get_width() // 2, 10))


    # Draw optional legend
    if show_legend:
        legend_font = pygame.font.SysFont("Arial", 16)
        legend_items = [
            ("TD Error", (249, 193, 169), "rect"),
            ("True Goal", (154, 255, 157), "rect"),
            ("Candidate Goals", (20, 30, 20), "rect"),
            ("Hypothesized Goals (Color-Coded)", (40, 120, 240), "circle"),
            ("Wall", (94, 99, 99), "rect"),
        ]

        # Dimensions and position
        legend_x = 20
        legend_y = screen_size[1] - 140
        legend_width = 300
        legend_height = 20 * len(legend_items) + 20
        padding = 10

        # Background box
        pygame.draw.rect(screen, (245, 245, 245), (legend_x, legend_y, legend_width, legend_height))
        pygame.draw.rect(screen, (100, 100, 100), (legend_x, legend_y, legend_width, legend_height), 2)

        for i, (label, color, shape) in enumerate(legend_items):
            cy = legend_y + padding + i * 20
            if shape == "rect":
                pygame.draw.rect(screen, color, (legend_x + 10, cy, 15, 15))
            elif shape == "circle":
                center = (legend_x + 17, cy + 8)  # center of 15x15 box
                pygame.draw.circle(screen, color, center, 7)
            text = legend_font.render(label, True, (0, 0, 0))
            screen.blit(text, (legend_x + 30, cy))

    pygame.image.save(screen, save_path)
