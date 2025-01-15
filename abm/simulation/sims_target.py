import contextlib
with contextlib.redirect_stdout(None): # blocks pygame initialization messages
    import pygame

import numpy as np
import sys
import time

from abm import colors
from abm.sprites import supcalc
from abm.sprites.agent import Agent
from abm.sprites.resource import Resource
from abm.sprites.wall import Wall
from abm.sprites.landmark import Landmark
from abm.monitoring import tracking, plot_funcs
# from abm.monitoring.screen_recorder import ScreenRecorder
# from abm.helpers import timer

class Simulation:
    # @timer
    def __init__(self, env_size, window_pad,
                 N, T, with_visualization, framerate, print_enabled, plot_trajectory, log_zarr_file, save_ext,
                 agent_radius, max_vel, vis_field_res, vision_range, agent_fov, show_vision_range, agent_consumption, 
                 N_res, patch_radius, res_pos, res_units, res_quality, regenerate_patches, 
                 NN, other_input, vis_transform, percep_angle_noise_std, percep_dist_noise_std, action_noise_std,
                 boundary_scale, sim_type
                 ):
        """
        Initializing the main simulation instance
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param window_pad: padding of the environment in simulation window in pixels
        :param N: number of agents
        :param T: simulation time
        :param with_visualization: turns visualization on or off. For large batch autmatic simulation should be off so
            that we can use a higher/maximal framerate
        :param framerate: framerate of simulation
        :param print_enabled:
        :param plot_trajectory:
        :param log_zarr_file:
        :param save_ext:
        :param agent_radius: radius of the agents
        :param max_vel:
        :param vis_field_res: projection field (visual + proximity) resolution in pixels
        :param vision_range: range (in px) of agents' vision
        :param agent_fov (float): the field of view of the agent as percentage. e.g. if 0.5, the the field of view is
                                between -pi/2 and pi/2
        :param show_vision_range: bool to switch visualization of visual range for agents. If true the limit of far
                                and near field visual field will be drawn around the agents
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param N_res: number of resource patches in the environment
        :param patch_radius: radius of resource patches
        :param min_res_perpatch: minimum resource unit per patch
        :param max_res_perpatch: maximum resource units per patch
        :param min_res_quality: minimum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        : param max_res_quality: maximum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param NN:
        """
        # Arena parameters
        self.WIDTH, self.HEIGHT = env_size
        self.window_pad = window_pad
        self.coll_boundary_thickness = agent_radius

        self.x_min, self.x_max = 0, self.WIDTH
        self.y_min, self.y_max = 0, self.HEIGHT
        self.boundary_info_coll = (agent_radius*2, self.WIDTH - agent_radius*2, 
                                   agent_radius*2, self.HEIGHT - agent_radius*2)

        self.boundary_endpts = [
            np.array([ -boundary_scale, -boundary_scale ]),
            np.array([ self.WIDTH+boundary_scale, -boundary_scale ]),
            np.array([ -boundary_scale, self.HEIGHT+boundary_scale ]),
            np.array([ self.WIDTH+boundary_scale, self.HEIGHT+boundary_scale ])
        ]
        self.boundary_endpts_wp = [endpt + self.window_pad for endpt in self.boundary_endpts]

        # Simulation parameters
        self.N = N
        self.T = T
        self.t = 0
        self.with_visualization = with_visualization
        if self.with_visualization:
            self.framerate_orig = framerate
        else:
            # this is more than what is possible with pygame so it will use the maximal framerate
            self.framerate_orig = 2000
        self.framerate = self.framerate_orig # distinguished for varying in-game framerate
        self.is_paused = False
        self.print_enabled = print_enabled
        self.plot_trajectory = plot_trajectory
        self.sim_type = sim_type

        # Tracking parameters
        self.log_zarr_file = log_zarr_file

        if not self.log_zarr_file: # set up agent/resource data logging
            self.data_agent = np.zeros( (self.N, self.T, 4) ) # (pos_x, pos_y, mode, coll_res)
            self.data_res = []

        self.elapsed_time = 0
        # self.fitnesses = []
        self.save_ext = save_ext

        # Agent parameters
        self.respawn_counter = 0
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.vis_field_res = vis_field_res
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.show_vision_range = show_vision_range
        self.agent_consumption = agent_consumption

        # Boundary-Ray Collision parameters
        phis = np.linspace(-agent_fov*np.pi, agent_fov*np.pi, vis_field_res)
        self.phi_angle_diff = phis[1] - phis[0]
        self.fwd_traj = np.array([[0,0],[0,0]])
        self.last_moves = []
        self.ellipse_counter = 0
        self.single_ray_colls = []
        self.dual_ray_colls = []
        self.colls_no_ellipse = []

        # Resource parameters
        self.N_res = N_res
        self.res_radius = patch_radius
        self.res_pos = res_pos
        self.min_res_units, self.max_res_units = res_units
        self.min_res_quality, self.max_res_quality = res_quality
        # fix units/quality to single values if not ranges
        if self.max_res_units <= self.min_res_units:
            self.max_res_units = self.min_res_units + 1 # randint is exclusive
        if self.max_res_quality < self.min_res_quality:
            self.max_res_quality = self.min_res_quality # uniform is inclusive
        self.regenerate_resources = regenerate_patches

        # Neural Network parameters
        self.model = NN

        if N == 1:  self.num_class_elements = 4 # single-agent --> perception of 4 walls
        else:       self.num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
        # self.num_class_elements = 4

        self.other_input = other_input
        self.max_dist = np.hypot(self.WIDTH, self.HEIGHT)
        self.min_dist = agent_radius
        self.vis_transform = vis_transform
        self.percep_angle_noise_std = percep_angle_noise_std*2*np.pi # noise std as percentage * range
        self.percep_dist_noise_std = percep_dist_noise_std
        self.action_noise_std = action_noise_std*2

        # Initializing pygame
        if self.with_visualization:
            pygame.init()
            self.screen = pygame.display.set_mode([self.WIDTH + self.window_pad*2, self.HEIGHT + self.window_pad*2])
            self.font = pygame.font.Font(None, int(self.window_pad/2))
            # self.recorder = ScreenRecorder(self.WIDTH + self.window_pad*2, self.HEIGHT + self.window_pad*2, framerate, out_file='sim.mp4')
        else:
            pygame.display.init()
            pygame.display.set_mode([1,1])

        # pygame related class attributes
        self.walls = pygame.sprite.Group()
        self.objs = pygame.sprite.Group()
        self.agents = pygame.sprite.Group()
        self.resources = pygame.sprite.Group()
        self.clock = pygame.time.Clock() # todo: look into this more in detail so we can control dt

### -------------------------- DRAWING FUNCTIONS -------------------------- ###

    def draw_walls(self):
        """Drawing walls on the arena according to initialization"""
        TL,TR,BL,BR = self.boundary_endpts_wp
        pygame.draw.line(self.screen, colors.BLACK, TL, TR)
        pygame.draw.line(self.screen, colors.BLACK, TR, BR)
        pygame.draw.line(self.screen, colors.BLACK, BR, BL)
        pygame.draw.line(self.screen, colors.BLACK, BL, TL)
    
    def draw_objs(self):
        for obj in self.objs:
            pygame.draw.circle(self.screen, obj.color, obj.position + self.window_pad, obj.radius)

    def draw_status(self):
        """Showing framerate, sim time and pause status on simulation windows"""
        status = [
            # f"FPS: {self.framerate}  |  t = {self.t}/{self.T}",
            f"t = {self.t}/{self.T}",
        ]
        if self.is_paused:
            status.append("-Paused-")
        for i, stat_i in enumerate(status):
            text = self.font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (self.window_pad, self.window_pad - self.agent_radii))

    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information"""
        font = pygame.font.Font(None, font_size)
        for agent in self.agents:
            status = [ 
                # f'ID: {agent.id}',
                # f'res: {agent.collected_r}',
                f'ori: {int(agent.orientation*180/np.pi)} deg',
                f'NNout: {agent.action:.2f}',
                f'turn: {agent.action*180/np.pi:.2f} deg',
                f'vel: {agent.velocity:.2f} / {self.max_vel}',
            ]
            for i, stat_i in enumerate(status):
                text = font.render(stat_i, True, colors.BLACK)
                self.screen.blit(text, (agent.position[0] + 8*agent.radius,
                                        agent.position[1] - 1*agent.radius + i * (font_size + spacing)))

    def draw_visual_fields(self):
        """Visualizing range of vision as opaque circles around the agents""" 
        vis_proj_distance = 30
        vis_project_IDbubble_size = 4
        
        for agent in self.agents:
            start_pos = agent.pt_eye + self.window_pad
            # Show visual range as circle if non-limiting FOV
            if self.agent_fov == 1:
                pygame.draw.circle(self.screen, colors.GREY, agent.pt_eye + self.window_pad, vis_proj_distance, width=1)
            else: # self.agent_fov < 1 --> show limits of FOV as radial lines with length of visual range
                angles = (agent.orientation + agent.phis[0], 
                          agent.orientation + agent.phis[-1])
                for angle in angles: ### draws lines that don't quite meet borders
                    end_pos = (start_pos[0] + np.cos(angle) * vis_proj_distance,
                               start_pos[1] - np.sin(angle) * vis_proj_distance)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

            # draw projections as gray lines, either ending at walls (if dist_field is calculated) or extending beyond
            if self.vis_transform:

                for phi, vis_name, dist, dist_input in zip(agent.phis, agent.vis_field, agent.dist_field, agent.dist_input):

                    # draw lines to walls + bubble at wall end
                    end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * dist,
                                start_pos[1] - np.sin(agent.orientation - phi) * dist)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)
                    pygame.draw.circle(self.screen, colors.BLACK, end_pos, 2)

                    # draw bubbles reflecting perceived identities (wall/agents) with radius proportional to dist_input
                    if vis_name == 'wall_north': # --> red
                        pygame.draw.circle(
                            self.screen, colors.TOMATO, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_south': # --> green
                        pygame.draw.circle(
                            self.screen, colors.LIME, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_east': # --> blue
                        pygame.draw.circle(
                            self.screen, colors.CORN, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_west': # --> yellow
                        pygame.draw.circle(
                            self.screen, colors.GOLD, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    # elif vis_name == 'agent_exploit':
                    #     pygame.draw.circle(
                    #         self.screen, colors.VIOLET, 
                    #         (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                    #         start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                    #         radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name.startswith('obj'):
                        pygame.draw.circle(
                            self.screen, colors.BLACK, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)

            else:
                for phi, vis_name in zip(agent.phis, agent.vis_field):

                    # draw lines to walls
                    end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * 1500,
                                start_pos[1] - np.sin(agent.orientation - phi) * 1500)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

                    # draw bubbles reflecting perceived identities (wall/agents)
                    if vis_name == 'wall_north': # --> red
                        pygame.draw.circle(
                            self.screen, colors.TOMATO, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_south': # --> green
                        pygame.draw.circle(
                            self.screen, colors.LIME, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_east': # --> blue
                        pygame.draw.circle(
                            self.screen, colors.CORN, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_west': # --> yellow
                        pygame.draw.circle(
                            self.screen, colors.GOLD, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    # elif vis_name == 'agent_exploit':
                    #     pygame.draw.circle(
                    #         self.screen, colors.VIOLET, 
                    #         (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                    #         start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                    #         radius = vis_project_IDbubble_size)
                    elif vis_name.startswith('obj'):
                        pygame.draw.circle(
                            self.screen, colors.BLACK, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)

            # # draw line to patch
            # end_pos = np.array(self.res_pos)
            # length = 100

            # # line from agent to corner
            # end_pos = np.array([0,0])
            # len = 500
            # # res_pos[1] = self.y_max - res_pos[1]
            # disp_from_patch = end_pos - agent.position
            # angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
            # angle_diff = angle_to_patch - agent.orientation
            # angle_diff = (angle_diff - np.pi) % (2*np.pi) - np.pi
            # # print(angle_to_patch, angle_diff)
            # end_pos = (start_pos[0] + np.cos(angle_to_patch)*len,
            #            start_pos[1] - np.sin(angle_to_patch)*len)
            # pygame.draw.line(self.screen, colors.BLACK, start_pos, end_pos, 1)

            # # line from agent to corner
            # end_pos = np.array([0,1000])
            # len = 500
            # # res_pos[1] = self.y_max - res_pos[1]
            # disp_from_patch = end_pos - agent.position
            # angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
            # angle_diff = angle_to_patch - agent.orientation
            # angle_diff = (angle_diff - np.pi) % (2*np.pi) - np.pi
            # # print(angle_to_patch, angle_diff)
            # end_pos = (start_pos[0] + np.cos(angle_to_patch)*len,
            #            start_pos[1] - np.sin(angle_to_patch)*len)
            # pygame.draw.line(self.screen, colors.BLACK, start_pos, end_pos, 1)

            # agent traveled traj
            for i in range(self.N):
                for t_step in range(1,self.t):

                    start = self.data_agent[i, t_step-1, :2]
                    end = self.data_agent[i, t_step, :2]
                    start = np.array([start[0], self.y_max - start[1]])
                    end = np.array([end[0], self.y_max - end[1]])

                    pygame.draw.line(self.screen, colors.BLACK, start + self.window_pad, end + self.window_pad, 1)

    # @timer
    def draw_frame(self):
        """Drawing environment, agents and every other visualization in each timestep"""
        pygame.display.flip()
        self.screen.fill(colors.WHITE)
        # pygame.draw.circle(self.screen, colors.BLACK, (700,425), 5)
        self.walls.draw(self.screen)
        self.resources.draw(self.screen)
        self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_objs()
        self.draw_status()
        # self.draw_agent_stats()

        # vision range + projection field
        if self.show_vision_range: 
            self.draw_visual_fields()
    
### -------------------------- ENV FUNCTIONS -------------------------- ###
    
    def create_walls(self):

        walls = [
            ('wall_north', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_min ])),
            ('wall_south', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_max - self.coll_boundary_thickness ])),
            ('wall_east', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_max - self.coll_boundary_thickness, self.y_min ])),
            ('wall_west', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_min, self.y_min ]))
        ]

        for id, size, position in walls:
            wall = Wall(
                id=id,
                size=size,
                position=position,
                window_pad=self.window_pad
            )
            self.walls.add(wall)
    
    def create_objs(self):

        if self.sim_type == 'walls, 2x pinball':

            radius = 20
            color = colors.BLACK

            # obj_info = [
            #     np.array([ 200, 1000-550 ]),
            #     np.array([ 250, 1000-500 ]),
            #     np.array([ 300, 1000-450 ]),
            #     np.array([ 350, 1000-400 ]),
            #     np.array([ 400, 1000-350 ]),
            #     np.array([ 450, 1000-300 ]),
            # ]

            for id, x in enumerate(range(200,400,20)):
                obj = Landmark(
                    id=id,
                    color=color,
                    radius=radius,
                    position=np.array([x, 250 + x]),
                    window_pad=self.window_pad
                )
                self.objs.add(obj)
            id_last = id

            for id, y in enumerate(range(300,500,20)):
                obj = Landmark(
                    id=id+id_last+1,
                    color=color,
                    radius=radius,
                    position=np.array([600, y]),
                    window_pad=self.window_pad
                )
                self.objs.add(obj)

### -------------------------- AGENT FUNCTIONS -------------------------- ###

    # @timer
    def create_agents(self):
        """
        Instantiates agent objects according to simulation parameters
        Randomly initializes position (center within arena borders)
        Randomly initializes orientation (0 : right, pi/2 : up)
        Adds agent class to PyGame sprite group class (faster operations than lists)
        """
        x_min, x_max, y_min, y_max = self.boundary_info_coll

        if self.N == 1:
            colliding_resources = [0]
            retries = 0 
            while len(colliding_resources) > 0:

                x = np.random.randint(x_min, x_max)
                y = np.random.randint(y_min, y_max)
                # x,y = x_min, y_min
                
                orient = np.random.uniform(0, 2 * np.pi)

                # x,y = 700,800
                # orient = 3
                # x,y = 400,400
                # orient = np.pi

                # inits = [
                #     [700, 1000-200, np.pi], #BR-W
                #     [100, 1000-900, 3*np.pi/2], #TL-S
                #     [800, 1000-900, 3*np.pi/2], #TR-S
                #     [100, 1000-200, np.pi/2], #BL-N
                #     [700, 1000-400, np.pi/2], #TR-N
                #     [600, 1000-900, np.pi], #BR-W
                # ]

                # # plotted + backtracked from pt_eye
                # inits = [
                #     [684.5945945945946+5, 1000-175.67567567567562, np.pi],
                #     [802.2052127020294, 1000-866.9774208911033-5, 3*np.pi/2],
                #     [97.83783783783784, 1000-185.67567567567562-5, 3*np.pi/2],
                #     [97.83783783783784+5, 1000-866.2162162162163, np.pi],
                # ]

                # if self.respawn_counter < len(inits): 
                #    x,y,orient = inits[self.respawn_counter]
                # else:
                #    pass
                

                agent = Agent(
                        id=0,
                        position=(x, y),
                        orientation=orient,
                        max_vel=self.max_vel,
                        FOV=self.agent_fov,
                        vision_range=self.vision_range,
                        num_class_elements=self.num_class_elements,
                        vis_field_res=self.vis_field_res,
                        consumption=self.agent_consumption,
                        model=self.model,
                        boundary_endpts=self.boundary_endpts,
                        window_pad=self.window_pad,
                        radius=self.agent_radii,
                        color=colors.BLUE,
                        vis_transform=self.vis_transform,
                        percep_angle_noise_std=self.percep_angle_noise_std,
                        sim_type=self.sim_type
                    )
                
                colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)

                retries += 1
                if retries > 10: print(f'Retries > 10')
            self.agents.add(agent)

        else: # N > 1

            # starts = np.array([
            #     (100, 100, 5.33),
            #     (100, 300, 0),
            #     (100, 500, 0),
            #     (100, 700, 0),
            #     (100, 900, 0.66),
            #     (300, 100, 4.5),
            #     (500, 100, 4.5),
            #     (700, 100, 4.5),
            #     (900, 100, 3.66),
            #     (900, 300, 3),
            #     (900, 500, 3),
            #     (900, 700, 3),
            #     (900, 900, 2.33),
            #     (300, 900, 1.5),
            #     (500, 900, 1.5),
            #     (700, 900, 1.5),
            # ])

            # starts = np.array([
            #     (50,600,0),
            #     (600,800,3),
            #     (800,700,1.5),
            #     (800,100,3),
            # ])

            for i in range(self.N):

                colliding_resources = [0]
                colliding_agents = [0]

                retries = 0
                while len(colliding_resources) > 0 or len(colliding_agents) > 0:

                    x = np.random.randint(x_min, x_max)
                    y = np.random.randint(y_min, y_max)

                    # x = 950
                    # y = 50 + 100*i
                    
                    orient = np.random.uniform(0, 2 * np.pi)

                    # orient = 3

                    # x,y,orient = starts[i,:]

                    agent = Agent(
                            id=i,
                            position=(x, y),
                            orientation=orient,
                            max_vel=self.max_vel,
                            FOV=self.agent_fov,
                            vision_range=self.vision_range,
                            num_class_elements=self.num_class_elements,
                            vis_field_res=self.vis_field_res,
                            consumption=self.agent_consumption,
                            model=self.model,
                            boundary_endpts=self.boundary_endpts,
                            window_pad=self.window_pad,
                            radius=self.agent_radii,
                            color=colors.BLUE,
                            vis_transform=self.vis_transform,
                            percep_angle_noise_std=self.percep_angle_noise_std,
                            sim_type=self.sim_type
                        )
                    
                    colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)
                    colliding_agents = pygame.sprite.spritecollide(agent, self.agents, False, supcalc.within_group_collision)

                    retries += 1
                    if retries > 10: print(f'Retries > 10')
                self.agents.add(agent)
    
        # self.respawn_counter += 1

    # @timer
    def save_data_agent(self):
        """Tracks key variables (position, mode, resources collected) via array for current timestep"""
        for agent in self.agents:
            x, y = agent.pt_eye
            pos_x = x
            pos_y = self.y_max - y

            if agent.mode == 'explore': mode_num = 0
            elif agent.mode == 'exploit': mode_num = 1
            elif agent.mode == 'collide': mode_num = 2
            else: raise ValueError('Agent Mode not tracked')

            self.data_agent[agent.id, self.t, :] = np.array((pos_x, pos_y, mode_num, agent.collected_r))


    def log_ray_boundary_collision(self, agent, action):
        if action not in self.last_moves:
            # compare last two observations
            vis_diff = [x != y for x,y in zip(agent.vis_field, agent.last_vis_field)]
            vis_diff_idx = np.where(vis_diff)[0]
            # print('') # linebreak
            print(f"t={self.t} \t| Agent {agent.id} turns after seeing change @ {vis_diff_idx}")
            intersect = False

            if len(vis_diff_idx) > 0:
                start_pos = agent.pt_eye + self.window_pad
                self.corner_intersecting_rays = [start_pos]
                for i,phi in enumerate(agent.phis):
                    # print(f"phi: {phi}")
                    end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * 1500,
                                start_pos[1] - np.sin(agent.orientation - phi) * 1500)
                    if self.with_visualization and i in vis_diff_idx:
                        pygame.draw.line(self.screen, colors.BLACK, start_pos, end_pos, 1)

                    for pt in self.boundary_endpts:
                        # dist = supcalc.distance_to_line(pt+self.window_pad, start_pos, end_pos)
                        vec_between = pt - agent.pt_eye
                        angle_bw = supcalc.angle_bw_vis(agent.vec_self_dir, vec_between, agent.radius, np.linalg.norm(vec_between))
                        # print(f"relative offset: {round(np.abs(angle_bw-phi) / self.phi_angle_diff, 2)} | dist: {round(dist,2)}")
                        # print(f"relative offset: {round(np.abs(angle_bw-phi) / self.phi_angle_diff, 2)}")

                        # single ray collision --> find nearest corner
                        if len(vis_diff_idx) == 1 and i == vis_diff_idx[0]:
                            if np.abs(angle_bw-phi) / self.phi_angle_diff < 1:
                                # print(f"\t\t intersect @ {pt}")
                                intersect = True
                                if self.with_visualization: 
                                    pygame.draw.circle(self.screen, colors.BLACK, pt+self.window_pad, 5)

                                    locs = []
                                    loc = agent.pt_eye
                                    for _ in range(250):
                                        rel_pos = loc - pt
                                        rel_ori = np.arctan2(rel_pos[1],rel_pos[0])
                                        travel_ori = rel_ori - phi
                                        travel_ori_comp = np.array((np.cos(travel_ori), np.sin(travel_ori)))
                                        loc = loc - travel_ori_comp*agent.max_vel
                                        locs.append(loc)
                                    self.fwd_traj = np.array(locs)

                                if len(set(agent.vis_field)) == 2:
                                    self.single_ray_colls.append((pt, agent.pt_eye, 2))
                                elif len(set(agent.vis_field)) == 3:
                                    self.single_ray_colls.append((pt, agent.pt_eye, 3))
                                else:
                                    print(f'num walls: {len(set(agent.vis_field))}')

                        # check for ellipse - looser query (single/multi rays), stricter criteria (10% proximity)
                        if np.abs(angle_bw-phi) / self.phi_angle_diff < 0.1:
                            if self.with_visualization: 
                                pygame.draw.circle(self.screen, colors.BLACK, pt+self.window_pad, 5)
                            self.corner_intersecting_rays.append(end_pos)

                # for 2 corners intersecting --> ellipse
                if len(self.corner_intersecting_rays[1:]) == 2:
                    # print(f"\t\t intersect @ {pt}")
                    intersect = True
                    self.dual_ray_colls.append(agent.pt_eye)
                    self.ellipse_ints = self.corner_intersecting_rays
                    self.ellipse_counter = 8

                # no single ray coll + no ellipse
                elif intersect == False:
                    self.colls_no_ellipse.append(agent.pt_eye)
        
        # fading memory of last moves
        self.last_moves.append(action)
        if len(self.last_moves) > 2:
            self.last_moves.pop(0)


    def draw_ray_boundary_collision(self):

        # fading memory of dual corner intersection
        if self.ellipse_counter > 0:
            start_pos = self.ellipse_ints[0]
            for end_pos in self.ellipse_ints[1:]:
                pygame.draw.line(self.screen, colors.VIOLET, start_pos, end_pos, int(self.ellipse_counter))
            self.ellipse_counter -= 1

        # keep last estimated fwd traj
        pygame.draw.lines(self.screen, colors.BLACK, False, self.fwd_traj+self.window_pad, 3)

        for (corner, coll_pt, num_walls) in self.single_ray_colls:
            if num_walls == 2:
                pygame.draw.circle(self.screen, colors.DARK_GREY, coll_pt+self.window_pad, 5)
            elif num_walls == 3:
                pygame.draw.circle(self.screen, colors.BLACK, coll_pt+self.window_pad, 5)

            if corner[0] == 0 and corner[1] == 0:
                pygame.draw.circle(self.screen, colors.TOMATO, coll_pt+self.window_pad, 4)
            elif corner[0] == 0 and corner[1] == 1000:
                pygame.draw.circle(self.screen, colors.GOLD, coll_pt+self.window_pad, 4)
            elif corner[0] == 1000 and corner[1] == 1000:
                pygame.draw.circle(self.screen, colors.LIME, coll_pt+self.window_pad, 4)
            elif corner[0] == 1000 and corner[1] == 0:
                pygame.draw.circle(self.screen, colors.CORN, coll_pt+self.window_pad, 4)
            
        for coll_pt in self.dual_ray_colls:
            pygame.draw.circle(self.screen, colors.VIOLET, coll_pt+self.window_pad, 5)
        
        for coll_pt in self.colls_no_ellipse:
            pygame.draw.circle(self.screen, colors.BLACK, coll_pt+self.window_pad, 3)

### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

    # @timer
    def create_resources(self):

        # creates single resource patch
        id = 0
        units = np.random.randint(self.min_res_units, self.max_res_units)
        quality = np.random.uniform(self.min_res_quality, self.max_res_quality)

        resource = Resource(id, self.res_radius, self.res_pos, units, quality)
        self.resources.add(resource)

        if not self.log_zarr_file: # save in sim instance
            x,y = resource.position
            pos_x = x
            pos_y = self.y_max - y
            self.data_res.append([pos_x, pos_y, self.res_radius])

    # def consume(self, agent):
    #     """Carry out agent-resource interactions (depletion, destroying, notifying)"""
    #     # Call resource agent is on
    #     resource = agent.res_to_be_consumed

    #     # Increment remaining resource quantity
    #     depl_units, destroy_res = resource.deplete(agent.consumption)

    #     # Update agent info
    #     if depl_units > 0:
    #         agent.collected_r += depl_units
    #         agent.mode = 'exploit'
    #     else:
    #         agent.mode = 'explore'

    #     # Kill + regenerate patch when fully depleted
    #     if destroy_res:
    #         resource.kill()
    #         if self.regenerate_resources:
    #             # self.add_new_resource_patch_random()
    #             self.add_new_resource_patch_stationary_single()

### -------------------------- COLLISION FUNCTIONS -------------------------- ###

    # @timer
    def collide_agent_res(self):

        # Create dict of every agent that has collided : [colliding resources]
        collision_group_ar = pygame.sprite.groupcollide(self.agents, self.resources, False, False, pygame.sprite.collide_circle)

        # Switch on all agents currently on a resource 
        for agent, resource_list in collision_group_ar.items():
            for resource in resource_list:
                # Flip bool variable if agent is within patch boundary
                if supcalc.distance(agent.position, resource.position) <= resource.radius:
                    agent.mode = 'exploit'
                    agent.on_res = 1
                    agent.on_res_last_step = 1
                    agent.res_to_be_consumed = resource
                    break

    def collide_agent_wall(self):
        
        # Create dict of every agent that has collided : [colliding walls]
        collision_group_aw = pygame.sprite.groupcollide(self.agents, self.walls, False, False)

        # Change agent mode + note points of contact (carry out velocity-stopping check later in agent.move())
        for agent, wall_list in collision_group_aw.items():

            agent.mode = 'collide'

            for wall in wall_list:

                clip = agent.rect.clip(wall.rect)
                if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

                # print(f'agent {agent.rect.center, agent.position} collided with {wall.id} @ {clip.center}')

                agent.collided_points.append(np.array(clip.center) - self.window_pad)

                # hits = [edge for edge in ['bottom', 'top', 'left', 'right'] if getattr(clip, edge) == getattr(agent.rect, edge)]
                # text = self.font.render(f'Collision at {", ".join(hits)}', True, pygame.Color('black'))
                # self.screen.blit(text, (self.window_pad, int(self.window_pad/2)))

    def collide_agent_objs(self):

        # Create dict of every agent that has collided : [colliding landmarks]
        collision_group = pygame.sprite.groupcollide(self.agents, self.objs, False, False, pygame.sprite.collide_circle)

        # Carry out agent-landmark collisions + generate list of collided landmarks
        for agent, obj_list in collision_group.items():

            agent.mode = 'collide'

            for obj in obj_list:

                # position between agent + landmark
                mid_pos = (agent.position + obj.position) / 2
                # if self.with_visualization: pygame.draw.circle(self.screen, pygame.Color('red'), mid_pos + self.window_pad, 5)
                # print(f'agent {agent.rect.center} collided with {landmark.id} @ {mid_pos}')
                agent.collided_points.append(np.array(mid_pos) - self.window_pad)

    def collide_agent_agent(self):

        # Create dict of every agent that has collided : [colliding agents]
        collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False, supcalc.within_group_collision)
        
        # Carry out agent-agent collisions + generate list of collided agents
        for agent1, other_agents in collision_group_aa.items():

            agent1.mode = 'collide'

            for agentX in other_agents:

                clip = agent1.rect.clip(agentX.rect)
                if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

                # print(f'agent {agent.rect.center} collided with {wall.id} @ {clip.center}')

                agent1.collided_points.append(np.array(clip.center) - self.window_pad)

### -------------------------- HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        for event in events:
            # Exit if requested
            if event.type == pygame.QUIT:
                sys.exit()

            # Pause on Space
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused

            # Speed up on s and down on f. reset default framerate with d
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.framerate -= 1
                if self.framerate < 1:
                    self.framerate = 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.framerate += 1
                if self.framerate > 100:
                    self.framerate = 100
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.framerate = self.framerate_orig


##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = time.time()
        self.create_walls()
        self.create_objs()
        self.create_resources()
        self.create_agents()

        # obs_times = np.zeros(self.T)
        # mod_times = np.zeros(self.T)
        # sav_times = np.zeros(self.T)
        # ful_times = np.zeros(self.T)

        ### ---- START OF SIMULATION ---- ###

        while self.t < self.T:

            if not self.is_paused:

                # self.recorder.capture_frame(self.screen)
                
                ### ---- OBSERVATIONS ---- ###

                # obs_start = time.time()

                # Refresh agent behavioral states
                for agent in self.agents:
                    
                    agent.collided_points = []
                    agent.mode = 'explore'
                    # if agent.on_res_last_step > 0: # 1 timestep memory
                    #     agent.on_res_last_step = 0
                    # elif agent.on_res > 0:
                    #     agent.on_res = 0

                # Evaluate sprite interactions + flip agent modes to 'collide'/'exploit' (latter takes precedence)
                self.collide_agent_wall()
                self.collide_agent_objs()
                self.collide_agent_agent()
                self.collide_agent_res()
                # respawn_counter = 0

                # Update visual projections
                for agent in self.agents:
                    agent.visual_sensing(self.objs, self.agents)

                # obs_times[self.t] = time.time() - obs_start

                ### ---- VISUALIZATION ---- ###

                if self.with_visualization:
                    for agent in self.agents:
                        agent.draw_update() 
                    for res in self.resources:
                        res.draw_update() 
                    self.draw_frame()
                else: # still have to update rect for collisions
                    for agent in self.agents:
                        agent.rect = agent.image.get_rect(center = agent.position + self.window_pad)

                ### ---- TRACKING ---- ### 

                # sav_start = time.time()
                if self.log_zarr_file:
                    tracking.save_agent_data_RAM(self)
                    tracking.save_resource_data_RAM(self)
                else:
                    self.save_data_agent()
                # sav_times[self.t] = time.time() - sav_start

                ### ---- MODEL + ACTIONS ---- ###

                # mod_start = time.time()
                for agent in self.agents:

                    # Observe + encode sensory inputs
                    vis_input = agent.encode_one_hot(agent.vis_field)
                    agent.dist_input = np.array(agent.dist_field)

                    if self.vis_transform != '':
                        if self.vis_transform == 'minmax':
                            agent.dist_input = (agent.dist_input - self.min_dist)/(self.max_dist - self.min_dist) # flipped --> todo: switch + scale
                        elif self.vis_transform == 'far':
                            agent.dist_input = self.min_dist*2 / agent.dist_input # holdover ---> change if using
                        # elif self.vis_transform == 'far':
                        #     dist_input = ( self.min_dist / dist_input + .25 ) / 1.25
                        # elif self.vis_transform == 'mfar':
                        #     dist_input = ( np.power(self.min_dist / dist_input, 1.2) + .25 ) / 1.25
                        # elif self.vis_transform == 'vfar':
                        #     dist_input = ( np.power(self.min_dist / dist_input, 2) + .25 ) / 1.25
                        elif self.vis_transform == 'maxWF':
                            agent.dist_input = 1.465 - np.log(agent.dist_input) / 5 # bounds [min, max] within [0, 1]
                        elif self.vis_transform == 'p9WF':
                            agent.dist_input = 1.29 - np.log(agent.dist_input) / 6.1 # bounds [min, max] within [0.1, 0.9]
                        elif self.vis_transform == 'p8WF':
                            agent.dist_input = 1.09 - np.log(agent.dist_input) / 8.2 # bounds [min, max] within [0.2, 0.8]
                        elif self.vis_transform == 'WF':
                            agent.dist_input = 1.24 - np.log(agent.dist_input) / 7 # bounds [min, max] within [0.2, 0.9]
                        elif self.vis_transform == 'mlWF':
                            agent.dist_input = 1 - np.log(agent.dist_input) / 9.65 # bounds [min, max] within [0.25, 0.75]
                        elif self.vis_transform == 'mWF':
                            agent.dist_input = .9 - np.log(agent.dist_input) / 12 # bounds [min, max] within [0.3, 0.7]
                        elif self.vis_transform == 'msWF':
                            agent.dist_input = .8 - np.log(agent.dist_input) / 16 # bounds [min, max] within [0.35, 0.65]
                        elif self.vis_transform == 'sWF':
                            agent.dist_input = .7 - np.log(agent.dist_input) / 24 # bounds [min, max] within [0.4, 0.6]
                        elif self.vis_transform == 'ssWF':
                            agent.dist_input = .6 - np.log(agent.dist_input) / 48 # bounds [min, max] within [0.45, 0.55]
                        # elif self.vis_transform == 'minmax_buffer':
                        #     dist_input = (dist_input - self.min_dist + 100)/(self.max_dist - self.min_dist + 200)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input *= dist_input
                        # elif self.vis_transform == 'minmax_scalp1':
                        #     dist_input = (dist_input - self.min_dist)/(self.max_dist - self.min_dist)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input -= dist_input*.1
                        # elif self.vis_transform == 'minmax_scalp5':
                        #     dist_input = (dist_input - self.min_dist)/(self.max_dist - self.min_dist)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input -= dist_input*.5

                        # add noise + perturbation + clip
                        agent.dist_input += np.random.randn(agent.dist_input.shape[0]) * self.percep_dist_noise_std
                        # agent.dist_input += np.random.randn(agent.dist_input.shape[0]) * .025
                        # dist_input *= 1.1
                        # dist_input /= 1.5
                        # dist_input += .05
                        agent.dist_input = np.clip(agent.dist_input, 0,1)

                        vis_input *= agent.dist_input

                    # print(np.round(np.sum(vis_input,axis=0),4))

                    # if agent.mode == 'collide': other_input = np.array([agent.on_res, 1])
                    # else:                       other_input = np.array([agent.on_res, 0])

                    # Calculate action
                    if self.other_input == 2:
                        agent.action, agent.hidden = agent.model.forward(vis_input, np.array([agent.on_res, agent.acceleration / self.max_vel]), agent.hidden)
                    else:
                        agent.action, agent.hidden = agent.model.forward(vis_input, np.array([agent.on_res]), agent.hidden)

                    # Food present --> consume (if food is still available)
                    if agent.mode == 'exploit':
                        # agent.kill()
                        # self.create_agents()

                        ### ---- END OF SIMULATION (found food - premature termination) ---- ###

                        pygame.quit()
                        # # compute simulation time in seconds
                        self.elapsed_time = round( (time.time() - start_time) , 2)
                        # if self.print_enabled:
                        #     print(f"Elapsed_time: {self.elapsed_time}")

                        if self.log_zarr_file:
                            # conclude agent/resource tracking
                            # convert tracking agent/resource dicts to N-dimensional zarr arrays + save to offline file
                            ag_zarr, res_zarr = tracking.save_zarr_file(self.t+1, self.save_ext, self.print_enabled)
                            plot_data = ag_zarr, res_zarr
                        else: # use ag/res tracking from self instance
                            # convert list to 3D array similar to zarr file
                            data_res_array = np.zeros( (len(self.data_res), 1, 3 ))
                            for id, (pos_x, pos_y, radius) in enumerate(self.data_res):
                                data_res_array[id, 0, 0] = pos_x
                                data_res_array[id, 0, 1] = pos_y
                                data_res_array[id, 0, 2] = radius

                        #     # assign plot data as numpy arrays
                        #     plot_data = self.data_agent, data_res_array
                        # # display static map of simulation
                        # if self.plot_trajectory:
                        #     plot_funcs.plot_map(plot_data, self.WIDTH, self.HEIGHT, self.coll_boundary_thickness, save_name=self.save_ext)

                        # # extract total fitnesses of each agent + save into sim instance (pulled for EA)
                        # # self.fitnesses = np.array([self.t]) # --> use time taken to find food instead

                        return self.t, 0, self.elapsed_time

                    else: # No food --> move (stay stationary if collided object in front)
                        action = agent.action + np.random.randn()*self.action_noise_std

                        # res_pos = np.array(self.res_pos)
                        # # res_pos[1] = self.y_max - res_pos[1]
                        # disp_from_patch = res_pos - agent.position
                        # angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
                        # angle_diff = angle_to_patch - agent.orientation
                        # # angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                        # angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
                        # print(agent.orientation, angle_to_patch, angle_diff)
                        # angle_diff_scaled = angle_diff / np.pi
                        # action = (2*0.005)**.5 * np.random.uniform(-1,1) + angle_diff_scaled

                        # self.log_ray_boundary_collision(agent, action)
                        # if self.with_visualization: 
                        #     self.draw_ray_boundary_collision()

                        agent.move(action)
                        # agent.move(0.1)
                        # agent.move(np.random.uniform(-0.1,0.1))
                        # rot_diff = 0.001
                        # agent.move((2*rot_diff)**.5 * np.random.uniform(-1,1))

                # mod_times[self.t] = time.time() - mod_start

            ### ---- BACKGROUND PROCESSES ---- ###
        
                # Step sim time forward
                self.t += 1

                # Step clock time to calculate fps
                if self.with_visualization:
                    self.clock.tick(self.framerate)
                    if self.print_enabled and (self.t % 500 == 0):
                        print(f"t={self.t} \t| FPS: {round(self.clock.get_fps(),1)}")
                
                # ful_times[self.t-1] = time.time() - obs_start

            # Carry out user interactions even when not paused
            if self.with_visualization:
                events = pygame.event.get() 
                self.interact_with_event(events)

        ### ---- END OF SIMULATION ---- ###

        # self.recorder.end_recording()
        pygame.quit()

        # compute simulation time in seconds
        self.elapsed_time = round( (time.time() - start_time) , 2)
        if self.print_enabled:
            print(f"Elapsed_time: {self.elapsed_time}")

        if self.log_zarr_file:
            # conclude agent/resource tracking
            # convert tracking agent/resource dicts to N-dimensional zarr arrays + save to offline file
            ag_zarr, res_zarr = tracking.save_zarr_file(self.T, self.save_ext, self.print_enabled)
            plot_data = ag_zarr, res_zarr
        else: # use ag/res tracking from self instance
            # convert list to 3D array similar to zarr file
            data_res_array = np.zeros( (len(self.data_res), 1, 3 ))
            for id, (pos_x, pos_y, radius) in enumerate(self.data_res):
                data_res_array[id, 0, 0] = pos_x
                data_res_array[id, 0, 1] = pos_y
                data_res_array[id, 0, 2] = radius

            # assign plot data as numpy arrays
            plot_data = self.data_agent, data_res_array
        # display static map of simulation
        if self.plot_trajectory:
            plot_funcs.plot_map(plot_data, self.WIDTH, self.HEIGHT, self.coll_boundary_thickness, save_name=self.save_ext)

        # extract total fitnesses + save into sim instance (pulled for EA)
        dist_to_res = supcalc.distance(self.agents.sprites()[0].position, self.resources.sprites()[0].position)
        # self.fitnesses = np.array([self.T + dist_to_res]) # --> max time + proximity as extra error signal
        # self.fitnesses = np.array([self.T]) # --> max time + proximity as extra error signal

        return self.T, dist_to_res, self.elapsed_time