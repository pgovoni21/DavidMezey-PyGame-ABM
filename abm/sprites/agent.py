"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

from abm import colors
from abm.sprites import supcalc
# from abm.helpers import timer

import pygame
import numpy as np

class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """
    # @timer
    def __init__(self, id, position, orientation, max_vel, 
                 FOV, vision_range, num_class_elements, vis_field_res, consumption,
                 model, boundary_endpts, window_pad, radius, color, 
                 vis_transform, percep_angle_noise_std, sim_type
                 ):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent (0: right, pi/2: up, pi: left, 3*pi/2: down)
        :param max_vel: 
        :param vis_field_res: resolution of the visual projection field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param consumption: (resource unit/time unit) consumption efficiency of agent
        :param NN: 
        :param boundary_endpts: 
        :param radius: radius of the agent in pixels
        :param color: color of the agent as (R, G, B)
        """
        # PyGame Sprite superclass
        super().__init__()

        # Unique parameters
        self.id = id 

        # Movement/behavior parameters
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.velocity = 0  # (absolute)
        self.acceleration = 0
        
        self.pt_eye = np.array([
            self.position[0] + np.cos(orientation) * radius, 
            self.position[1] - np.sin(orientation) * radius])

        self.mode = "explore"  # explore / exploit / collide
        self.max_vel = max_vel
        self.collided_points = []

        # Visual field parameters
        self.vis_field_res = vis_field_res
        # FOV = 0.41
        # FOV = 0.45
        self.FOV = FOV
        self.phis = np.linspace(-FOV*np.pi, FOV*np.pi, vis_field_res) # array of raycasts (- : left / + : right)
        self.vision_range = vision_range
        self.num_class_elements = num_class_elements
        self.vis_field = [0] * vis_field_res
        if vis_transform:
            self.dist_field = [0] * vis_field_res
            self.dist_input = np.array(self.dist_field)
        else:
            self.dist_field = None
            self.dist_input = None
        self.percep_angle_noise_std = percep_angle_noise_std

        # Resource parameters
        self.collected_r = 0  # resource units collected by agent 
        self.on_res = 0 # binary : whether agent is currently on top of a resource patch or not
        self.on_res_last_step = 0 # allows on_res to stay on 1 timestep (for agent to use this info next timestep)
        self.consumption = consumption

        # Neural network init
        self.model = model
        self.hidden = None
        self.action = 0

        # Environment related parameters
        TL,TR,BL,BR = boundary_endpts
        self.window_pad = window_pad
        # define names for each endpoint (top/bottom + left/right)
        # TL = np.array([ 100, 0 ])
        # TL = np.array([ 0, 100 ])
        self.boundary_endpts = [
            ('TL', TL),
            ('TR', TR),
            ('BL', BL),
            ('BR', BR)
        ]
        self.extra_coll_block = 0 * np.pi / 180 # extra collision degrees where agent vel = 0 (since clip collision is slow 1 timestep)
        self.sim_type = sim_type

        # Visualization / human interaction parameters
        self.radius = radius
        self.color = color
        # self.selected_color = colors.BLACK
        # self.show_stats = False
        # self.is_moved_with_cursor = 0

        # Initializing body + position
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(colors.WHITE)
        self.image.set_colorkey(colors.WHITE)
        self.rect = self.image.get_rect(center = self.position + self.window_pad)

### -------------------------- VISUAL FUNCTIONS -------------------------- ###

    # @timer
    def gather_self_percep_info(self):
        """
        update position/direction points + vector of self
        """
        # front-facing point on agent's perimeter according to its orientation
        self.pt_eye = np.array([
            self.position[0] + np.cos(self.orientation) * self.radius, 
            self.position[1] - np.sin(self.orientation) * self.radius])

        # direction vector, magnitude = radius, flipped y-axis
        self.vec_self_dir = self.pt_eye - self.position
        ## where v1[0] --> + : right, 0 : center, - : left, 10 : max
        ## where v1[1] --> + : down, 0 : center, - : up, 10 : max

    # @timer
    def gather_boundary_endpt_info(self):
        """
        create dictionary storing visually relevant information for each boundary endpoint
        """
        # print()
        # print(f'self \t {np.round(self.vec_self_dir/self.radius,2)} \t {np.round(self.orientation*90/np.pi,0)}')
        # print(np.round(self.phis*90/np.pi,0))

        self.boundary_endpt_dict = {}
        for endpt_name, endpt_coord in self.boundary_endpts:

            # calc vector between boundary endpoint + direction point (front-facing)
            vec_between = endpt_coord - self.pt_eye

            # calc magnitude/norm
            distance = np.linalg.norm(vec_between)

            # calc orientation angle
            angle_bw = supcalc.angle_bw_vis(self.vec_self_dir, vec_between, self.radius, distance)
            ## relative to perceiving agent, in radians between [-pi (left/CCW), +pi (right/CW)]

            # update dictionary with added info
            self.boundary_endpt_dict[endpt_name] = (angle_bw, endpt_coord)

            # print(f'{endpt_name} \t {np.round(vec_between/distance,2)} \t {np.round(angle_bw*90/np.pi,0)}'

    # @timer
    def gather_boundary_wall_info(self):

        # initialize wall dict
        self.vis_field_wall_dict = {}
        # strings to name walls + call corresponding L/R endpts
        walls = [
            ('wall_north', 'TL', 'TR'),
            ('wall_south', 'BR', 'BL'),
            ('wall_east', 'TR', 'BR'),
            ('wall_west', 'BL', 'TL'),
        ]
        # loop over the 4 walls
        for wall_name, pt_L, pt_R in walls:

            # unpack dict entry for each corresponding endpt
            angle_L, coord_L = self.boundary_endpt_dict[pt_L]
            angle_R, coord_R = self.boundary_endpt_dict[pt_R]

            self.vis_field_wall_dict[wall_name] = {}
            self.vis_field_wall_dict[wall_name]['angle_L'] = angle_L
            self.vis_field_wall_dict[wall_name]['angle_R'] = angle_R
            self.vis_field_wall_dict[wall_name]['coord_L'] = coord_L
            self.vis_field_wall_dict[wall_name]['coord_R'] = coord_R


    def gather_obj_info(self, objs):

        # initialize obj dict
        self.vis_field_obj_dict = {}

        # for all objs in the simulation
        for obj in objs:

            # exclude self from list
            if obj.id != self.id:

                # exclude objs outside range of vision (calculate distance bw obj center + self eye)
                obj_coord = obj.position
                vec_between = obj_coord - self.pt_eye
                obj_distance = np.linalg.norm(vec_between)

                if obj_distance <= self.vision_range:

                    # exclude objs outside FOV limits (calculate visual boundaries of obj)

                    # orientation angle relative to perceiving obj, in radians between [-pi (left/CCW), +pi (right/CW)]
                    angle_bw = supcalc.angle_bw_vis(self.vec_self_dir, vec_between, self.radius, obj_distance)

                    # # apply noise
                    # angle_bw += np.random.randn()*self.LM_angle_noise_std
                    # # angle_bw += np.random.randn()*.05
                    # lm_radius = obj.radius + np.random.randn()*self.LM_radius_noise_std
                    # obj_distance += np.random.randn()*self.LM_dist_noise_std
                    # # obj_distance += np.random.randn()*200
                    # # ensure no negative radii/distances
                    # lm_radius = np.maximum(lm_radius, 0.01)
                    # obj_distance = np.maximum(obj_distance, 0.01)
    
                    # exclusionary angle between obj + self, taken to L/R boundaries
                    angle_edge = np.arctan(obj.radius / obj_distance)
                    angle_L = angle_bw - angle_edge
                    angle_R = angle_bw + angle_edge
                    # unpack L/R angle limits of visual projection field
                    phi_L_limit = self.phis[0]
                    phi_R_limit = self.phis[-1]
                    if (phi_L_limit <= angle_L <= phi_R_limit) or (phi_L_limit <= angle_R <= phi_R_limit): 

                        # update dictionary with all relevant info
                        obj_name = f'obj_{obj.id}'
                        self.vis_field_obj_dict[obj_name] = {}
                        self.vis_field_obj_dict[obj_name]['id'] = obj.id
                        self.vis_field_obj_dict[obj_name]['distance'] = obj_distance
                        self.vis_field_obj_dict[obj_name]['angle_L'] = angle_L
                        self.vis_field_obj_dict[obj_name]['angle_R'] = angle_R
                        self.vis_field_obj_dict[obj_name]['coord'] = obj_coord

    # @timer
    def gather_agent_info(self, agents):

        # initialize agent dict
        self.vis_field_agent_dict = {}

        # for all agents in the simulation
        for ag in agents:

            # exclude self from list
            if ag.id != self.id:

                # orientation angle relative to perceiving agent, in radians between [-pi (left/CCW), +pi (right/CW)]
                agent_coord = ag.position
                vec_between = agent_coord - self.pt_eye
                agent_distance = np.linalg.norm(vec_between)
                angle_bw = supcalc.angle_bw_vis(self.vec_self_dir, vec_between, self.radius, agent_distance)
                # exclusionary angle between agent + self, taken to L/R boundaries
                angle_edge = np.arctan(self.radius / agent_distance)
                angle_L = angle_bw - angle_edge
                angle_R = angle_bw + angle_edge
                # unpack L/R angle limits of visual projection field
                phi_L_limit = self.phis[0]
                phi_R_limit = self.phis[-1]

                # exclude agents outside FOV limits (calculate visual boundaries of agent)
                if (phi_L_limit <= angle_L <= phi_R_limit) or (phi_L_limit <= angle_R <= phi_R_limit): 

                    # update dictionary with all relevant info
                    agent_name = f'agent_{ag.id}'
                    self.vis_field_agent_dict[agent_name] = {}
                    self.vis_field_agent_dict[agent_name]['mode'] = ag.mode
                    self.vis_field_agent_dict[agent_name]['distance'] = agent_distance
                    self.vis_field_agent_dict[agent_name]['angle_L'] = angle_L
                    self.vis_field_agent_dict[agent_name]['angle_R'] = angle_R

    # @timer
    def fill_vis_field_walls(self):
        """
        Mark projection field according to each wall
        """
        # for each discretized perception angle within FOV range
        # for i in range(self.vis_field_res):
        for i,phi in enumerate(self.phis):

            # look for intersections
            for obj_name, v in self.vis_field_wall_dict.items():
                if v["angle_L"] <= self.phis[i] <= v["angle_R"]:
                    self.vis_field[i] = obj_name

                    if self.dist_field is not None:
                        self.fill_dist_field(i, phi, v["coord_L"], v["coord_R"])
            
            # no intersections bc one endpoint is behind back, iterate again
            if self.vis_field[i] == 0:
                for obj_name, v in self.vis_field_wall_dict.items():
                    if v["angle_L"] > v["angle_R"]:
                        self.vis_field[i] = obj_name

                        if self.dist_field is not None:
                            self.fill_dist_field(i, phi, v["coord_L"], v["coord_R"])

    def fill_dist_field(self, i, phi, coord_L, coord_R):

        # transfrom perception ray angle to vector to coord
        orient_global = phi - self.orientation
        vec_percep = np.array([ np.cos(orient_global), np.sin(orient_global) ])
        pt_percep = self.pt_eye + vec_percep

        # find where perception ray intersects with boundary/object + calc distance
        pt_cross = supcalc.get_intersection( coord_L,coord_R, self.pt_eye,pt_percep )
        if pt_cross is None: print('error: perception ray is parallel to perceived edge') # shouldn't happen

        # fill in appropriate field entry with calculated distance
        distance = np.linalg.norm(self.pt_eye - pt_cross)
        self.dist_field[i] = distance

    def fill_vis_field_objs(self):
        """
        Mark projection field according to each obj
        """
        # for each discretized perception angle within FOV range
        for i, phi in enumerate(self.phis):

            # look for intersections + keep list of occluded surfaces
            occ_objs = []
            for obj_name, v in self.vis_field_obj_dict.items():
                if v["angle_L"] <= phi <= v["angle_R"]:
                    occ_objs.append( (v["distance"], obj_name) )

            if occ_objs:
                # use closest object to fill in field
                occ_objs.sort()
                dist, name = occ_objs[0]

                self.vis_field[i] = name

                if self.dist_field is not None:
                    self.dist_field[i] = dist

    # @timer
    def fill_vis_field_agents(self):
        """
        Mark projection field according to each agent
        """
        # for each discretized perception angle within FOV range
        for i in range(self.vis_field_res):

            # look for intersections + keep list of occluded surfaces
            occ_objs = []
            for obj_name, v in self.vis_field_agent_dict.items():
                if v["angle_L"] <= self.phis[i] <= v["angle_R"]:
                    occ_objs.append( (v["distance"], obj_name, v["mode"]) )

            if occ_objs:
                # use closest object to fill in field
                occ_objs.sort()
                _, _, mode = occ_objs[0]

                if mode == "exploit": 
                    self.vis_field[i] = "agent_exploit"
                else: # mode == "explore" or "collide": 
                    self.vis_field[i] = "agent_explore"

    # @timer
    def visual_sensing(self, objs, agents):
        """
        Accumulates visual sensory functions
        """
        # Carry last
        self.last_vis_field = self.vis_field
        # Zero from previous step
        self.vis_field = [0] * self.vis_field_res
        if self.dist_field is not None:
            self.dist_field = [0] * self.vis_field_res
        
        # Add perceptual noise if specified
        orientation_real = self.orientation
        noise = np.random.randn()*self.percep_angle_noise_std
        # noise += np.random.randn()*.1
        # if noise > 0: noise = 0
        # if noise < 0: noise = 0
        self.orientation += noise

        # Gather relevant info for self / boundary endpoints / walls
        self.gather_self_percep_info()
        if not self.sim_type.startswith('nowalls'):
            self.gather_boundary_endpt_info()
            self.gather_boundary_wall_info()
            self.gather_obj_info(objs)
        if len(agents) > 1: 
            self.gather_agent_info(agents)

        # Fill in vis_field with id info (wall name / agent mode) for each visual perception ray
        if not self.sim_type.startswith('nowalls'):
            self.fill_vis_field_walls()
            self.fill_vis_field_objs()
        if len(agents) > 1: 
            self.fill_vis_field_agents()

        # Reset orientation
        self.orientation = orientation_real

### -------------------------- MOVEMENT FUNCTIONS -------------------------- ###

    def bind_orientation(self):
        """
        Restricts agent's orientation angle to [0 : 2*pi]
        """
        while self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        while self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    # @timer
    def move(self, NN_output):
        """
        Incorporates NN outputs (change in orientation, or velocity + orientation)
        Calculates next position with collisions as absorbing boundary conditions
        """
        # NN output via tanh scales to a range of [-1 : 1]
        # Scale to max 90 deg turns [-pi/2 : pi/2] per timestep
        turn = NN_output * np.pi / 2
        # turn = NN_output * np.pi / 4 # actspacehalf

        # Shift orientation accordingly + bind to [0 : 2pi]
        self.orientation += turn
        # self.orientation += np.pi/16
        self.bind_orientation()

        # Update velocity (constrained by turn angle)
        velocity_last_step = self.velocity
        self.velocity = self.max_vel * (1 - abs(NN_output))
        # self.velocity = self.max_vel * np.exp(- 0.5 * (NN_output)**2 / (0.1)**2 ) # actspacenarrow
        self.acceleration = self.velocity - velocity_last_step

        # Check for velocity-stopping collisions for each point of contact
        if self.mode == 'collide':
            for pt in self.collided_points:

                # calc vector between collided point + agent center
                vec_coll = pt - self.position

                # calc orientation angle
                distance = np.linalg.norm(vec_coll)
                angle_coll = supcalc.angle_bw_coll(vec_coll, np.array([10,0]), distance, self.radius)
                # print(np.round(angle_coll,3), np.round(self.orientation,3))

                # check if collision angle is within 180d of current orientation + wrapping constraints
                L_limit = angle_coll - np.pi/2 - self.extra_coll_block
                R_limit = angle_coll + np.pi/2 + self.extra_coll_block
                # print(np.round(L_limit,3), np.round(R_limit,3))

                if R_limit > 2*np.pi:
                    if L_limit < self.orientation or (R_limit - 2*np.pi) > self.orientation:
                        self.velocity = 0
                        # print('block wrap +')
                elif L_limit < 0:
                    if R_limit > self.orientation or (L_limit + 2*np.pi) < self.orientation:
                        self.velocity = 0
                        # print('block wrap -')
                else:
                    if L_limit < self.orientation < R_limit:
                        self.velocity = 0
                        # print('block')

        # Calculate agent's next position
        orient_comp = np.array((
            np.cos(self.orientation),
            -np.sin(self.orientation)
        ))
        self.position += self.velocity * orient_comp

### -------------------------- NEURAL NETWORK FUNCTIONS -------------------------- ###

    # @timer
    def encode_one_hot(self, field):
        """
        one hot encode the visual field according to class indices:
            single-agent: (wall_east, wall_north, wall_west, wall_south)
            multi-agent: (wall_east, wall_north, wall_west, wall_south, agent_expl, agent_nonexpl)
        """
        field_onehot = np.zeros((self.num_class_elements, len(field)))

        if self.num_class_elements == 4:
            for i,x in enumerate(field):
                if x == 'wall_north': field_onehot[0,i] = 1
                elif x == 'wall_south': field_onehot[1,i] = 1
                elif x == 'wall_east': field_onehot[2,i] = 1
                elif x == 'wall_west': field_onehot[3,i] = 1
                # else x == 'obj': field_onehot[4,i] = 1

        elif self.num_class_elements == 2:
            for i,x in enumerate(field):
                if x == 'agent_explore': field_onehot[0,i] = 1
                elif x == 'agent_exploit': field_onehot[1,i] = 1
                # else nothing is perceived

        elif self.num_class_elements == 6:
            for i,x in enumerate(field):
                if x == 'wall_north': field_onehot[0,i] = 1
                elif x == 'wall_south': field_onehot[1,i] = 1
                elif x == 'wall_east': field_onehot[2,i] = 1
                elif x == 'wall_west': field_onehot[3,i] = 1
                elif x == 'agent_explore': field_onehot[4,i] = 1
                elif x == 'agent_exploit': field_onehot[5,i] = 1
                else: print('error - nothing is perceived')

        return field_onehot
    
    def encode_labels(self, field):
        """
        one hot encode the visual field according to class indices:
            single-agent: (wall_east, wall_north, wall_west, wall_south)
            multi-agent: (wall_east, wall_north, wall_west, wall_south, agent_expl, agent_nonexpl)
        """
        field_onehot = np.zeros(len(field))

        for i,x in enumerate(field):
            if x == 'wall_north': field_onehot[i] = -.75
            elif x == 'wall_south': field_onehot[i] = .25
            elif x == 'wall_east': field_onehot[i] = .25
            elif x == 'wall_west': field_onehot[i] = .75
            # elif x == 'agent_exploit': field_onehot[4,i] = 1
            # else: # x == 'agent_explore
            #     field_onehot[5,i] = 1
        return field_onehot


### -------------------------- VISUALIZATION / HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.mode == 'explore':
            self.color = colors.BLUE
        elif self.mode == 'exploit':
            self.color = colors.GREEN
        elif self.mode == 'collide':
            self.color = colors.RED
    # @timer
    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # change agent color according to mode
        self.change_color()

        # update surface according to new orientation
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)
        pygame.draw.line(self.image, colors.WHITE, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius), 3)
        self.rect = self.image.get_rect(center = self.position + self.window_pad)


    # def move_with_mouse(self, mouse, left_state, right_state):
    #     """Moving the agent with the mouse cursor, and rotating"""
    #     if self.rect.collidepoint(mouse):
    #         # setting position of agent to cursor position
    #         self.position = mouse - self.radius
    #         if left_state:
    #             self.orientation += 0.1
    #         if right_state:
    #             self.orientation -= 0.1
    #         self.prove_orientation()
    #         self.is_moved_with_cursor = 1
    #         # updating agent visualization to make it more responsive
    #         self.draw_update()
    #     else:
    #         self.is_moved_with_cursor = 0
