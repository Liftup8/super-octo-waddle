import math
import numpy as np
import gym
gym.logger.set_level(40)
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation as R
import pyglet
import random

from config import Config
from ACEnvironment import ACEnvironment2D

<<<<<<< HEAD
=======

>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
class DubinsAC2Denv(gym.Env):
    # gym.Env creates an environment that follows gym interface
    metadata = {'render.modes': ['human']}

    _redAC = None  # Null
    _blueAC = None  # Null

    _vel_mps = None  # No value initially
    _action_time_s = None  # No value initially

    def __init__(self,
                 actions='discrete'
                 ):  # using discrete action space for air combat scenario
        self._load_config()
        self.viewer = None  # related to rendering the episode

        self._vel_mps = 20  # velocity in terms of pixel/s
        self._action_time_s = 0.5  # time step in environment
        self.actionIntegral = 0

        # 'err_x': spaces.Box(low=-self.area_width, high=self.area_width, shape=(1,), dtype=np.float32),
        # 'err_y': spaces.Box(low=-self.area_height, high=self.area_height, shape=(1,), dtype=np.float32),
        # 'LOS_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'ATA_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'AA_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'redATA_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'blue_heading': spaces.Box(low=0, high=359, shape=(1,), dtype=np.float32),
        # 'blue_bank': spaces.Box(low=-90, high=90, shape=(1,), dtype=np.float32)

        # err_x: difference in x position
        # err_y: difference in y position
        # los_deg: line of sight degree
        # ata_deg: ata angle wrt blue ac
        # aa_deg: aa angle wrt blue ac
        # redata_deg: ata angle wrt red ac
        # blue_heading: yaw(heading) angle of blue ac
        # blue_bank: roll(bank) angle of blue ac

        lowlim = np.array(
            [-800., -self.window_height, -180., -180., -180., -180., 0.,
             -90.])  # lower limit of feature values
        highlim = np.array(
            [800., self.window_height, 180., 180., 180., 180., 359.,
             90.])  # upper limit of feature values

        self.observation_space = spaces.Box(
            low=lowlim, high=highlim,
            dtype=np.float32)  # creating the observation space

        if actions == 'discrete':  # using discrete actions
            self.action_space = spaces.Discrete(15)
# there are action values from 0 to 14 to enable both negative and positive bank angles along with creating 6 different bank angle values
# from 0 to 90 degrees
        else:
            self.action_space = spaces.Box(low=-1.,
                                           high=1.,
                                           shape=(1, ),
                                           dtype=np.float32)

        self.seed(2)
        self.d_min = 25
        self.d_max = 150

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(
        self, action
    ):  # by using input action, environment returns an observation and reward
        """ This function returns new state, reward, terminal info for one training time step. Output of env.step function returns new line per training time step"""
        assert self.action_space.contains(action)
        cmd_bank_deg = (action - 7) * 90 / 7  # bank angle command to blue ac
        # action takes values from 0 to 14

        self._blueAC.takeaction(
            cmd_bank_deg, 0, self._vel_mps,
            self._action_time_s)  # aircraft takes action based on bank command

        self._redAC.takeaction(0, 0, self._vel_mps, self._action_time_s)
        """In part below red aircraft bounces back from edges of map to stay in sight of user"""
        if self._redAC._pos_m[0] > self.window_width:
            self._redAC._heading_rad = np.mod(
                self._redAC._heading_rad - np.pi / 2, 2 * np.pi)
        elif self._redAC._pos_m[0] < 0:
            self._redAC._heading_rad = np.mod(
                self._redAC._heading_rad + np.pi / 2, 2 * np.pi)

        if self._redAC._pos_m[1] > self.window_height:
            self._redAC._heading_rad = np.mod(
                self._redAC._heading_rad - np.pi / 2, 2 * np.pi)
        elif self._redAC._pos_m[1] < 0:
            self._redAC._heading_rad = np.mod(
                self._redAC._heading_rad + np.pi / 2, 2 * np.pi)

        self.actionIntegral += (cmd_bank_deg * cmd_bank_deg * 0.25 * 0.0001)

        envSta = self._get_sta_env_v2(
        )  # envSta list is what will be fed into Q-network

        reward, terminal, damage, info = self._terminal_reward_2()

        return envSta, reward, terminal, damage, info
# info: diagnostic information useful for debugging
# terminal(or done): A boolean value stating whether it’s time to reset the environment again

<<<<<<< HEAD
    def reset(self): # env.reset() sets new episode initial position by random
        pos , head = self._random_pos2()
=======
    def reset(self):
        pos, head = self._random_pos2()
>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
        pos[0] += 200
        pos[1] += 200
        self._redAC = ACEnvironment2D(position=np.array([pos[0], pos[1], 0]),
                                      vel_mps=self._vel_mps,
                                      heading_deg=head)

        bpos, bhead = self._random_pos()

        i = 0
        while (self._distance(pos, bpos) < 200) and i < 20:
            bpos, bhead = self._random_pos()
            i += 1

        _, _, hdg = self._calc_posDiff_hdg_deg(bpos, pos)
        self._blueAC = ACEnvironment2D(
            position=np.array([bpos[0], bpos[1], 0]),
            vel_mps=self._vel_mps,
            heading_deg=bhead)  # creates _blueAC object from class

        return self._get_sta_env_v2()

    def render(self, mode='human'):
<<<<<<< HEAD
        
        self.r_min = 25
        self.r_max = 150
=======

>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width,
                                           self.window_height)
            display = pyglet.canvas.Display()
            screen = display.get_default_screen()
            screen_width = screen.width
            screen_height = screen.height
            self.viewer.window.set_location(
                screen_width - self.window_height // 2,
                screen_height - self.window_height // 2)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        @self.viewer.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.Q:
                self.viewer.close()

        import os
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # draw red aircraft
        pos, _, att, pos_hist = self._redAC.get_sta()
<<<<<<< HEAD
        self.viewer.draw_polyline(pos_hist[ ::5 , [-2, -3]],) # draws line
        red_ac_img = rendering.Image(os.path.join(__location__, 'images/f16_red.png'), 48, 48)
        red_ac_img._color.vec4 = (1, 1, 1, 1) # siyah renk hatası giderildi / 15.02 
        jtransform = rendering.Transform(rotation= -att[2], translation= np.array([pos[1], pos[0]]))
        red_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(red_ac_img)

        transform2 = rendering.Transform(translation=(self.goal_pos[1], self.goal_pos[0]))  # Relative offset
        self.viewer.draw_circle(1).add_attr(transform2) # uçağın merkezindeki nokta yok edildi / 15.02

        transform2 = rendering.Transform(translation=(self.goal_pos[1], self.goal_pos[0]))  # Relative offset
        self.viewer.draw_circle(self.r_min,filled=False).add_attr(transform2)

        transform3 = rendering.Transform(translation=(pos[1], pos[0]))  # red dangerous circle
        self.viewer.draw_circle(self.r_max,filled=False).add_attr(transform3)

=======
        red_ac_img = rendering.Image(
            os.path.join(__location__, 'images/f16_red.png'), 48, 48)
        red_ac_img._color.vec4 = (1, 1, 1, 1)
        jtransform = rendering.Transform(rotation=-att[2],
                                         translation=np.array([pos[1],
                                                               pos[0]]))
        red_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(red_ac_img)
        foo = len(pos_hist) - 200 if len(pos_hist) > 200 else 0
        self.viewer.draw_polyline(pos_hist[foo::5, [-2, -3]],
                                  color=(0.9, 0.15, 0.2),
                                  linewidth=1.5)
        foo1 = (pos[1], pos[0])
        foo2 = (pos[1] + np.cos(-att[2] + np.deg2rad(30) + np.pi / 2) * 150,
                pos[0] + np.sin(-att[2] + np.deg2rad(30) + np.pi / 2) * 150)
        foo3 = (pos[1] + np.cos(-att[2] - np.deg2rad(30) + np.pi / 2) * 150,
                pos[0] + np.sin(-att[2] - np.deg2rad(30) + np.pi / 2) * 150)
        test = self.viewer.draw_polygon((foo1, foo2, foo3))
        index = 0
        test._color.vec4 = (.9, .15, .2, .3)

        # transform2 = rendering.Transform(translation=(self.goal_pos[1], self.goal_pos[0]))  # Relative offset
        # self.viewer.draw_circle().add_attr(transform2)

        gridx = np.arange(0, self.window_width + 1, self.window_width)
        gridy = np.arange(0, self.window_height + 1, self.window_height)
        ystep = 5
        xstep = 5
        for foo in np.linspace(0, self.window_height, ystep * 5 + 1):
            self.viewer.draw_line((0, foo), (self.window_width, foo),
                                  color=(.8, .8, .8))
        for foo in np.linspace(0, self.window_width, xstep * 5 + 1):
            self.viewer.draw_line((foo, 0), (foo, self.window_height),
                                  color=(.8, .8, .8))
        for foo in np.linspace(0, self.window_height, ystep + 1):
            test = self.viewer.draw_line((0, foo), (self.window_width, foo))
            test.linewidth.stroke = 2
        for foo in np.linspace(0, self.window_width, xstep + 1):
            test = self.viewer.draw_line((foo, 0), (foo, self.window_height))
            test.linewidth.stroke = 2

        label = pyglet.text.Label('Hello, world',
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=10,
                                  y=10)

        label.draw()

        transform2 = rendering.Transform(
            translation=(self.goal_pos[1],
                         self.goal_pos[0]))  # Relative offset
        self.viewer.draw_circle(self.d_min, filled=False).add_attr(transform2)

        transform3 = rendering.Transform(
            translation=(pos[1], pos[0]))  # red dangerous circle
        self.viewer.draw_circle(self.d_max, filled=False).add_attr(transform3)
>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789

        # draw blue aircraft
        pos, _, att, pos_hist = self._blueAC.get_sta()
        foo = len(pos_hist) - 200 if len(pos_hist) > 200 else 0
        self.viewer.draw_polyline(pos_hist[foo::5, [-2, -3]],
                                  color=(0.00, 0.28, 0.73),
                                  linewidth=1.5)
        blue_ac_img = rendering.Image(
            os.path.join(__location__, 'images/f16_blue.png'), 48, 48)
        blue_ac_img._color.vec4 = (1, 1, 1, 1)
        jtransform = rendering.Transform(rotation=-att[2],
                                         translation=np.array([pos[1],
                                                               pos[0]]))
        blue_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(blue_ac_img)

        # Cones
        foo1 = (pos[1], pos[0])
        foo2 = (pos[1] + np.cos(-att[2] + np.deg2rad(30) + np.pi / 2) * 150,
                pos[0] + np.sin(-att[2] + np.deg2rad(30) + np.pi / 2) * 150)
        foo3 = (pos[1] + np.cos(-att[2] - np.deg2rad(30) + np.pi / 2) * 150,
                pos[0] + np.sin(-att[2] - np.deg2rad(30) + np.pi / 2) * 150)
        test = self.viewer.draw_polygon((foo1, foo2, foo3))
        test._color.vec4 = (0.30, 0.65, 1.00, .3)

        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height
        self.EPISODES = Config.EPISODES
        self.G = Config.G
        self.tick = Config.tick
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.max_steps = Config.max_steps

    def _random_pos(self):
        pos0 = np.array([self.window_width / 4, self.window_height / 4])

        return (pos0 + np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width / 2, self.window_height / 2])),
                np.random.uniform(low=-180, high=180))

    def _random_pos2(self):
        return (np.random.uniform(low=np.array([0, 0]),
                                  high=np.array([
                                      self.window_width * 0.5,
                                      self.window_height * 0.5
                                  ])), np.random.uniform(low=-180, high=180))

    def _get_sta_env_v2(
            self):  # returns state of blue aircraft which is the input of DQN
        # this function creates features based on angles
        Rpos, Rvel, Ratt_rad, _ = self._redAC.get_sta()
        Bpos, Bvel, Batt_rad, _ = self._blueAC.get_sta()
        self.Bpos = Bpos  # position of blue aircraft at the current time (x, y)
        self.Rpos = Rpos  # position of red aircraft at the current time (x, y)

        target_dist = np.array([0., 0., 0.])
        r = R.from_euler('zyx', [Ratt_rad[2], Ratt_rad[1], Ratt_rad[0]])
        target_dist = np.matmul(r.as_matrix(), target_dist)
        self.goal_pos = Rpos - target_dist

        self.errPos, self.errDist, _ = self._calc_posDiff_hdg_rad(
            Bpos, self.goal_pos)
        _, _, self.LOS_deg = self._calc_posDiff_hdg_rad(Bpos, Rpos)

        self.ATA_deg = np.rad2deg(self._pi_bound(self.LOS_deg - Batt_rad[2]))
        self.AA_deg = np.rad2deg(self._pi_bound(Ratt_rad[2] - self.LOS_deg))

        self.LOS_deg = np.rad2deg(self._pi_bound(self.LOS_deg))

        _, self.targetDist, self.redATA_deg = self._calc_posDiff_hdg_rad(
            Rpos, Bpos)
        self.redATA_deg = np.rad2deg(
            self._pi_bound(self.redATA_deg - Ratt_rad[2]))

        return np.array([
            self.errPos[0], self.errPos[1], self.LOS_deg, self.ATA_deg,
            self.AA_deg, self.redATA_deg,
            np.rad2deg(Batt_rad[2]),
            np.rad2deg(Batt_rad[0])
        ],
                        dtype=np.float32)

    def _get_sta_env_v2_redAC(self):  # return state of red aircraft
        Rpos, Rvel, Ratt_rad, _ = self._redAC.get_sta()
        Bpos, Bvel, Batt_rad, _ = self._blueAC.get_sta()

        # target_dist = np.array([70., 0., 0.])
        target_dist = np.array([0., 0., 0.])
        r = R.from_euler('zyx', [Batt_rad[2], Batt_rad[1], Batt_rad[0]])
        target_dist = np.matmul(r.as_matrix(), target_dist)
        goal_pos = Bpos - target_dist

        errPos, errDist, _ = self._calc_posDiff_hdg_rad(Rpos, goal_pos)
        _, _, LOS_deg = self._calc_posDiff_hdg_rad(Rpos, Bpos)

        ATA_deg = np.rad2deg(self._pi_bound(LOS_deg - Ratt_rad[2]))
        AA_deg = np.rad2deg(self._pi_bound(Batt_rad[2] - LOS_deg))

        LOS_deg = np.rad2deg(self._pi_bound(LOS_deg))

        _, targetDist, redATA_deg = self._calc_posDiff_hdg_rad(Bpos, Rpos)
        redATA_deg = np.rad2deg(self._pi_bound(redATA_deg - Batt_rad[2]))

        return np.array([
            errPos[0], errPos[1], LOS_deg, ATA_deg, AA_deg, redATA_deg,
            np.rad2deg(Ratt_rad[2]),
            np.rad2deg(Ratt_rad[0])
        ],
                        dtype=np.float32)

<<<<<<< HEAD
    def _terminal_reward_2(self): # before assignment hatası
        self.d_min = 25
        self.d_max = 150
        INFO = 'win/loss' # değiştirilecek
        REWARD = 0 # credit: kerem aydın
=======
    def _terminal_reward_2(self):  # before assignment hatası
        INFO = 'win/loss'
        REWARD = 0
>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
        DAMAGE_redAC = 0
        distance_ = math.sqrt(
            (self.Bpos[0] - self.Rpos[0])**2 +
            (self.Bpos[1] - self.Rpos[1])**2)  # distance between two aircrafts
        TERMINALSTATE = False
        # UnboundLocalError: local variable 'reward' referenced before assignment hatası çözüldü: if loopundan önce initialize edildi ve başka isimle kullanıldı
<<<<<<< HEAD
        if 0 < self.ATA_deg < 60 and 0 < self.AA_deg < 30 and self.d_min < distance_ < self.d_max: # dominant area, blue win (for how much duration? 1 action-time?)
            if random.random() < 0.8: # chance to hit enemy in dominant area
              REWARD = 100
              DAMAGE_redAC = 1
              print('Red aircraft was hit!\n')
            else:
              REWARD = 50
              DAMAGE_redAC = 0
              print('Missed gunfire on red aircraft!\n')
        elif 180 > self.ATA_deg > 120 and 180 > self.AA_deg > 150 and self.d_min < distance_ < self.d_max:
=======
        if self.ATA_deg < 60 and self.AA_deg < 30 and self.d_min < distance_ < self.d_max:  # dominant area, blue win (for how much duration? 1 action-time?)
            if random.random() < 0.8:  # chance to hit enemy in dominant area
                REWARD = 100
                DAMAGE_redAC = 1
                print('\n[HIT] Red')
            else:
                REWARD = 50
                DAMAGE_redAC = 0
                print('\n[MISS] Blue')
        elif self.ATA_deg > 120 and self.AA_deg > 150 and self.d_min < distance_ < self.d_max:
>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
            REWARD = -500
            print('\n[DOM] Red')
            TERMINALSTATE = True
        if (any(self.Bpos < 0) or any(self.Bpos > self.window_height)):
            print('\n[OOB] Blue')
            REWARD = -100
            TERMINALSTATE = True
<<<<<<< HEAD
        elif self.Bpos[0] < 0: # x ekseninde negatif tarafa geçilirse
            REWARD = -100
            print(' Blue is out of defined limits (Exceeded lower limit)')
            TERMINALSTATE = True
        elif self.Bpos[1] > self.window_width: # y ekseninde üst limit aşılırsa
            # blueac position source # pozisyona erişilemiyor # x dikey eksen y yatay eksen
            REWARD = -100
            print(' Blue is out of defined limits (Exceeded rightmost limit)')
            TERMINALSTATE = True
        elif self.Bpos[1] < 0: # y ekseninde negatif tarafa geçilirse
            REWARD = -100
            print(' Blue is out of defined limits (Exceeded leftmost limit)')
            TERMINALSTATE = True
        elif distance_<= self.d_min: # collision reward (terminal)
            REWARD = -300
            print(' Aircrafts crashed into each other!')
            TERMINALSTATE = True
=======


# elif self._action_time_s
>>>>>>> 43398dbaf1fa64257240a5d12d21c75b18235789
        else:
            TERMINALSTATE = False

        return REWARD, TERMINALSTATE, DAMAGE_redAC, {
            'result': INFO,
            'redObs': self._get_sta_env_v2_redAC()
        }

    def _calc_posDiff_hdg_rad(self, start: np.array, dest: np.array):

        posDiff = dest - start  # dest, start?
        angleDiff = np.arctan2(posDiff[1], posDiff[0])

        distance = np.linalg.norm(posDiff)

        return posDiff, distance, angleDiff  # anglediff = LOS_deg

    def _calc_posDiff_hdg_deg(self, start: np.array, dest: np.array):

        pos, dist, angle = self._calc_posDiff_hdg_rad(start, dest)

        return pos, dist, np.rad2deg(angle)

    def _distance(red, start: np.array, target: np.array):  

        return np.linalg.norm(target - start)

    def _pi_bound(self, u):

        if u > np.pi:
            y = u - 2 * np.pi
        elif u < -np.pi:
            y = u + 2 * np.pi
        else:
            y = u

        return y

    def _pi_bound_deg(self, u):

        if u > 180.:
            y = u - 360.
        elif u < -180.:
            y = u + 360.
        else:
            y = u

        return y
