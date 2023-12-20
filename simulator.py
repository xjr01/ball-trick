import taichi as ti

ti.init(arch=ti.cpu)

@ti.data_oriented
class Balls:
	def __init__(self):
		self.n_ball = 100
		self.n_wall = 100
		self.max_collision = 100000
		self.unit_length = 1.
		self.r_ball = self.unit_length / 620
		self.r_wall = 1.5 * self.r_ball
		self.stiffness = 1e3
		self.damping = 1.
		self.gravity = 9.8
		self.mass = 1.
		
		self.min_corner = ti.Vector([0., 0.])
		self.max_corner = ti.Vector([self.unit_length, self.unit_length])

		self.positions = ti.Vector.field(2, dtype=float, shape=(self.n_ball,))
		self.velocities = ti.Vector.field(2, dtype=float, shape=(self.n_ball,))
		self.colors = ti.Vector.field(3, dtype=ti.uint8, shape=(self.n_ball,))
		self.wall_pos = ti.Vector.field(2, dtype=float, shape=(self.n_wall,))
		
		self.xk = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.yk = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.rhs = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.collision_pairs = ti.Vector.field(2, dtype=int, shape=(self.max_collision,))
		self.cnt_collision = ti.field(dtype=int, shape=())
	
	@ti.kernel
	def get_collision_pairs(self):
		# Collision detection
		self.cnt_collision[None] = 0
		# Collision between balls
		for i in range(self.n_ball):
			for j in range(i + 1, self.n_ball):
				if (self.positions[i] - self.positions[j]).norm() < self.r_ball * 2:
					self.collision_pairs[self.cnt_collision[None]] = [i, j]
					self.cnt_collision[None] += 1
		# Collision with walls
		for i in range(self.n_ball):
			for j in range(self.n_wall):
				if (self.positions[i] - self.wall_pos[j]).norm() < self.r_wall + self.r_ball:
					self.collision_pairs[self.cnt_collision[None]] = [i, j + self.n_ball]
					self.cnt_collision[None] += 1
			if self.positions[i][0] < self.min_corner[0]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -1] # left boundary
				self.cnt_collision[None] += 1
			if self.positions[i][1] < self.min_corner[1]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -2] # lower boundary
				self.cnt_collision[None] += 1
			if self.positions[i][0] > self.max_corner[0]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -3] # right boundary
				self.cnt_collision[None] += 1
			if self.positions[i][1] > self.max_corner[1]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -4] # upper boundary
				self.cnt_collision[None] += 1
		
	@ti.kernel
	def calculate_vectors(self, dt: float, xk: ti.types.ndarray(), yk: ti.types.ndarray(), rhs: ti.types.ndarray()):
		for i in range(self.n_ball):
			xk[i * 2] = self.positions[i][0]
			xk[i * 2 + 1] = self.positions[i][1]
			y = self.positions[i] + dt * self.velocities[i] + dt * dt * [0, -self.gravity]
			yk[i * 2] = y[0]
			yk[i * 2 + 1] = y[1]
			rhs[i * 2] = -1. / dt / dt * self.mass * (xk[i * 2] - yk[i * 2])
			rhs[i * 2 + 1] = -1. / dt / dt * self.mass * (xk[i * 2 + 1] - yk[i * 2 + 1])
		for i in range(self.cnt_collision[None]):
			a, b = self.collision_pairs[i]
			f = ti.Vector([0., 0.])
			if b == -1: # collision with left boundary
				f += self.stiffness * (self.positions[a][0] - self.min_corner[0] - self.r_ball) * [-1, 0]
			elif b == -2: # collision with lower boundary
				f += self.stiffness * (self.positions[a][1] - self.min_corner[1] - self.r_ball) * [0, -1]
			elif b == -3: # collision with right boundary
				f += self.stiffness * (self.max_corner[0] - self.positions[a][0] - self.r_ball) * [1, 0]
			elif b == -4: # collision with upper boundary
				f += self.stiffness * (self.max_corner[1] - self.positions[a][1] - self.r_ball) * [0, 1]
			elif b < self.n_ball: # collision with another ball
				delta_x = self.positions[b] - self.positions[a]
				f += self.stiffness * (delta_x.norm() - 2. * self.r_ball) * delta_x / delta_x.norm()
				rhs[b * 2] -= f[0]
				rhs[b * 2 + 1] -= f[1]
			else: # collision with wall
				delta_x = self.wall_pos[b - self.n_ball] - self.positions[a]
				f += self.stiffness * (delta_x.norm() - self.r_ball - self.r_wall) * delta_x / delta_x.norm()
			rhs[a * 2] += f[0]
			rhs[a * 2 + 1] += f[1]
	
	@ti.kernel
	def build_matrix(self, dt: float, mat_builder: ti.types.sparse_matrix_builder()):
		for i in range(2 * self.n_ball):
			mat_builder[i, i] += self.mass / dt / dt
		for i in range(self.cnt_collision[None]):
			pass
	
	def advance(self, dt: float):
		self.get_collision_pairs()
		self.calculate_vectors(dt, self.xk, self.yk, self.rhs)
		mat_builder = ti.linalg.SparseMatrixBuilder(2 * self.n_ball, 2 * self.n_ball)
		self.build_matrix(dt, mat_builder)

balls = Balls()
balls.advance(1.)