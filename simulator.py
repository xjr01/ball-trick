import taichi as ti
import math

ti.init(arch=ti.cpu)

@ti.data_oriented
class Balls:
	def __init__(self, ball_row: int, ball_col: int, wall_row: int, wall_col: int):
		self.n_ball = ball_row * ball_col
		self.n_wall = wall_row * wall_col
		self.max_collision = 100000
		self.unit_length = 1.
		self.r_ball = self.unit_length / (ball_col * 2 + 1)
		self.r_wall = self.unit_length / (wall_col - .5) / 8
		self.stiffness = 1e5
		self.damping = 0.
		self.gravity = 1.
		self.mass = 1.
		
		self.min_corner = ti.Vector([0., 0.])
		self.max_corner = ti.Vector([self.unit_length, self.unit_length])

		self.positions = ti.Vector.field(2, dtype=float, shape=(self.n_ball,))
		self.velocities = ti.Vector.field(2, dtype=float, shape=(self.n_ball,))
		self.colors = ti.Vector.field(3, dtype=float, shape=(self.n_ball,))
		self.wall_pos = ti.Vector.field(2, dtype=float, shape=(self.n_wall,))
		
		self.xk = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.yk = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.rhs = ti.ndarray(dtype=float, shape=(2 * self.n_ball,))
		self.collision_pairs = ti.Vector.field(2, dtype=int, shape=(self.max_collision,))
		self.cnt_collision = ti.field(dtype=int, shape=())
		
		self.set_initial_condition(ball_row, ball_col, wall_row, wall_col)
	
	@ti.kernel
	def set_initial_condition(self, ball_row: int, ball_col: int, wall_row: int, wall_col: int):
		# Placing balls
		color_list = ti.Matrix([[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [160, 32, 240]], dt=float) / 255.
		for i in range(ball_row):
			for j in range(ball_col):
				self.positions[i * ball_col + j] = [self.min_corner[0] + self.r_ball * (1. if i % 2 == 0 else 2.) + j * 2. * self.r_ball, self.max_corner[1] - (self.r_ball + i * 3. ** .5 * self.r_ball)]
				self.velocities[i * ball_col + j] = [0., 0.]
				self.colors[i * ball_col + j] = color_list[i % len(color_list), :]
		# Placing walls
		spacing = self.unit_length / (wall_col - .5)
		for i in range(wall_row):
			for j in range(wall_col):
				self.wall_pos[i * wall_col + j] = [self.min_corner[0] + (0. if i % 2 == 0 else spacing * .5) + j * spacing, self.max_corner[1] - (self.r_ball + ball_row * 3. ** .5 * self.r_ball + self.r_wall * 2. + i * 3 ** .5 * .5 * spacing)]
	
	@ti.kernel
	def collision_detect(self, dt: float):
		# Apply external force
		for i in range(self.n_ball):
			self.velocities[i] += dt * [0., -self.gravity]
		# Collision detection
		self.cnt_collision[None] = 0
		# Collision between balls
		for i in range(self.n_ball):
			for j in range(i + 1, self.n_ball):
				if (self.positions[i] + dt * self.velocities[i] - self.positions[j] - dt * self.velocities[j]).norm() < self.r_ball * 2:
					self.collision_pairs[self.cnt_collision[None]] = [i, j]
					self.cnt_collision[None] += 1
		# Collision with walls
		for i in range(self.n_ball):
			for j in range(self.n_wall):
				if (self.positions[i] + dt * self.velocities[i] - self.wall_pos[j]).norm() < self.r_wall + self.r_ball:
					self.collision_pairs[self.cnt_collision[None]] = [i, j + self.n_ball]
					self.cnt_collision[None] += 1
			if (self.positions[i] + dt * self.velocities[i])[0] - self.r_ball < self.min_corner[0]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -1] # left boundary
				self.cnt_collision[None] += 1
			if (self.positions[i] + dt * self.velocities[i])[1] - self.r_ball < self.min_corner[1]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -2] # lower boundary
				self.cnt_collision[None] += 1
			if (self.positions[i] + dt * self.velocities[i])[0] + self.r_ball > self.max_corner[0]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -3] # right boundary
				self.cnt_collision[None] += 1
			if (self.positions[i] + dt * self.velocities[i])[1] + self.r_ball > self.max_corner[1]:
				self.collision_pairs[self.cnt_collision[None]] = [i, -4] # upper boundary
				self.cnt_collision[None] += 1
		
	@ti.kernel
	def calculate_vectors(self, dt: float, xk: ti.types.ndarray(), yk: ti.types.ndarray(), rhs: ti.types.ndarray()):
		for i in range(self.n_ball):
			xk[i * 2] = self.positions[i][0]
			xk[i * 2 + 1] = self.positions[i][1]
			y = self.positions[i] + dt * self.velocities[i]
			yk[i * 2] = y[0]
			yk[i * 2 + 1] = y[1]
			rhs[i * 2] = -self.mass * (xk[i * 2] - yk[i * 2])
			rhs[i * 2 + 1] = -self.mass * (xk[i * 2 + 1] - yk[i * 2 + 1])
		for _ in range(1):
			for i in range(self.cnt_collision[None]):
				a, b = self.collision_pairs[i]
				f = ti.Vector([0., 0.])
				if b == -1: # collision with left boundary
					f += self.stiffness * (self.positions[a][0] - self.min_corner[0] - self.r_ball) * [-1, 0]\
						- self.damping * self.velocities[a][0] * [1, 0]
				elif b == -2: # collision with lower boundary
					f += self.stiffness * (self.positions[a][1] - self.min_corner[1] - self.r_ball) * [0, -1]\
						- self.damping * self.velocities[a][1] * [0, 1]
				elif b == -3: # collision with right boundary
					f += self.stiffness * (self.max_corner[0] - self.positions[a][0] - self.r_ball) * [1, 0]\
						- self.damping * self.velocities[a][0] * [1, 0]
				elif b == -4: # collision with upper boundary
					f += self.stiffness * (self.max_corner[1] - self.positions[a][1] - self.r_ball) * [0, 1]\
						- self.damping * self.velocities[a][1] * [0, 1]
				elif b < self.n_ball: # collision with another ball
					delta_x = self.positions[b] - self.positions[a]
					f += self.stiffness * (delta_x.norm() - 2. * self.r_ball) * delta_x / delta_x.norm()\
						- self.damping * (self.velocities[a] - self.velocities[b]).dot(delta_x) * delta_x / delta_x.norm() ** 2
					rhs[b * 2] -= dt * dt * f[0]
					rhs[b * 2 + 1] -= dt * dt * f[1]
				else: # collision with wall
					delta_x = self.wall_pos[b - self.n_ball] - self.positions[a]
					f += self.stiffness * (delta_x.norm() - self.r_ball - self.r_wall) * delta_x / delta_x.norm()\
						- self.damping * (self.velocities[a].dot(delta_x)) * delta_x / delta_x.norm() ** 2
				rhs[a * 2] += dt * dt * f[0]
				rhs[a * 2 + 1] += dt * dt * f[1]
	
	@ti.kernel
	def build_matrix(self, dt: float, mat_builder: ti.types.sparse_matrix_builder()):
		# 1 / dt ** 2 * self.mass
		for _ in range(1):
			for i in range(2 * self.n_ball):
				mat_builder[i, i] += self.mass
		# Hessian of spring forces
		for _ in range(1):
			for i in range(self.cnt_collision[None]):
				a, b = self.collision_pairs[i]
				delta_x = ti.Vector([0., 0.])
				rest_length = 0.
				if b == -1: # collision with left boundary
					delta_x += [self.positions[a][0] - self.min_corner[0], 0.]
					rest_length += self.r_ball
				elif b == -2: # collision with lower boundary
					delta_x += [0., self.positions[a][1] - self.min_corner[1]]
					rest_length += self.r_ball
				elif b == -3: # collision with right boundary
					delta_x += [self.positions[a][0] - self.max_corner[0], 0.]
					rest_length += self.r_ball
				elif b == -4: # collision with upper boundary
					delta_x += [0., self.positions[a][1] - self.max_corner[1]]
					rest_length += self.r_ball
				elif b < self.n_ball: # collision with another ball
					delta_x += self.positions[a] - self.positions[b]
					rest_length += 2. * self.r_ball
				else: # collision with wall
					delta_x += self.positions[a] - self.wall_pos[b - self.n_ball]
					rest_length += self.r_ball + self.r_wall
				outer_prod = ti.Matrix([[delta_x[0]], [delta_x[1]]]) @ ti.Matrix([[delta_x[0], delta_x[1]]]) / delta_x.norm() ** 2
				H_e = dt * dt * (self.stiffness * outer_prod +\
					self.stiffness * (1. - rest_length / delta_x.norm()) * (ti.Matrix([[1., 0.], [0., 1.]]) - outer_prod))
				mat_builder[a * 2, a * 2] += H_e[0, 0]
				mat_builder[a * 2, a * 2 + 1] += H_e[0, 1]
				mat_builder[a * 2 + 1, a * 2] += H_e[1, 0]
				mat_builder[a * 2 + 1, a * 2 + 1] += H_e[1, 1]
				if 0 <= b < self.n_ball:
					mat_builder[b * 2, b * 2] += H_e[0, 0]
					mat_builder[b * 2, b * 2 + 1] += H_e[0, 1]
					mat_builder[b * 2 + 1, b * 2] += H_e[1, 0]
					mat_builder[b * 2 + 1, b * 2 + 1] += H_e[1, 1]
					mat_builder[a * 2, b * 2] -= H_e[0, 0]
					mat_builder[a * 2, b * 2 + 1] -= H_e[0, 1]
					mat_builder[a * 2 + 1, b * 2] -= H_e[1, 0]
					mat_builder[a * 2 + 1, b * 2 + 1] -= H_e[1, 1]
					mat_builder[b * 2, a * 2] -= H_e[0, 0]
					mat_builder[b * 2, a * 2 + 1] -= H_e[0, 1]
					mat_builder[b * 2 + 1, a * 2] -= H_e[1, 0]
					mat_builder[b * 2 + 1, a * 2 + 1] -= H_e[1, 1]
		# Damping
		for _ in range(1):
			for i in range(self.cnt_collision[None]):
				a, b = self.collision_pairs[i]
				delta_x = ti.Vector([0., 0.])
				if b == -1: # collision with left boundary
					delta_x += [self.positions[a][0] - self.min_corner[0], 0.]
				elif b == -2: # collision with lower boundary
					delta_x += [0., self.positions[a][1] - self.min_corner[1]]
				elif b == -3: # collision with right boundary
					delta_x += [self.positions[a][0] - self.max_corner[0], 0.]
				elif b == -4: # collision with upper boundary
					delta_x += [0., self.positions[a][1] - self.max_corner[1]]
				elif b < self.n_ball: # collision with another ball
					delta_x += self.positions[a] - self.positions[b]
				else: # collision with wall
					delta_x += self.positions[a] - self.wall_pos[b - self.n_ball]
				delta_v = ti.Vector([0., 0.])
				if 0 <= b < self.n_ball:
					delta_v += self.velocities[a] - self.velocities[b]
				else:
					delta_v += self.velocities[a]
				delta_n = delta_x / delta_x.norm()
				mat_n = ti.Matrix([[delta_n[0]], [delta_n[1]]])
				H_d = dt * dt * self.damping / delta_x.norm() * (mat_n @ ti.Matrix([[delta_v[0], delta_v[1]]]) + delta_v.dot(delta_n) * (ti.Matrix([[1., 0.], [0., 1.]]) - 2. * mat_n @ mat_n.transpose()))
				mat_builder[a * 2, a * 2] += H_d[0, 0]
				mat_builder[a * 2, a * 2 + 1] += H_d[0, 1]
				mat_builder[a * 2 + 1, a * 2] += H_d[1, 0]
				mat_builder[a * 2 + 1, a * 2 + 1] += H_d[1, 1]
				if 0 <= b < self.n_ball:
					mat_builder[b * 2, b * 2] += H_d[0, 0]
					mat_builder[b * 2, b * 2 + 1] += H_d[0, 1]
					mat_builder[b * 2 + 1, b * 2] += H_d[1, 0]
					mat_builder[b * 2 + 1, b * 2 + 1] += H_d[1, 1]
					mat_builder[a * 2, b * 2] -= H_d[0, 0]
					mat_builder[a * 2, b * 2 + 1] -= H_d[0, 1]
					mat_builder[a * 2 + 1, b * 2] -= H_d[1, 0]
					mat_builder[a * 2 + 1, b * 2 + 1] -= H_d[1, 1]
					mat_builder[b * 2, a * 2] -= H_d[0, 0]
					mat_builder[b * 2, a * 2 + 1] -= H_d[0, 1]
					mat_builder[b * 2 + 1, a * 2] -= H_d[1, 0]
					mat_builder[b * 2 + 1, a * 2 + 1] -= H_d[1, 1]
	
	@ti.kernel
	def update_states(self, dt: float, delta_pos: ti.types.ndarray()):
		for i in range(self.n_ball):
			delta_i = ti.Vector([delta_pos[i * 2], delta_pos[i * 2 + 1]])
			self.positions[i] += delta_i
			self.velocities[i] = delta_i / dt
	
	def advance(self, dt: float):
		self.collision_detect(dt)
		self.calculate_vectors(dt, self.xk, self.yk, self.rhs)
		mat_builder = ti.linalg.SparseMatrixBuilder(2 * self.n_ball, 2 * self.n_ball, max_num_triplets=32 * self.max_collision + 2 * self.n_ball)
		self.build_matrix(dt, mat_builder)
		mat = mat_builder.build()
		solver = ti.linalg.SparseSolver(solver_type='LU')
		solver.analyze_pattern(mat)
		solver.factorize(mat)
		delta_pos = solver.solve(self.rhs)
		if not solver.info():
			print('Warning: Failed to solve the linear system.')
			with open('output.txt', 'w') as fd:
				fd.write('\n'.join([' '.join([str(mat[i, j]) for j in range(2 * self.n_ball)]) for i in range(2 * self.n_ball)]))
				exit()
			for i in range(2 * self.n_ball):
				if delta_pos[i] != 0.:
					print('Not all zero!!!')
				if math.isnan(delta_pos[i]):
					print('Nan in solution!!!')
					break
			for i in range(2 * self.n_ball):
				if math.isnan(self.rhs[i]):
					print('Nan in rhs!!!')
					break
			for i in range(2 * self.n_ball):
				for j in range(2 * self.n_ball):
					if math.isnan(mat[i, j]):
						print('Nan in mat!!!')
						break
		self.update_states(dt, delta_pos)
	
	@ti.kernel
	def get_max_time_step(self) -> float:
		max_v = 1e-3
		for i in range(self.n_ball):
			max_v = max(max_v, self.velocities[i].norm())
		return min(self.r_ball, self.r_wall) * .1 / max_v

balls = Balls(1, 30, 1, 6)
print(balls.n_ball, balls.n_wall, balls.n_ball ** 2 + balls.n_ball * (balls.n_wall + 4))
fps_limit = 1000
time_step = 1 / fps_limit

win_size = 700
window = ti.ui.Window(name='Ball trick', fps_limit=fps_limit, res=(win_size, win_size))
canvas = window.get_canvas()

while window.running:
	canvas.set_background_color((0.067, 0.184, 0.255))
	canvas.circles(balls.positions, balls.r_ball, per_vertex_color=balls.colors)
	canvas.circles(balls.wall_pos, balls.r_wall)
	window.show()
	rest_t = time_step
	while rest_t > 0.:
		dt = balls.get_max_time_step()
		if dt >= rest_t:
			balls.advance(rest_t)
			rest_t = 0.
		else:
			balls.advance(dt)
			rest_t -= dt
	# balls.advance(min(balls.get_max_time_step(), time_step))