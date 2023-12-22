import taichi as ti
import numpy as np

ti.init(ti.cpu)

data = np.load('output/states_324.npz')
positions = data['positions']
r_ball = data['r_ball']
colors = data['colors']
wall_pos = data['wall_pos']
r_wall = data['r_wall']

ti_positions = ti.Vector.field(2, dtype=ti.f64, shape=(positions.shape[0],))
ti_colors = ti.Vector.field(3, dtype=ti.f64, shape=(colors.shape[0],))
ti_wall_pos = ti.Vector.field(2, dtype=ti.f64, shape=(wall_pos.shape[0],))

@ti.kernel
def convert_to_fields(positions: ti.types.ndarray(), colors: ti.types.ndarray(), wall_pos: ti.types.ndarray()):
	for i in range(positions.shape[0]):
		ti_positions[i][0] = positions[i, 0]
		ti_positions[i][1] = positions[i, 1]
	for i in range(colors.shape[0]):
		ti_colors[i][0] = colors[i, 0]
		ti_colors[i][1] = colors[i, 1]
		ti_colors[i][2] = colors[i, 2]
	for i in range(wall_pos.shape[0]):
		ti_wall_pos[i][0] = wall_pos[i, 0]
		ti_wall_pos[i][1] = wall_pos[i, 1]

print(positions.shape)
convert_to_fields(positions, colors, wall_pos)

win_size = 700
window = ti.ui.Window(name='Ball trick', fps_limit=30, res=(win_size, win_size))
canvas = window.get_canvas()

while True:
	canvas.set_background_color((0.067, 0.184, 0.255))
	canvas.circles(ti_positions, r_ball.item(), per_vertex_color=ti_colors)
	canvas.circles(ti_wall_pos, r_wall.item())
	window.show()