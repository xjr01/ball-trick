import taichi as ti
import numpy as np
import cv2

ti.init(ti.cpu)

win_size = 700
window = ti.ui.Window(name='Ball trick', fps_limit=30, res=(win_size, win_size), show_window=False)
canvas = window.get_canvas()

@ti.data_oriented
class ImageExtractor:
	def __init__(self, filename=None, id=None):
		data = np.load(filename if filename is not None else f'output/states_{id}.npz')
		self.positions = data['positions']
		self.r_ball = data['r_ball']
		self.colors = data['colors']
		self.wall_pos = data['wall_pos']
		self.r_wall = data['r_wall']
		
		self.n_ball = self.positions.shape[0]
		self.n_wall = self.wall_pos.shape[0]

		self.ti_positions = ti.Vector.field(2, dtype=ti.f64, shape=(self.positions.shape[0],))
		self.ti_colors = ti.Vector.field(3, dtype=ti.f64, shape=(self.colors.shape[0],))
		self.ti_wall_pos = ti.Vector.field(2, dtype=ti.f64, shape=(self.wall_pos.shape[0],))

	@ti.kernel
	def convert_to_fields(self, positions: ti.types.ndarray(dtype=ti.f64), colors: ti.types.ndarray(dtype=ti.f64), wall_pos: ti.types.ndarray(dtype=ti.f64)):
		for i in range(positions.shape[0]):
			self.ti_positions[i][0] = positions[i, 0]
			self.ti_positions[i][1] = positions[i, 1]
		for i in range(colors.shape[0]):
			self.ti_colors[i][0] = colors[i, 0]
			self.ti_colors[i][1] = colors[i, 1]
			self.ti_colors[i][2] = colors[i, 2]
		for i in range(wall_pos.shape[0]):
			self.ti_wall_pos[i][0] = wall_pos[i, 0]
			self.ti_wall_pos[i][1] = wall_pos[i, 1]
	
	def get_image_as_numpy(self):
		self.convert_to_fields(self.positions, self.colors, self.wall_pos)

		canvas.set_background_color((0.067, 0.184, 0.255))
		canvas.circles(self.ti_positions, self.r_ball.item(), per_vertex_color=self.ti_colors)
		canvas.circles(self.ti_wall_pos, self.r_wall.item())
		img = window.get_image_buffer_as_numpy()

		img = np.round(img[:, ::-1, 2::-1] * 255).astype(np.uint8).transpose(1, 0, 2)
		return img


last_frame = ImageExtractor(id=8200)
# Get final color
color_list = np.array([[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [160, 32, 240]], dtype=np.float64) / 255.
final_colors = np.zeros_like(last_frame.colors)
for i in range(last_frame.n_ball):
	final_colors[i] = color_list[int((last_frame.positions[i][1] - last_frame.r_ball * .5) // (np.sqrt(3.) * last_frame.r_ball)) % len(color_list)]