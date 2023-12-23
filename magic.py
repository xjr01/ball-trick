import taichi as ti
import numpy as np
import cv2

ti.init(ti.cpu)

win_size = 700
window = ti.ui.Window(name='Ball trick', fps_limit=30, res=(win_size, win_size), show_window=False)
canvas = window.get_canvas()

@ti.data_oriented
class ImageExtractor:
	def __init__(self, id=None, filename=None):
		data = np.load(f'output/states_{id}.npz' if id is not None else filename)
		self.positions = data['positions']
		self.r_ball = data['r_ball']
		self.colors = data['colors']
		self.wall_pos = data['wall_pos']
		self.r_wall = data['r_wall']
		
		self.n_ball = self.positions.shape[0]
		self.n_wall = self.wall_pos.shape[0]

	@ti.kernel
	def convert_to_fields(self, positions: ti.types.ndarray(dtype=ti.f64), colors: ti.types.ndarray(dtype=ti.f64), wall_pos: ti.types.ndarray(dtype=ti.f64)):
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
	
	def get_image_as_numpy(self):
		self.convert_to_fields(self.positions, self.colors, self.wall_pos)

		canvas.set_background_color((0.067, 0.184, 0.255))
		canvas.circles(ti_positions, self.r_ball.item(), per_vertex_color=ti_colors)
		canvas.circles(ti_wall_pos, self.r_wall.item())
		img = window.get_image_buffer_as_numpy()

		img = np.round(img[:, ::-1, 2::-1] * 255).astype(np.uint8).transpose(1, 0, 2)
		return img


N_frame = 8200

last_frame = ImageExtractor(N_frame - 1)
ti_positions = ti.Vector.field(2, dtype=ti.f64, shape=(last_frame.positions.shape[0],))
ti_colors = ti.Vector.field(3, dtype=ti.f64, shape=(last_frame.colors.shape[0],))
ti_wall_pos = ti.Vector.field(2, dtype=ti.f64, shape=(last_frame.wall_pos.shape[0],))
last_img = last_frame.get_image_as_numpy()
# Get final color
color_list = np.array([[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [160, 32, 240]], dtype=np.float64) / 255.
final_colors = np.zeros_like(last_frame.colors)
for i in range(last_frame.n_ball):
	final_colors[i] = color_list[int((last_frame.positions[i][1] - last_frame.r_ball * .5) // (np.sqrt(3.) * last_frame.r_ball)) % len(color_list)]

# Let the magic begins!
in_fps, out_fps = 1000, 50
n_pics = in_fps // out_fps

needs_change = np.abs(last_frame.colors - final_colors).sum(axis=1).astype(bool)

def simulated_annealing():
	t_start, t_end, t_rate = 1e4, 1e-5, .99
	def n_iter(t: float) -> int:
		return 1 + int(2 // t)
	
	def evaluate(change_frame: np.ndarray) -> float:
		def evaluate_once(original_ext, extractors) -> tuple[int, int]:
			if not extractors:
				return (0, 0)
			original_img = np.zeros_like(last_img, dtype=np.int64)
			img = np.zeros_like(last_img, dtype=np.int64)
			for o, e in zip(original_ext, extractors):
				original_img += o.get_image_as_numpy()
				img += e.get_image_as_numpy()
			original_img //= len(extractors)
			img //= len(extractors)
			diff = np.abs(original_img - img)
			return (diff.sum(), diff.astype(bool).sum())
		
		ball_id = np.argsort(change_frame)
		original_ext, extractors = [], []
		really_changed = False
		err_sum, err_cnt = 0, 0
		for i in range(last_frame.n_ball):
			ths_bin, lst_bin = change_frame[ball_id[i]] // n_pics, change_frame[ball_id[i - 1]] // n_pics
			if not i or ths_bin != lst_bin:
				if really_changed:
					ths_sum, ths_cnt = evaluate_once(original_ext, extractors)
					err_sum += ths_sum
					err_cnt += ths_cnt
				original_ext = [ImageExtractor(j) for j in range(ths_bin * n_pics, (ths_bin + 1) * n_pics)]
				extractors = [ImageExtractor(j) for j in range(ths_bin * n_pics, (ths_bin + 1) * n_pics)]
				really_changed = False
			if needs_change[ball_id[i]]:
				really_changed = True
				for j in range(change_frame[ball_id[i]] - ths_bin * n_pics, n_pics):
					extractors[j].colors[ball_id[i]] = final_colors[ball_id[i]]
		if really_changed:
			ths_sum, ths_cnt = evaluate_once(original_ext, extractors)
			err_sum += ths_sum
			err_cnt += ths_cnt
		
		assert err_cnt != 0
		return err_sum / err_cnt
	
	best_change_frame = change_frame = np.random.randint(0, N_frame, last_frame.n_ball)
	min_cost = cost = evaluate(change_frame)
	fd = open('output_log.txt', 'w')
	fd.write(f'cost: {cost}\n')
	fd.close()
	
	t = t_start
	while t >= t_end:
		n = n_iter(t)
		for iter in range(n):
			# Get neighboring solution
			new_change_frame = (change_frame + np.random.randint(-100, 101, last_frame.n_ball)).clip(0, N_frame - 1)
			new_cost = evaluate(new_change_frame)
			fd.write(f'cost: {new_cost}\n')
			if new_cost < min_cost:
				best_change_frame = new_change_frame
				min_cost = new_cost
			delta = new_cost - cost
			if delta > 0 or np.random.rand() < np.exp(delta / t):
				change_frame = new_change_frame
				cost = new_cost
		t *= t_rate
	
	fd.close()
	return best_change_frame, min_cost

# Generate video
change_frame = simulated_annealing()[0]
ball_id = np.argsort(change_frame)
cur_ball = 0
colors = last_frame.colors
fourcc = cv2.VideoWriter_fourcc(*'X264')
video_out = cv2.VideoWriter('output/magic.mp4', fourcc, 30, (win_size, win_size))
blurred = np.zeros_like(last_img, dtype=np.int64)
for i in range(N_frame):
	while change_frame[ball_id[cur_ball]] == i:
		colors[ball_id[cur_ball]] = final_colors[ball_id[cur_ball]]
		cur_ball += 1
	e = ImageExtractor(i)
	e.colors = colors
	blurred += e.get_image_as_numpy()
	if i % n_pics == n_pics - 1:
		video_out.write((blurred // n_pics).astype(np.uint8))
		blurred = np.zeros_like(last_img, dtype=np.int64)
video_out.release()