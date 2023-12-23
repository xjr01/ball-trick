import numpy as np
import cv2
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

win_size = 700

class ImageExtractor:
	def __init__(self, id=None, filename=None):
		data = np.load(f'output/states_{id}.npz' if id is not None else filename)
		self.positions = np.round(data['positions'] * win_size).astype(int)
		self.r_ball = int(round(data['r_ball'].item() * win_size))
		self.colors = np.round(data['colors'] * 255).astype(np.uint8)
		self.wall_pos = np.round(data['wall_pos'] * win_size).astype(int)
		self.r_wall = int(round(data['r_wall'].item() * win_size))
		
		self.n_ball = self.positions.shape[0]
		self.n_wall = self.wall_pos.shape[0]
		
		self.bkg_img = np.array((0.067, 0.184, 0.255)) * 255 + np.zeros((win_size, win_size, 3))
		self.bkg_img = np.round(self.bkg_img).astype(np.uint8)

	def get_image_as_numpy(self):
		img = self.bkg_img.copy()
		for p, c in zip(self.positions, self.colors):
			cv2.circle(img, p, self.r_ball, (c[0].item(), c[1].item(), c[2].item()), -1)
		for p in self.wall_pos:
			cv2.circle(img, p, self.r_wall, (128, 128, 128), -1)
		img = img[::-1, :, 2::-1]
		return img


N_frame = 8200

last_frame = ImageExtractor(N_frame - 1)
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

def evaluate_once(bin_id, events) -> tuple[int, int]:
	extractors = [ImageExtractor(j) for j in range(bin_id * n_pics, (bin_id + 1) * n_pics)]
	original_ext = deepcopy(extractors)
	for event in events:
		for j in range(event[1], n_pics):
			extractors[j].colors[event[0]] = final_colors[event[0]]
	
	original_img = np.zeros_like(last_img, dtype=np.int32)
	img = np.zeros_like(last_img, dtype=np.int32)
	for o, e in zip(original_ext, extractors):
		original_img += o.get_image_as_numpy()
		img += e.get_image_as_numpy()
	original_img //= len(extractors)
	img //= len(extractors)
	diff = np.abs(original_img - img)
	return (diff.sum(), diff.astype(bool).sum())
	
def evaluate(change_frame: np.ndarray) -> float:
	events = dict()
	for i in range(last_frame.n_ball):
		bin_id = change_frame[i] // n_pics
		if not (bin_id in events):
			events[bin_id] = []
		events[bin_id].append((i, change_frame[i] - bin_id * n_pics))
	
	err_sum, err_cnt = 0, 0
	with ProcessPoolExecutor(max_workers=55) as executor:
		futures = []
		for i in events.items():
			futures.append(executor.submit(evaluate_once, i[0], i[1]))
			# ths_sum, ths_cnt = evaluate_once(*i)
			# err_sum += ths_sum
			# err_cnt += ths_cnt
		for f in futures:
			ths_sum, ths_cnt = f.result()
			err_sum += ths_sum
			err_cnt += ths_cnt
	
	assert err_cnt != 0
	return err_sum / err_cnt

def simulated_annealing():
	t_start, t_end, t_rate = 1e4, 1e-3, .9
	def n_iter(t: float) -> int:
		return 1 + int(2 // t)
	
	best_change_frame = change_frame = np.random.randint(0, N_frame, last_frame.n_ball)
	min_cost = cost = evaluate(change_frame)
	fd = open('output/cost.txt', 'w')
	fd.write(f'{cost}\n')
	print('cost:', cost)
	
	t = t_start
	while t >= t_end:
		n = n_iter(t)
		for iter in range(n):
			# Get neighboring solution
			new_change_frame = (change_frame + np.random.randint(-100, 101, last_frame.n_ball)).clip(0, N_frame - 1)
			new_cost = evaluate(new_change_frame)
			fd.write(f'{new_cost}\n')
			print('cost:', new_cost)
			if new_cost < min_cost:
				best_change_frame = new_change_frame
				min_cost = new_cost
				np.savez('output/change_frame.npz', change_frame=best_change_frame, cost=np.array(min_cost))
			delta = new_cost - cost
			if delta < 0 or np.random.rand() < np.exp(-delta / t):
				change_frame = new_change_frame
				cost = new_cost
		t *= t_rate
	
	fd.close()
	return best_change_frame, min_cost

if __name__ == '__main__':
	# Generate video
	change_frame = simulated_annealing()[0]
	ball_id = np.argsort(change_frame)
	cur_ball = 0
	colors = last_frame.colors
	fourcc = cv2.VideoWriter_fourcc(*'X264')
	video_out = cv2.VideoWriter('output/magic.mp4', fourcc, 30, (win_size, win_size))
	blurred = np.zeros_like(last_img, dtype=np.int32)
	for i in range(N_frame):
		while change_frame[ball_id[cur_ball]] == i:
			colors[ball_id[cur_ball]] = final_colors[ball_id[cur_ball]]
			cur_ball += 1
		e = ImageExtractor(i)
		e.colors = colors
		blurred += e.get_image_as_numpy()
		if i % n_pics == n_pics - 1:
			video_out.write((blurred // n_pics).astype(np.uint8))
			blurred = np.zeros_like(last_img, dtype=np.int32)
	video_out.release()