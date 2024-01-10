import numpy as np
import cv2
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

win_size = 700

class ImageExtractor:
	def __init__(self, id=None, filename=None, for_optimize=True):
		data = np.load(f'output/states_{id}.npz' if id is not None else filename)
		self.positions = np.round(data['positions'] * win_size).astype(int)
		self.r_ball = int(round(data['r_ball'].item() * win_size)) - (1 if for_optimize else 0)
		self.colors = np.round(data['colors'] * 255).astype(np.uint8)
		self.wall_pos = np.round(data['wall_pos'] * win_size).astype(int)
		self.r_wall = int(round(data['r_wall'].item() * win_size)) - (1 if for_optimize else 0)
		
		self.n_ball = self.positions.shape[0]
		self.n_wall = self.wall_pos.shape[0]
		
		self.bkg_img = np.array((0.067, 0.184, 0.255)) * 255 + np.zeros((win_size, win_size, 3))
		self.bkg_img = np.round(self.bkg_img).astype(np.uint8)

	def get_image_as_numpy(self):
		img = self.bkg_img.copy()
		for p, c in zip(self.positions, self.colors):
			cv2.circle(img, tuple(p), self.r_ball, (c[0].item(), c[1].item(), c[2].item()), -1)
		for p in self.wall_pos:
			cv2.circle(img, tuple(p), self.r_wall, (128, 128, 128), -1)
		img = img[::-1, :, 2::-1]
		return img


N_frame = 8200

last_frame = ImageExtractor(N_frame - 1)
last_img = last_frame.get_image_as_numpy()
# Get final color
color_list = np.array([[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [160, 32, 240]], dtype=np.uint8)

def rainbow():
	final_colors = np.zeros_like(last_frame.colors)
	for i in range(last_frame.n_ball):
		final_colors[i] = color_list[int((last_frame.positions[i][1] - (last_frame.r_ball + 1) * .5) // (np.sqrt(3.) * (last_frame.r_ball + 1))) % len(color_list)]
	return final_colors

def letters_IOMA():
	final_colors = last_frame.colors.copy()
	final_colors[200:250] = color_list[-3] # blue -> green
	blue_poses = np.array([
		[11, 1],
		[12, 2],
		[13, 3],
		[14, 4],
		[15, 5],
		[16, 6],
		[17, 7],
		[18, 6],
		[19, 5],
		[20, 4],
		[21, 3],
		[22, 2],
		[23, 1],
		[15, 3],
		[17, 3],
		[19, 3], # A
		[27, 1],
		[28, 2],
		[29, 3],
		[30, 4],
		[31, 5],
		[32, 6],
		[33, 7],
		[34, 6],
		[35, 5],
		[36, 4],
		[37, 5],
		[38, 6],
		[39, 7],
		[40, 6],
		[41, 5],
		[42, 4],
		[43, 3],
		[44, 2],
		[45, 1], # M
		[48, 4],
		[49, 5],
		[50, 6],
		[51, 7],
		[53, 7],
		[55, 7],
		[57, 7],
		[58, 6],
		[59, 5],
		[60, 4],
		[59, 3],
		[58, 2],
		[57, 1],
		[55, 1],
		[53, 1],
		[51, 1],
		[50, 2],
		[49, 3], # O
		[65, 1],
		[65, 3],
		[65, 5],
		[65, 7],
		[66, 2],
		[66, 4],
		[66, 6],
		[67, 1],
		[67, 3],
		[67, 5],
		[67, 7]  # I
	], dtype=float)
	blue_poses[:, 0] += 12
	blue_poses[:, 0] = win_size - blue_poses[:, 0] * (last_frame.r_ball + 1)
	blue_poses[:, 1] = (1. + blue_poses[:, 1] * np.sqrt(3.)) * (last_frame.r_ball + 1)
	for i in range(last_frame.n_ball):
		if (((last_frame.positions[i] - blue_poses) ** 2).sum(axis=1) < last_frame.r_ball ** 2.).any():
			final_colors[i] = color_list[-2] # blue
	return final_colors

with open('settings.json', 'r') as fd:
	final_colors = eval(eval(fd.read())['case'] + '()')

# Let the magic begin!
in_fps, out_fps = 1000, 50
n_pics = in_fps // out_fps

needs_change = np.abs(last_frame.colors - final_colors).sum(axis=1).astype(bool)

def evaluate_once(ball_id, frame_id) -> float:
	if not needs_change[ball_id]:
		return 0.
	
	extractors = [ImageExtractor(j) for j in range(frame_id, frame_id + n_pics)]
	original_ext = deepcopy(extractors)
	for o in original_ext:
		o.colors[ball_id] = final_colors[ball_id]
	
	original_img = np.zeros_like(last_img, dtype=np.int32)
	img = np.zeros_like(last_img, dtype=np.int32)
	for o, e in zip(original_ext, extractors):
		original_img += o.get_image_as_numpy()
		img += e.get_image_as_numpy()
	original_img = cv2.cvtColor((original_img // len(extractors)).astype(np.uint8), cv2.COLOR_RGB2Lab)
	img = cv2.cvtColor((img // len(extractors)).astype(np.uint8), cv2.COLOR_RGB2Lab)
	diff = ((original_img.astype(float) - img.astype(float)) ** 2).sum(axis=2) ** .5
	return diff.sum() / diff.astype(bool).sum()
	
def evaluate(change_frame: np.ndarray) -> float:
	err_sum = 0.
	with ProcessPoolExecutor(max_workers=55) as executor:
		futures = []
		for i, frame in enumerate(change_frame):
			futures.append(executor.submit(evaluate_once, i, frame))
		for f in futures:
			err_sum += f.result()
	
	return err_sum

def simulated_annealing():
	t_start, t_end, t_rate = 1e4, 1e-3, .9
	def n_iter(t: float) -> int:
		return 1 + int(2 // t)
	
	best_change_frame = change_frame = np.random.randint((N_frame - n_pics) // 7, (N_frame - n_pics) // 3, last_frame.n_ball)
	min_cost = cost = evaluate(change_frame)
	np.savez('output/change_frame.npz', change_frame=best_change_frame, cost=np.array(min_cost))
	with open('output/cost.txt', 'w') as fd:
		fd.write(f'{cost}\n')
	print('cost:', cost)
	
	t = t_start
	while t >= t_end:
		n = n_iter(t)
		for iter in range(n):
			# Get neighboring solution
			new_change_frame = (change_frame + np.random.randint(-10, 11, last_frame.n_ball) * (np.random.rand(last_frame.n_ball) < .2)) % (N_frame - n_pics)
			new_cost = evaluate(new_change_frame)
			if new_cost < min_cost:
				best_change_frame = new_change_frame
				min_cost = new_cost
				np.savez('output/change_frame.npz', change_frame=best_change_frame, cost=np.array(min_cost))
			delta = new_cost - cost
			if delta < 0 or np.random.rand() < np.exp(-delta / t):
				change_frame = new_change_frame
				cost = new_cost
				with open('output/cost.txt', 'a') as fd:
					fd.write(f'{new_cost}\n')
				print('cost:', new_cost)
		t *= t_rate
	
	return best_change_frame, min_cost

if __name__ == '__main__':
	print(simulated_annealing())