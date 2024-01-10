import numpy as np
import cv2
from magic import ImageExtractor, rainbow, letters_IOMA

N_frame = 8200
win_size = 700

last_frame = ImageExtractor(N_frame - 1, for_optimize=False)
last_img = last_frame.get_image_as_numpy()
# Get final color
color_list = np.array([[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [160, 32, 240]], dtype=np.uint8)
with open('settings.json', 'r') as fd:
	final_colors = eval(eval(fd.read())['case'] + '()')

data = np.load('output/change_frame.npz')
change_frame = data['change_frame']
cost = data['cost'].item()
ball_id = np.argsort(change_frame)
print('cost:', cost)

in_fps, out_fps = 2000, 50
n_pics = in_fps // out_fps

cnt_frame = 0
cur_ball = 0
colors = last_frame.colors
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('output/magic.mp4', fourcc, out_fps, (win_size, win_size))
blurred = np.zeros_like(last_img, dtype=np.int32)
for i in range(N_frame):
	while cur_ball < ball_id.shape[0] and change_frame[ball_id[cur_ball]] == i:
		colors[ball_id[cur_ball]] = final_colors[ball_id[cur_ball]]
		cur_ball += 1
	e = ImageExtractor(i, for_optimize=False)
	e.colors = colors
	blurred += e.get_image_as_numpy()
	if i % n_pics == n_pics - 1:
		video_out.write((blurred // n_pics).astype(np.uint8))
		print(f'Frame {cnt_frame} finished.')
		cnt_frame += 1
		blurred = np.zeros_like(last_img, dtype=np.int32)
video_out.release()