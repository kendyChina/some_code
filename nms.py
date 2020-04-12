import torch


def intersection(box_a, box_b):
	A = box_a.size(0)
	B = box_b.size(0)
	# bottom right
	# A, B, (x2, y2)
	br_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand((A, B, 2)),
					  box_b[:, 2:4].unsqueeze(0).expand((A, B, 2)))
	# top left
	# A, B, (x1, y1)
	tl_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand((A, B, 2)),
					  box_b[:, 0:2].unsqueeze(0).expand((A, B, 2)))
	# A, B, (w, h)
	inter =torch.clamp((br_xy - tl_xy), min=0)
	# return shape(A, B)
	# w * h
	return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
	# box_a: x1, y1, x2, y2
	# box_b: x1, y1, x2, y2
	inter = intersection(box_a, box_b)
	# print(inter)

	area_a = ((box_a[:, 2] - box_a[:, 0]) *
			  (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
	area_b = ((box_b[:, 2] - box_b[:, 0]) *
			  (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
	union = area_a + area_b - inter
	return inter / union


def NMS(lists, threshold):
	# lists[0: 4]: x1, y1, x2, y2
	# lists[4]: score
	# threshold:

	# overlaps.shape: [lists.shape[0], lists.shape[0]]
	overlaps = jaccard(lists, lists)
	# print(overlaps)

	res_idxs = []
	_, idxs = lists[:, 4].sort(0, descending=True)
	for idx in idxs:
		tag = True
		for r_idx in res_idxs:
			if overlaps[idx, r_idx] > threshold:
				tag = False
				break
		if tag:
			res_idxs.append(idx.item())
	return lists[res_idxs]


if __name__ == '__main__':
	lists = []
	threshold = 0.8
	lists.append([1, 1, 3, 3, 0.95])
	lists.append([1, 1, 3, 4, 0.93])
	lists.append([1, 0.9, 3.6, 3, 0.98])
	lists.append([1, 0.9, 3.5, 3, 0.97])
	lists = torch.tensor(lists)

	res = NMS(lists, threshold)
	print(res)
	print(lists)