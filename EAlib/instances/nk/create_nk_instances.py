# Instance generator from Arkadiy

import os
import itertools
import numpy as np

def nk_solver(N, K, S, subfunctions, precision=4):
	overlap = K - S
	matrix = np.zeros((N, 2**overlap), dtype=np.float64)
	fitness = 0.0

	if overlap == 0:
		return sum(np.max([v for v in sf.values()]) for sf in subfunctions)

	for i, subfunction in enumerate(subfunctions):
		for pos in subfunction:
			#print(pos)
			value = subfunction[pos]
			#print(value)
			if i > 0:
				overlap_left = pos[:overlap]
				overlap_left_ind = np.sum([int(overlap_left[len(overlap_left)-1-j])*(2**j) for j in range(len(overlap_left))])
				#print(pos[:overlap], overlap_left_ind)
				value_from_overlap_left = matrix[i-1, overlap_left_ind]
			else:
				value_from_overlap_left = 0

			contrib = value_from_overlap_left + value
			#print (value_from_overlap_left,value,contrib)

			if i != len(subfunctions) - 1:
				overlap_right = pos[-overlap:]
				overlap_right_ind = np.sum([int(overlap_right[len(overlap_right)-1-j])*(2**j) for j in range(len(overlap_right))])
				# print(pos[-overlap:], overlap_right_ind)
				
				matrix[i, overlap_right_ind] = np.max([contrib, matrix[i, overlap_right_ind]])
			else:
				matrix[i,:] = np.max([matrix[i,0], contrib])

	vtr = np.max(matrix)
	print(np.max(matrix), vtr, fitness)
	return vtr

#######################################################################################

def generate_nk_landscape(k, s, dims, precision=4, num_instances=10):
	
	np.random.seed(42)

	for dim in dims:
		
		folder_name = './instances/n%d_s%d/L%d' % (k, s, dim)
		
		if not os.path.exists('./instances/n%d_s%d' % (k, s)):
			os.mkdir('./instances/n%d_s%d' % (k, s))

		if not os.path.exists(folder_name):
			os.mkdir(folder_name)

		for instance in range(1, num_instances+1):
			filename = os.path.join(folder_name, '%d.txt' % instance)
			file = open(filename,'w')
			M = (dim-k+s) // s
			file.write('%d %d\n' % (dim, M))
			subfunctions = []
			for i in range(0, dim-k+1, s):
				file.write('%d ' % k)
				for j in range(i, i+k):
					file.write(str(j)+' ')
				file.write('\n')
				table = {}
				for comb in itertools.product(['0','1'], repeat=k):
					comb = ''.join(comb)
					value = np.round(np.random.uniform(0,1), precision)
					file.write('\"%s\" %.4f\n' % (comb, value))
					table[comb] = value
				subfunctions.append(table)

			vtr = nk_solver(dim, k, s, subfunctions)
			filename_vtr = os.path.join(folder_name, '%d_vtr.txt' % instance)		
			file = open(filename_vtr,'w')
			file.write('%f' % vtr)
			file.close()

generate_nk_landscape(5, 1, [20,40,80,160,320,640,1280], 4, 10) #N5-S1 (overlap 4)
generate_nk_landscape(5, 2, [20,40,80,160,320,640,1280], 4, 10) #N5-S2 (overlap 3)
generate_nk_landscape(5, 4, [20,40,80,160,320,640,1280], 4, 10) #N5-S4 (overlap 1)
generate_nk_landscape(5, 5, [20,40,80,160,320,640,1280], 4, 10) #N5-S4 (overlap 0)

generate_nk_landscape(4, 1, [20,40,80,160,320,640,1280], 4, 10) #N4-S1 (overlap 3)
generate_nk_landscape(4, 3, [20,40,80,160,320,640,1280], 4, 10) #N4-S2 (overlap 1)
generate_nk_landscape(4, 4, [20,40,80,160,320,640,1280], 4, 10) #N4-S2 (overlap 0)

generate_nk_landscape(6, 1, [20,40,80,160,320,640,1280], 4, 10) #N6-S1 (overlap 5)
generate_nk_landscape(6, 5, [20,40,80,160,320,640,1280], 4, 10) #N6-S5 (overlap 1)
generate_nk_landscape(6, 6, [20,40,80,160,320,640,1280], 4, 10) #N6-S5 (overlap 0)

#######################################################################################
