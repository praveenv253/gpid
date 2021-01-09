from generate import generate_system
from estimate import approx_pid

import random

if __name__ == '__main__':
	for _ in range(20):

		hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm = generate_system(
									dm=random.randrange(1,10),
									dx=random.randrange(1,10),
									dy=random.randrange(1,10))
		imx,imy,imxy,defx,defy,uix,uiy,ri,si = approx_pid(hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm)
		if (ri>0) and (si>0):
			pass
		else:
			print(f"RI: {ri}")
			print(f"SI: {si}")


	systems = ['random',
			   'both fully unique',
			   'only one unique',
			   'one is very unique',
			   'some of each',
			   'univariate']

	for system in systems:
		print(f" ------ {system} ------")
		hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm = generate_system(system)
		imx,imy,imxy,defx,defy,uix,uiy,ri,si = approx_pid(hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm)
		print(f"UIX: {uix}")
		print(f"UIY: {uiy}")
		print(f"RI: {ri}")
		print(f"SI: {si}")
		