from gpid.generate import generate_system
from gpid.estimate import approx_pid

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random
import time


def experiment(n=10):
	"""
	Run n experiments and save the results in a csv file

	Param
	-----------
	n : int
		number of simulated systems to generate

	"""

	dms = []
	dxs = []
	dys = []
	imxs = []
	imys = []
	imxys = []
	defxs = []
	defys = []
	uixs = []
	uiys = []
	ris = []
	sis = []
	flags = []
	countr = 0
	counts = 0
	countu = 0
	nu = 0
	for i in range(n):

		if np.mod(i,1000)==0:
			print(f'------- Completed {i} Simulations -------')

		try:
			# dm < dx <= dy
			if i <= n/4:
				# select dimensions
				dm = random.randrange(1,10)
				d1 = random.randrange(dm+1,11)
				d2 = random.randrange(dm+1,11)	
			# dx <= dy < dm
			elif n/4 < i <= n/2:
				# select dimensions
				dm = random.randrange(2,11)
				d1 = random.randrange(1,dm)
				d2 = random.randrange(1,dm)	
			# dx < dm < dy
			elif n/2 < i <= 3*n/4:
				# select dimensions
				dm = random.randrange(2,10)
				d1 = random.randrange(1,dm)
				d2 = random.randrange(dm+1,11)	
			# dm = dx = dy
			else:
				# select dimensions
				dm = random.randrange(1,11)
				d1 = dm
				d2 = dm
			# WLOG assume dx <= dy
			dx = min(d1,d2)
			dy = max(d1,d2)


			# generate a random multivariate system
			hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm = generate_system(dm=dm,dx=dx,dy=dy)
			

			attempt = 1
			solved = False
			while not solved:
				flag = ''

				# estimate the PID
				imx,imy,imxy,defx,defy,uix,uiy,ri,si = approx_pid(hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm,
										maxiter=attempt*5000,eps=1*(10**(-10+attempt)))
				# failure checks
				if ri <= 0:
					flag += '   negative redundancy \n'
				if si <= 0: 
					flag += '   negative synergy \n'
				if dm == 1:
					if (uix > 0.00001) and (uiy > 0.00001):
						flag += '   both have unique with univariate message \n'

				if flag != '':
					print('--------------')
					print(f'Warnings in Simulation {i} attempt {attempt}:')
					print(flag)
					print(f"RI: {ri}")
					print(f"SI: {si}")
					print(f"UIX: {uix}")
					print(f"UIY: {uiy}")
					print(f"imx: {imx}")
					print(f"imy: {imy}")
					print(f"imxy: {imxy}")
					print(f"defx: {defx}")
					print(f"defy: {defy}")
					print(f"hx: {hx}")
					print(f"hy: {hy}")
					print(f"dm: {dm}")
					print(f"dx: {dx}")
					print(f"dy: {dy}")
					print('--------------')
				
					if attempt == 10:
						if ri <= 0:
							count_r += 1
						if si <= 0: 
							count_s += 1
						if dm == 1:
							if (uix > 0.00001) and (uiy > 0.00001):
								count_u += 1
						break
					else:
						attempt += 1


				else:
					if attempt > 1:
						print("Success!")
					solved = True

			# store results
			dms.append(dm)
			dxs.append(dx)
			dys.append(dy)
			imxs.append(imx)
			imys.append(imy)
			imxys.append(imxy)
			defxs.append(defx)
			defys.append(defy)
			uixs.append(uix)
			uiys.append(uiy)
			ris.append(ri)
			sis.append(si)
			flags.append(flag)
		
		except KeyboardInterrupt:
			sys.exit()

		# if something breaks down, print and stor relevant details
		except:
			print('--------------')
			print('APPROXIMATION FAILED')
			print('ERROR: ', sys.exc_info()[0])
			print('HX')
			print(hx)
			print('HY')
			print(hy)
			dms.append(dm)
			dxs.append(dx)
			dys.append(dy)
			imxs.append(imx)
			imys.append(imy)
			imxys.append(imxy)
			defxs.append(0)
			defys.append(0)
			uixs.append(0)
			uiys.append(0)
			ris.append(0)
			sis.append(0)
			flags.append('error')
		

	if countr > 0:
		print(f"NEGATIVE REDUNDANCY {countr} OF {n} EXPERIMENTS")
	if counts > 0:
		print(f"NEGATIVE SYNERGY {counts} OF {n} EXPERIMENTS")
	if countu > 0:
		print(f"BOTH HAVE UNIQUE WITH dm==1 IN {countu} EXPERIMENTS")

	# create dataframe
	dic = {
		'dm':dms,
		'dx':dxs,
		'dy':dys,
		'imx':imxs,
		'imy':imys,
		'imxy':imxys,
		'defx':defxs,
		'defy':defys,
		'uix':uixs,
		'uiy':uiys,
		'ri':ris,
		'si':sis,
		'flag':flags
	}
	df = pd.DataFrame(dic)

	# add columns for normalized PID atoms
	df['nuix'] = df.uix.values/df.imxy.values
	df['nuiy'] = df.uiy.values/df.imxy.values
	df['nri'] = df.ri.values/df.imxy.values
	df['nsi'] = df.si.values/df.imxy.values

	df.to_csv('../results/results.csv')


def test_examples():
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
		
if __name__ == '__main__':
	random.seed(2010)
	np.random.seed(2010)

	start = time.perf_counter()
	experiment(n=80000)
	end = time.perf_counter()
	print(f'Experiment Completed. Time: {(end-start)/60}  Minutes')
	test_examples()
