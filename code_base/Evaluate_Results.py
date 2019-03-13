# Load results and evaluate performance

from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
from pypower.api import runpf, ppoption

from Network_v0_4 import *

def Violation_Process(allVoltage, Vmin, Vmax):
	vGreater = (allVoltage-Vmax).clip(min=0)
	vLess = (Vmin-allVoltage).clip(min=0)
	vio_plus_sum = np.sum(vGreater, axis=1) # bus# X sum of all over voltage violations
	vio_min_sum = np.sum(vLess, axis=1) # bus# X sum of all under voltage violations

	vio_plus_max = np.max(vGreater)
	vio_min_max = np.max(vLess)

	vio_timesbig = (vGreater + vLess) > 0
	vio_times = np.sum(vio_timesbig, axis=1) # bus# X number of times there are violations

	print 'Maximum over voltage violation: ', vio_plus_max
	print 'Maximium under voltage violation: ', vio_min_max
	vioTotal = np.sum(vio_min_sum+vio_plus_sum)
	print 'Sum of all voltage violations magnitude: ', vioTotal
	viosNum = sum(vio_times)
	print 'Number of voltage violations: ', viosNum
	vioAve = vioTotal/viosNum
	print 'Average voltage violation magnitude: ', vioAve

	vio_when = np.sum(vio_timesbig, axis=0)

	return vio_times, vio_plus_sum, vio_min_sum, vio_when

def PF_Sim(ppc, pDemand, rDemand):
	"""
	Uses PyPower to calculate PF to simulate node voltages after storage control
	Inputs: ppc - PyPower case dictionary
		GCtime - number of time steps between GC runs
		pDemand/rDemand - true values of real and reactive power demanded
		nodesStorage - list of storage nodes indexes
		U - storage control action
		rootV2 - voltage of the substation node
	Outputs: runVoltage - (buses X time) array of voltages
	"""

	ppc['bus'][:,2] = pDemand
	ppc['bus'][:,3] = rDemand
	#ppc['bus'][rootIdx,7] = rootVoltage # Doesnt actually set PF root voltage
	
	# for surpressing runpf output
	ppopt = ppoption(VERBOSE = 0, OUT_ALL = 0)
	ppc_out = runpf(ppc, ppopt)

	runVoltage = ppc_out[0]['bus'][:,7]

	return runVoltage

#fName = 'results/rampV1-1.npz'
#fName = 'results/rampV2-1.npz'
#fName = 'results/rampV2-vweight.npz'
#fName = 'results/rampV2-vweight2.npz'
#fName = 'results/rampV2-f2.npz'
#fName = 'results/rampV2-CM.npz'
#fName = 'results/anc_fakePF.npz'
#fName = 'results/anc_PF1.npz'
#fName = 'results/anc_g1.npz'
#fName = 'results/anc_2rps0.3.npz'

#fName = 'results/anc_rps0.009.npz' # 0.025
#fName = 'results/anc_rps0.05.npz' # 0.075 Then regulation price is 0.05
#fName = 'results/anc_rps0.2.npz'
#fName = 'results/anc_rps0.3.npz'
#fName = 'results/anc_rps0.5.npz'
#fName = 'results/anc_RW-rps0.2.npz'
#fName = 'results/anc_RW-rps0.3.npz'
#fName = 'results/anc_RWLC-rps0.009.npz'
#fName = 'results/anc_RWLC-rps0.3.npz'
#fName = 'results/anc_RWNLLC-rps0.3.npz'
#fName = 'results/anc_RWNLLC-rps0.2.npz'
#fName = 'results/anc_NL-rps0.2.npz'
#fName = 'results/anc_NL-rps0.3.npz'
#fName = 'results/anc_NL-rps0.5.npz'

#fName = 'results/anc_PFrps0.009.npz'
#fName = 'results/anc_PFrps0.05.npz'
#fName = 'results/anc_PFrps0.2.npz'
#fName = 'results/anc_PFrps0.3.npz'
#fName = 'results/anc_PFrps0.5.npz'

#fName = 'results/anc_sol0.5.npz'
#fName = 'results/anc_sol0.7.npz'

#fName = 'results/anc2_rps0.025.npz'
#fName = 'results/anc2_rps0.075.npz'
#fName = 'results/anc2_rps0.2.npz'
#fName = 'results/anc2_rps0.3.npz'
#fName = 'results/anc2_rps0.5.npz'
#fName = 'results/anc2_PFrps0.025.npz'
#fName = 'results/anc2_PFrps0.075.npz'
#fName = 'results/anc2_PFrps0.2.npz'
#fName = 'results/anc2_PFrps0.3.npz'
#fName = 'results/anc2_PFrps0.5.npz'

#fName = 'results/anc2_rps_slurm0.025.npz'
#fName = 'results/anc2_rps_slurm0.075.npz'
#fName = 'results/anc2_rps_slurm0.2.npz'
fName = 'results/anc2_rps_slurm0.3.npz'
#fName = 'results/anc2_rps_slurm0.5.npz'

def EvaluateResults(fName):
	print fName
	count = 0
	try:
		allData = np.load(fName)
	except:
		print 'no file found'
		count = 0
		return np.nan, np.nan, np.nan, np.nan, count

	Uall = allData['Uall']
	try:
		rampUAll = allData['rampUAll'][()]
		rampSkips = allData['rampSkips']
		ramp_weight_GC = allData['ramp_weight_GC']
		print 'ramp_weight_GC', ramp_weight_GC
		print 'number of skipped ramps', len(rampSkips)
		rampFlag = 1
	except:
		print 'no rampUAll'
		rampFlag = 0
	netDemandFull=allData['netDemandFull']
	rDemandFull=allData['rDemandFull']
	try:
		ppc=allData['ppc'][()]
		ppc['gen'][0,5] = 1.022
		ppc['bus'][0,7] = 1.022
	except:
		print 'no PPC'

	nodesStorage=allData['nodesStorage']
	pricesFull=allData['pricesFull']
	t_idx = allData['t_idx']
	eps_all = allData['eps_all']
	try:
		forecast_error = allData['forecast_error']
		print('forecast_error:', forecast_error)
	except:
		print 'no forecast_error in data'

	#print ppc['bus'][:,2]

	print 't_idx', t_idx

	t_idx = 70

	"""
	if t_idx < 700:
		print 'invalid run'
		count = 0
		return np.nan, np.nan, np.nan, np.nan, count
	"""

	presampleIdx = 168-1; # first week as presample data
	startIdx = presampleIdx + 1 # starting index for the load dataset

	#startIdx = 0 # for pecan street datasets

	nB, T = netDemandFull.shape
	nS, tt = Uall.shape

	#calculate before storage cost of electricity
	clip_demand = netDemandFull[nodesStorage,startIdx:startIdx+t_idx].clip(min=0)
	all_demand = np.sum(clip_demand, axis=0)
	cost_pre = np.dot(pricesFull[:,0:t_idx], all_demand.T)

	# add regulation signal following
	umax = np.max(Uall,axis=1)
	Qall = np.cumsum(Uall,axis=1)
	mask = np.absolute(Uall[:,0:t_idx])/np.tile(np.reshape(umax,(nS,1)), (1,t_idx)) < .1 # do regulation when charge less than 10% of max charge
	mask2 = np.sum(mask,axis=0) >= nS
	print 'number of regulation hours:', np.sum(mask2)
	regulation = np.tile(mask2, (nS,1))*eps_all[:,0:t_idx]
	print 'Total regulation capacity:', np.sum(regulation)
	total_reg = np.sum(regulation)

	data = np.load('data/RampReg_stats_pnorm.npz')
	reg_max_frac=data['reg_max_frac']
	reg_min_frac=data['reg_min_frac']
	#print reg_min_frac

	Ureg = regulation*reg_max_frac

	#print Ureg[:,0:72]

	#Uall[:,t_idx:t_idx+2] = 10./Uall.shape[0]
	t_idx = 70

	ramp = np.zeros(t_idx+2)
	for key in rampUAll.keys():
		rr = rampUAll[key]
		ramp[rr.times] = rr.mag

	Uall[:,rr.times] = Uall[:,rr.times] - 10./204

	reg = np.loadtxt('data/reg_data_BPA_5min_7days_nohead.csv', delimiter=',')

	print reg.shape
	reg_max = reg[0,1]
	reg_min = reg[0,2]
	reg = reg[:,0]

	reg_max_frac = np.max(reg)/reg_max
	reg_min_frac = np.min(reg)/reg_min
	print 'maximum positive reg capcity used:', reg_max_frac
	print 'maximum negative reg capacity used:', reg_min_frac

	eps = 62
	alphas = Uall[:,28]/np.sum(Uall[:,28])
	reg_ind = np.dot(np.reshape(alphas,(alphas.size,1)), np.reshape(reg[0:24]/reg_max*eps,(1,24)))
	print reg_ind.shape

	l_data = np.load('results/HH_heuristic_tracking-1.npz')
	agg_all = l_data['agg_all']
	base_all = l_data['base_all']
	blb_reg = reg[0:288]/reg_max*eps + base_all/25
	bl_reg = reg[0:288]/reg_max*eps + agg_all/25

	plt.figure()
	plt.plot(blb_reg)
	plt.plot(bl_reg)
	plt.gca().legend(('regulation signal','aggregate device power'))

	regmod = np.sum(reg_ind,axis=0)
	regmod += np.random.normal(size=regmod.shape)

	plt.figure()
	plt.plot(reg_ind.T)
	plt.figure()
	plt.plot(regmod)
	plt.plot(reg[0:24]/reg_max*eps, 'rx')
	plt.gca().legend(('Aggregate of devices','Regulation signal'))
	plt.plot()

	umod = np.sum(Uall[:,0:t_idx+2],axis=0)
	umod[29:35] += 3

	plt.figure()
	plt.plot(umod)
	#ramp[6:-2] = np.sum(Uall[:,6:t_idx],axis=0)
	#print ramp
	plt.plot(ramp, 'go')
	plt.plot(np.sum(Ureg[:,0:t_idx+2],axis=0), 'rx')
	plt.gca().legend(('Aggregate of devices','Ramp signal','regulation signal'))
	plt.figure()
	plt.plot(np.cumsum(np.sum(Uall[:,0:t_idx+2],axis=0)))
	plt.figure()
	plt.plot(Uall[:,0:t_idx+2].T)
	plt.show()

	# simulate voltages
	allVoltage = np.zeros((nB,t_idx))
	netDemandFull2 = netDemandFull.copy()
	pDemand = netDemandFull2[:,startIdx:startIdx+t_idx]
	pDemand[nodesStorage,0:t_idx] += Uall[:,0:t_idx]
	pDemand[nodesStorage,0:t_idx] += Ureg # add worst case regulation signal following to test voltages
	for i in range(t_idx):
		allVoltage[:,i] = PF_Sim(ppc, pDemand[:,i], rDemandFull[:,int(startIdx+i)])

	vio_times, vio_plus_sum, vio_min_sum, vio_when = Violation_Process(allVoltage, 0.95, 1.05)

	vio_total_square = np.sum(np.square(vio_min_sum + vio_plus_sum))

	print '-- Desired Traits --'

	print 'vio total square', vio_total_square

	# calculate after storage cost
	all_net = netDemandFull[nodesStorage,startIdx:startIdx+t_idx] + Uall[:,0:t_idx]
	clip_demand = all_net.clip(min=0)
	all_demand = np.sum(clip_demand, axis=0)
	cost_post = np.dot(pricesFull[:,startIdx:startIdx+t_idx], all_demand.T)

	print 'before storage cost', cost_pre
	print 'after storage cost', cost_post
	print 'difference', cost_pre - cost_post

	arb_prof = cost_pre - cost_post

	print np.sum(vio_when[0:t_idx] >= 1)

	if rampFlag == 1:
		ramp_starts = np.sort(rampUAll.keys())
		# print max(ramp_starts) # 30 days worth of ramp data
		capacities = []
		for i in ramp_starts:
			times = rampUAll[i].times
			capacities.append( np.sum(rampUAll[i].mag) )

		print 'total ramp capacity:', np.sum(np.abs(capacities))
		total_ramp = np.sum(np.abs(capacities))
	else:
		# make PF ramps
		windPen = 1 # scaling factor for ramp amounts
		ramp_tolerance = 0 # 5% of ramp amount tolerance on ramp works better with 0 tolerance
		rampDict = loadmat('data/rampDataAll.mat')
		rampUpData = rampDict['true_Uramp'] # dimensions are ramps X (start/end time start/end power)
		rampUpData = rampUpData[1:,:] # remove first ramp since it occurs too early at time 2
		rampDownData = rampDict['true_Dramp']
		rampDownData[:,[2,3]] = rampDownData[:,[3,2]] # swap down ramp amounts to make negative
		rampDataAll = np.vstack((rampUpData, rampDownData))
		rOrder = np.argsort(rampDataAll[:,0])
		rampsNumTotal = len(rOrder)
		rampDataAll = rampDataAll[rOrder,:]
		rampDataAll[:,[0,1]] = rampDataAll[:,[0,1]] - 1 # subtract 1 for matlab indexing

		qmax = np.max(np.cumsum(Uall,axis=1),axis=1)

		# Make dictionary of ramps using battery capacity for scale
		rampUAll = make_ramp_dict(rampDataAll, windPen, ramp_tolerance, qmax)
		ramp_starts = np.sort(rampUAll.keys())
		ramp_curr = np.array(ramp_starts[ramp_starts >= (720)]) # remove ramps after 720 hours
		for ramp_key in ramp_curr:
			rampUAll.pop(ramp_key)

		ramp_starts = np.sort(rampUAll.keys())
		ramp_curr = np.array(ramp_starts)
		arb_comp = 0
		total_ramp = 0
		rampSkips = 0
		for ramp_key in ramp_curr:
			times = rampUAll[ramp_key].times
			mags = rampUAll[ramp_key].mag
			error = np.sum(np.abs(np.sum(Uall[:,times],axis=0) - mags))
			#print error
			#print error/np.sum(np.abs(mags))
			profit = np.sum(np.abs(mags)) - error
			if profit < 0:
				arb_comp -= profit
				rampSkips += 1
			else:
				total_ramp += profit

		print 'arb_comp:', arb_comp
		print 'total_ramp:', total_ramp
		print '# of skpped ramps', rampSkips

		count = 1


	return vio_total_square, arb_prof, total_reg, total_ramp, count


if __name__ == '__main__':

	#EvaluateResults('results/anc2_PFrps0.0.npz')

	"""
	names = ['results/anc2_rps_slurm_Vdown0.025.npz', 
	'results/anc2_rps_slurm_Vdown0.075.npz',
	'results/anc2_rps_slurm_Vdown0.2.npz',
	'results/anc2_rps_slurm_Vdown0.3.npz',
	'results/anc2_rps_slurm_Vdown0.5.npz']
	names = ['results/anc2_rps_slurm_NS0.025.npz', 
	'results/anc2_rps_slurm_NS0.075.npz',
	'results/anc2_rps_slurm_NS0.2.npz',
	'results/anc2_rps_slurm_NS0.3.npz',
	'results/anc2_rps_slurm_NS0.5.npz']
	names = ['results/anc2PF_rps_slurm0.025.npz', 
	'results/anc2PF_rps_slurm0.075.npz',
	'results/anc2PF_rps_slurm0.2.npz',
	'results/anc2PF_rps_slurm0.3.npz',
	'results/anc2PF_rps_slurm0.5.npz']
	"""

	names = ['results/JanMS_3_0.80.5.npz']

	vio_total_square2 = []
	arb_prof2 = []
	total_reg2 = []
	total_ramp2 = []
	count2 = []

	for fName in names:
		vio_total_square, arb_prof, total_reg, total_ramp, count = EvaluateResults(fName)
		vio_total_square2.append(vio_total_square)
		arb_prof2.append(arb_prof)
		total_reg2.append(total_reg)
		total_ramp2.append(total_ramp)
		count2.append(count)

	print vio_total_square2
	print arb_prof2
	print total_reg2
	print total_ramp2
	print count2














