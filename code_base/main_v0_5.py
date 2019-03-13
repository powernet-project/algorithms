""" main
	v0.5
	Thomas Navidi

	How to run powernet algorithm modules

	Version Changes:
	- Includes some small virtual loads and random storage nodes
	- Went back to bounds since no bounds is inherently nonconvex
	- Loads ramp stats data for probability weighted prices
	- saves regulation hours and signal data at run time
	- split home hubs to run on individual nodes in series instead of all nodes at once

	To Do:
	- change vroot = 1.0436 comes from network data everywhere

"""

import numpy as np
import argparse
#from scipy.io import loadmat
import time

#from GC_algs_v0_5 import *
#from GC_algs_v0_1 import *

from LC_algs_v0_1 import *
from Network_v0_5 import *
from Forecaster_v0_1 import *

from GC_algs_NoScen_v0_5 import *

#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simulate Control')
parser.add_argument('--seed', default=0, help='random seed')
parser.add_argument('--storagePen', default=5, help='storage penetration percentage')
parser.add_argument('--solarPen', default=8, help='solar penetration percentage')
parser.add_argument('--test', default=1, help='Binary {0,1} for using small test dataset')
#parser.add_argument('--V_weight', default=500, help='voltage soft constraint weight')
FLAGS, unparsed = parser.parse_known_args()
print 'running with arguments: ({})'.format(FLAGS)
storagePen = float(FLAGS.storagePen)/10
solarPen = float(FLAGS.solarPen)/10
seed = int(FLAGS.seed)

np.random.seed(seed) # set random seed

#Initialize simulation parameters
nodesPen = np.maximum(solarPen,storagePen)
GCtime = 24
lookAheadTime = 24

if FLAGS.test:
	GCstepsTotal = 2
else:
	GCstepsTotal = 30

sellFactor = 0
GCscens = 1
LCscens = 1
V_weight = 1e5 # tuning parameter for voltage penalties
Vtol = .005 # tolerance bound on voltage penalties
ramp_weight_LC = 40 # make large to prioritize ramp following
NLweight = 4 # make 10X price when there are no bounds and 10X ramp

# Load Data
# IEEE 123 bus case PG&E data

loadMod = 1
presampleIdx = 168; # first week as presample data
startIdx = presampleIdx + 1 # starting index for the load dataset

# Load Residual Means and Covariance Dictionaries
#ResidualDict = loadmat('data/ResidualData123.mat')
#pMeans = ResidualDict['pMeans'][0,0]
#pCovs = ResidualDict['pCovs'][0,0]
pMeans = 0
pCovs = 0

# load power data and battery data
startIdx = 0
#load_data = np.load('data/Re__Simulation/processed_dat.npz') # 123 bus case 1 day data
load_data = np.load('static_data/gld_processed.npz') # 123 bus case 1 month data
#load_data = np.load('data/gld_processed_1000.npz') # 123 bus case 2 weeks data

netDemandFull = np.matrix(load_data['netpDemandFull'])
rDemandFull = np.matrix(load_data['netrDemandFull'])
pDemandFull = np.matrix(load_data['pDemandFull'])
battnodes = load_data['battnodes']
py2gld = load_data['py2gld_key']
qmax = np.matrix(np.reshape(load_data['qmax'], (len(battnodes),1)))
qmin = np.reshape(np.matrix(np.zeros(qmax.shape)), (len(battnodes),1))
umin = -qmax/3
umax = qmax/3
nn, T = netDemandFull.shape
print('power data shape:', netDemandFull.shape)
hours_total = T - GCtime - lookAheadTime
if hours_total < 0:
	rampDataAll = np.matrix(rampDataAll[0,:]) # use only first ramp for 24 hours of data
	hours_total = 1
	# For 1 day pecan street data
	GCtime = 12
	lookAheadTime = 12
	GCstepsTotal = 2

sr_dict = np.load('static_data/solar_ramp_data.npz')
rampUpData=sr_dict['rampUpData']
rampDownData=sr_dict['rampDownData']
sNormFull=sr_dict['sNormFull']
rampup_prob_hourly=sr_dict['rampup_prob_hourly']
rampdown_prob_hourly=sr_dict['rampdown_prob_hourly']

# Load network
network_data = np.load('static_data/network_data.npz')
#network_data = np.load('data/network_data_1000.npz')
root = 0
try:
	ppc = network_data['ppc'][()]
except:
	ppc = None
Ybus = network_data['Ybus'][()]

# Load Prices
prices = np.matrix(np.hstack((.25*np.ones((1,16)) , .35*np.ones((1,5)), .25*np.ones((1,3)))))
prices = np.tile(prices, (1,(GCtime*(GCstepsTotal+4)+startIdx)/24))
pricesFull = prices
reg_pricesFull = np.matrix(.05 * np.ones(pricesFull.shape)) # high value
#reg_pricesFull = np.matrix(.1* np.ones(pricesFull.shape)) # low value
# Load ramp probability
pData = np.load('static_data/RampReg_stats_pnorm.npz')
rampup_prob_hourly=pData['rampup_prob_hourly']
rampdown_prob_hourly=pData['rampdown_prob_hourly']
ramp_price_scale = 3
#ramp_weight_GC = ramp_price_scale
ramp_weight_GC = 40
#ramp_weight_LC = .3
ramp_prices_uFull = ramp_price_scale*np.reshape(rampup_prob_hourly, (1,rampup_prob_hourly.size))
ramp_prices_dFull = ramp_price_scale*np.reshape(rampdown_prob_hourly, (1, rampdown_prob_hourly.size))
# try ramp prices in between peak and low energy price

# Load Ramps
windPen = 1 # scaling factor for ramp amounts
ramp_tolerance = 0 # 5% of ramp amount tolerance on ramp works better with 0 tolerance
#rampDict = loadmat('data/rampDataAll.mat')
#rampUpData = rampDict['true_Uramp'] # dimensions are ramps X (start/end time start/end power)
rampUpData = rampUpData[1:,:] # remove first ramp since it occurs too early at time 2
#rampDownData = rampDict['true_Dramp']
rampDownData[:,[2,3]] = rampDownData[:,[3,2]] # swap down ramp amounts to make negative
rampDataAll = np.vstack((rampUpData, rampDownData))
#rampDataAll = np.array([ [9, 13, .9, .7], [30, 35, .9, .75] ])
#print 'ramps:', rampDataAll
rOrder = np.argsort(rampDataAll[:,0])
rampsNumTotal = len(rOrder)
rampDataAll = rampDataAll[rOrder,:]
rampDataAll[:,[0,1]] = rampDataAll[:,[0,1]] - 1 # subtract 1 for matlab indexing

# load regulation signals
reg = np.loadtxt('static_data/reg_data_BPA_5min_7days_nohead.csv', delimiter=',')
reg_max = reg[0,1]
reg_min = reg[0,2]
reg = reg[:,0]
reg_signal = reg[0:12]/100

# Make dictionary of ramps using battery capacity for scale
rampUAll = make_ramp_dict(rampDataAll, windPen, ramp_tolerance, qmax)
ramp_starts = np.sort(rampUAll.keys())
ramp_curr = np.array(ramp_starts[ramp_starts >= (720)]) # remove ramps after 720 hours
for ramp_key in ramp_curr:
	rampUAll.pop(ramp_key)
#print('all ramp times', np.sort(rampUAll.keys()))

# Random = True
#sNormFull = 0
sNormFull = sNormFull[:,0:T]
network = Network(storagePen, solarPen, nodesPen, pDemandFull, rDemandFull, pricesFull, ramp_prices_uFull, ramp_prices_dFull, reg_pricesFull, root, Ybus, startIdx, sNormFull, Vmin=0.95, Vmax=1.05, Vtol=0, v_root=1.022, random=True, rampUAll=rampUAll)

# Add Random Loads
loadPen = 0.3
AC_duty = 0.5 # air conditioner duty cycle
AC_energy = 2./1000*AC_duty # energy consumed by AC in an hour in MWh to maintain constant temp
ACs_perNode = 10 # 10 air conditioner per node
load_cap = ACs_perNode*AC_energy*0.5 * 1# energy capacity based on user preferences. Energy in temperature deadband. Assumed to be equal to energy consumed in 30 minutes
load_power = ACs_perNode * 2./1000 # power consumed by AC
AC_base = ACs_perNode*AC_energy
network.addRandomLoad(loadPen,load_cap,load_power)

qmax = network.qmax

#making your own ramps
qmax_total = np.sum(qmax)
times1 = np.array([12, 13, 14, 15])
mag1 = np.array([-0.41, -0.81, -1.22, -1.63])
# this next step is optional and can be removed
mag1 = mag1/np.abs(np.sum(mag1))*qmax_total*0.8 # scale ramp maginitude so total energy = 0.8 of total battery capacity

times2 = np.array([29, 30, 31, 32, 33, 34])
mag2 = np.array([0.41, 0.83, 1.25, 1.67, 2.09, 2.51])
mag2 = mag2/np.abs(np.sum(mag2))*qmax_total*0.85 # scale ramp maginitude so total energy = 0.85 of total battery capacity

times3 = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49]) + 10
mag3 = np.array([-0.19, -0.39, -0.58, -0.78, -0.97, -1.17, -1.36, -1.56, -1.75])
mag3 = mag3/np.abs(np.sum(mag3))*qmax_total*0.7 # scale ramp maginitude so total energy = 0.7 of total battery capacity
# controller automatically scales ramp signals that it cannot follow into ones that it can follow
# "cannot follow" also includes does not want to follow because it is too expensive based on economic optimization
# the new ramps are in variable rampUAll_followed and are also printed out

ramp_tolerance = 0
ramp1 = Ramp(times1, mag1, ramp_tolerance*mag1)
ramp2 = Ramp(times2, mag2, ramp_tolerance*mag2)
ramp3 = Ramp(times3, mag3, ramp_tolerance*mag3)
rampUAll = { times1[0] : ramp1, times2[0] : ramp2, times3[0] : ramp3 }
# adjust prices to coincide with custom ramps
ramp_prices_uFull[:,times2] = ramp_price_scale
ramp_prices_dFull[:,list(times1)+list(times3)] = ramp_price_scale
print('all ramp times', np.sort(rampUAll.keys()))

# initialize network
# Random = False
sNormFull = 0
#network = Network(storagePen, solarPen, nodesPen, pDemandFull, rDemandFull, pricesFull, ramp_prices_uFull, ramp_prices_dFull, reg_pricesFull, root, Ybus, startIdx, sNormFull, Vmin=0.95, Vmax=1.05, Vtol=0, v_root=1.0436, random=False)
#network.inputStorage(Ybus, netDemandFull, battnodes, qmin, qmax, umin, umax)
network.inputrampUAll(rampUAll, ramp_prices_uFull, ramp_prices_dFull)
print 'finished network initialization'


#get network information
nodesStorage = network.battnodes
storageNum = len(nodesStorage)
print('Storage nodes:', storageNum, nodesStorage)
qmin = network.qmin
qmax = network.qmax
umax = network.umax

# Initialize forecaster
#reformat data for network
"""
if nn > 300:
	for i in nodesStorage:
		#pMeans['b'+str(i+1)] = 0
		#pCovs['b'+str(i+1)] = 0
		pMeans = 0
		pCovs = 0
else:
	for i in nodesStorage:
		pMeans['b'+str(i+1)] = pMeans['b'+str(i+1)].flatten()

		#Make 0 for no forecast
		pMeans['b'+str(i+1)] = np.zeros(pMeans['b'+str(i+1)].shape)
		pCovs['b'+str(i+1)] = np.zeros(pCovs['b'+str(i+1)].shape)
"""

forecast_error = 0 #.1
forecaster = Forecaster(forecast_error, pMeans, pCovs)

# initialize controllers
t_idx = 0 # set controller t_idx to something non zero after this if wanted
GC = Global_Controller(network, forecaster, GCtime, lookAheadTime, GCscens, sellFactor, V_weight, Vtol, ramp_weight_GC)
q0 = np.matrix(np.zeros(qmax.shape)) #set initial q0 to be 0
LCs = {}
idx = 0
for node_idx in nodesStorage:
	# to split futher across homes do the same as below but divide all inputs except node_idx and pricesFull by # homes in node
	local_data = LocalData(node_idx, qmax[idx], qmin[idx], umax[idx], umin[idx], netDemandFull[node_idx,:], rDemandFull[node_idx,:], pricesFull, startIdx=0)
	# divide q0 by # homes in node
	LCs[idx] = Local_Controllers(local_data, forecaster, GCtime+lookAheadTime, LCscens, NLweight, sellFactor, ramp_weight_LC, nodesStorage[idx], q0[idx])
	idx += 1

# Initialize values to save
rampUAll_followed = {}
Qall = np.matrix(np.zeros((storageNum,GCtime*(GCstepsTotal+4)+1)))
Uall = np.matrix(np.zeros((storageNum,GCtime*(GCstepsTotal+4))))
Ureg_dict = {}
Ubound_minAll = np.matrix(np.zeros((storageNum,GCtime*(GCstepsTotal+4))))
Ubound_maxAll = np.matrix(np.zeros((storageNum,GCtime*(GCstepsTotal+4))))
eps_all = np.matrix(np.zeros((storageNum,GCtime*(GCstepsTotal+4))))

# Use when starting in the middle
"""
t_idx = 460
ramp_starts = np.sort(rampUAll.keys())
ramp_curr = np.array(ramp_starts[ramp_starts <= (t_idx + GCtime + lookAheadTime)]) # Consider ramps in both GC and lookahead time
for ramp_key in ramp_curr:
	rampUAll.pop(ramp_key)
"""

# Remove all ramps for just cost min
"""
ramp_starts = np.sort(rampUAll.keys())
ramp_curr = np.array(ramp_starts[ramp_starts <= (721 + GCtime + lookAheadTime)]) # Consider ramps in both GC and lookahead time
for ramp_key in ramp_curr:
	rampUAll.pop(ramp_key)
"""

for steps in range(GCstepsTotal):

	### Run Global Controller ###
	print('Running at time:', t_idx)
	start_time = time.time()
	realS, pricesCurrent, LCtime, rampFlag, RstartList, QiList, RsignList, ramp_next = GC.runStep(q0, t_idx)
	print 'GC comp time', time.time() - start_time

	#print ramp_next

	Ubound_minAll[:,t_idx:(t_idx+GCtime+lookAheadTime)] = GC.Ubound_min
	Ubound_maxAll[:,t_idx:(t_idx+GCtime+lookAheadTime)] = GC.Ubound_max
	eps_all[:,t_idx:(t_idx+GCtime+lookAheadTime)] = GC.eps
	U_sched = GC.U_sched

	# select regulation times
	nS = nodesStorage.size
	mask = np.absolute(U_sched/np.tile(np.reshape(umax,(nS,1)), (1,GCtime+lookAheadTime))) < .03 # do regulation when charge less than 3% of max charge
	mask2 = np.sum(mask,axis=0) == nS
	print 'Local controller running for hours:', LCtime
	#LC_cycles = 0
	if np.sum(mask2) > 0:
		reg_mag = np.multiply(np.tile(mask2, (nS,1)), GC.eps)
		reg_times = np.reshape(np.arange(GCtime+lookAheadTime), (1,GCtime+lookAheadTime))[mask2]
		#print 'hours for regulation signal following:', reg_times
		reg_times = reg_times[reg_times < LCtime]
		reg_times_abs = reg_times + t_idx
	else:
		reg_times = np.array([])

	LCtime_abs = LCtime + t_idx
	added_time = 0
	print 'number of regulation hours in period:', reg_times.size
	print 'actual hours for regulation signal following:', reg_times+t_idx

	### Run Local Controllers ###
	# run several time periods at once should be switched to sequentially
	print('running local controllers at time:', t_idx)
	for i in range(nodesStorage.size):
		if rampFlag == 0:
			RsignList_split = []
			QiList_split = []
		else:
			RsignList_split = np.reshape(RsignList[i,:], (1,RsignList[i,:].size))
			QiList_split = QiList[i,:]
		# to split further across homes do the same as below but input realS and QiList/#homes
		#st = time.time()
		Uall[i,t_idx:(t_idx+LCtime)], Qall[i,(t_idx+1):(t_idx+LCtime+1)], t_idx_new = LCs[i].runPeriod(t_idx, realS[i,:], pricesCurrent, LCtime, rampFlag, RstartList, QiList_split, RsignList_split)
		#print 'LC time', st - time.time()

	Uall[:,reg_times+t_idx] = 0
	Qall[:,reg_times+t_idx] = Qall[:,reg_times+t_idx-1]
	t_idx = t_idx_new

	"""
	for cyc in range(reg_times.size+1): # used to be reg_times.size+1 inside range
		if cyc > reg_times.size - 1:
			LCtime_cycle = LCtime_abs - t_idx
			regFlag = 0
		else:
			LCtime_cycle = reg_times_abs[cyc] - t_idx
			regFlag = 1

		### Run Local Controllers ###
		# run several time periods at once should be switched to sequentially
		print('running local controllers at time:', t_idx)
		for i in range(nodesStorage.size):
			if rampFlag == 0:
				RsignList_split = []
				QiList_split = []
			else:
				RsignList_split = np.reshape(RsignList[i,:], (1,RsignList[i,:].size))
				QiList_split = QiList[i,:]
			# to split further across homes do the same as below but input realS and QiList/#homes
			#st = time.time()
			Uall[i,t_idx:(t_idx+LCtime_cycle)], Qall[i,(t_idx+1):(t_idx+LCtime_cycle+1)], t_idx_new = LCs[i].runPeriod(t_idx, realS[i,added_time:], pricesCurrent[added_time:], LCtime_cycle, rampFlag, RstartList, QiList_split, RsignList_split)
			#print 'LC time', st - time.time()
		added_time += t_idx_new-t_idx
		t_idx = t_idx_new
		if regFlag == 1:
			print('reg hour', t_idx)
			Uall[:,t_idx] = 0
			Qall[:,t_idx+1] = Qall[:,t_idx]
			# split regulation signal
			reg_alphas = GC.eps[:,reg_times[cyc]]
			Ureg_dict[t_idx] = np.multiply(np.tile(reg_alphas, (1,reg_signal.size)),np.tile(reg_signal, (nS,1)))
			t_idx += 1
			added_time += 1
		print('after LC time', t_idx)
	"""
	
	q0 = Qall[:,t_idx+1]
	#print q0

	"""
	if t_idx > 71:
		realp = netDemandFull[nodesStorage,startIdx:(startIdx+t_idx)] + Uall[:,0:(t_idx)]
		plt.figure()
		plt.plot(realp[4,:].T)
		plt.figure()
		plt.plot(np.sum(Uall[:,0:(t_idx)], axis=0).T)
		plt.figure()
		plt.plot(np.sum(Qall[:,0:(t_idx)], axis=0).T)
		plt.figure()
		plt.plot(Uall[4,0:(t_idx)].T)
		plt.show()

		
		np.savetxt('real_power_inj_sample.csv', realp)
		np.savetxt('P_sample.csv', Uall[:,0:(t_idx+LCtime+1)])
		np.savetxt('Q_sample.csv', Qall[:,0:(t_idx+LCtime+1)])
		np.savetxt('node_order.csv', py2gld[battnodes])
	"""
	

	# get latest state of charge before splitting ramp
	q0 = np.reshape(Qall[:,t_idx], (storageNum,1))
	if np.any(np.less(q0, qmin)): # Correct for computational inaccuracies
		q0 += .00001
		print('q0 too low')
	elif np.any(np.greater(q0, qmax)):
		q0 += -.00001
		print('q0 too high')

	### Split Ramp Signal ###
	# split signal is sent directly to storage units to follow without running LC MPC optimization
	if rampFlag == 1:
		print('splitting ramp signal')
		start_time = time.time()
		ramp_maybe = rampUAll[ramp_next]
		U_ramp_test, ramp_duration, t_idx_new = GC.disaggregateRampSignal(t_idx, q0, ramp_next)
		print 'ramp distribute comp time', time.time() - start_time
		if np.any(np.isnan(U_ramp_test)):
			print 'ramp skipped at time', ramp_next
		else:
			rampUAll_followed[ramp_next] = ramp_maybe

			# U_ramp_test is the battery charging action during a ramp
			Uall[:,t_idx:(t_idx+ramp_duration)] = U_ramp_test
			Qall[:,(t_idx+1):(t_idx+ramp_duration+1)] = np.reshape(Qall[:,t_idx], (storageNum,1)) + np.cumsum(U_ramp_test,axis=1)
			t_idx = t_idx_new

		# get latest state of charge before running global controller
		q0 = np.reshape(Qall[:,t_idx], (storageNum,1))
		if np.any(np.less(q0, qmin)): # Correct for computational inaccuracies
			q0 += .00001
			print('q0 too low')
		elif np.any(np.greater(q0, qmax)):
			q0 += -.00001
			print('q0 too high')

	print 'after ramp time', t_idx

	#if (t_idx%100) < 20:
		# Save Data
	np.savez('results/t1_'+str(solarPen)+str(storagePen),Qall=Qall,Uall=Uall,t_idx=t_idx, Ubounds_vio=GC.Ubounds_vio, rampSkips=GC.rampSkips, rampUAll=rampUAll_followed,
				netDemandFull=network.netDemandFull, rDemandFull=network.rDemandFull, ppc=ppc, nodesStorage=nodesStorage, pricesFull=pricesFull,
				Ubound_minAll=Ubound_minAll,Ubound_maxAll=Ubound_maxAll, reg_pricesFull=reg_pricesFull, ramp_weight_GC=ramp_weight_GC, NLweight=NLweight, V_weight=V_weight,
				eps_all=eps_all, forecast_error=forecast_error, network=network, Ureg_dict=Ureg_dict )
	print('SAVED')


#print Uall
#print Qall
# Save Data
np.savez('results/t1_'+str(solarPen)+str(storagePen),Qall=Qall,Uall=Uall,t_idx=t_idx, Ubounds_vio=GC.Ubounds_vio, rampSkips=GC.rampSkips, rampUAll=rampUAll_followed,
			netDemandFull=network.netDemandFull, rDemandFull=network.rDemandFull, ppc=ppc, nodesStorage=nodesStorage, pricesFull=pricesFull,
			Ubound_minAll=Ubound_minAll,Ubound_maxAll=Ubound_maxAll, reg_pricesFull=reg_pricesFull, ramp_weight_GC=ramp_weight_GC, NLweight=NLweight, V_weight=V_weight,
			eps_all=eps_all, forecast_error=forecast_error, network=network, Ureg_dict=Ureg_dict)
print('SAVED and Run completed')




