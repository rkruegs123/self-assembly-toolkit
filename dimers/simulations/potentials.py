import numpy as np


def Potential_S(r,ron,rcut):
	if(r<ron):
		return 1
	if(r>rcut):
		return 0
	return ( (rcut**2-r**2)**2 * (rcut**2+2*r**2-3*ron**2) ) / ( (rcut**2 - ron**2)**3 )
def Potential_dS(r,ron,rcut):
	if(r<ron):
		return 0
	if(r>rcut):
		return 0
	return (12*r*(r - rcut)*(r + rcut)*(r - ron)*(r + ron))/( (rcut**2 - ron**2)**3 )
def Potential_d2S(r,ron,rcut):
	if(r<ron):
		return 0
	if(r>rcut):
		return 0
	return 12*(5*r**4 + rcut**2*ron**2 - 3*r**2*(rcut**2+ron**2))/( (rcut**2 - ron**2)**3 )
def Potential_S_dS(r,ron,rcut):
	return[Potential_S(r,ron,rcut), Potential_dS(r,ron,rcut)]
def Potential_S_dS_d2S(r,ron,rcut):
	return[Potential_S(r,ron,rcut), Potential_dS(r,ron,rcut), Potential_d2S(r,ron,rcut)]




class MorsePotential():
	#parameters:
	# rmin --> assumed to be zero!
	# rmax
	# D0
	# alpha
	# r0

	@staticmethod
	def GetDefaultParams():
		return dict(D0=0,alpha=5.0,r0=1)

	@staticmethod
	def E(r,rmin,rmax,D0,alpha,r0):
		#D0, alpha, r0, rmax = rmin,rmax,D0,alpha,r0
		if r>=rmax:
			return 0.
		return D0*(np.exp(-2*alpha*(r-r0)) - 2*np.exp(-alpha*(r-r0)))

	@staticmethod
	def dE(r,rmin,rmax,D0,alpha,r0):
		if r >= rmax:
			return 0.
		return D0*(-2*alpha*np.exp(-2*alpha*(r-r0)) + 2*alpha*np.exp(-alpha*(r-r0)))

	@staticmethod
	def d2E(r,rmin,rmax,D0,alpha,r0):
		if r >= rmax:
			return 0.
		return D0*(4*alpha**2*np.exp(-2*alpha*(r-r0)) - 2*alpha**2*np.exp(-alpha*(r-r0)))

	@staticmethod
	def E_dE(r,rmin,rmax,D0,alpha,r0):
		if r >= rmax:
			return (0.,0.)
		E = D0*(np.exp(-2*alpha*(r-r0)) - 2*np.exp(-alpha*(r-r0)))
		g = D0*(-2*alpha*np.exp(-2*alpha*(r-r0)) + 2*alpha*np.exp(-alpha*(r-r0)))
		return (E,g)

	@staticmethod
	def E_f(r,rmin,rmax,D0,alpha,r0):
		(E,g) = E_dE(r,rmin,rmax,D0,alpha,r0)
		return (E,-g)


	@staticmethod
	def E_dE_d2E(r,rmin,rmax,D0,alpha,r0):
		if r >= rmax:
			return (0.,0.,0.)
		E = D0*(np.exp(-2*alpha*(r-r0)) - 2*np.exp(-alpha*(r-r0)))
		g = D0*(-2*alpha*np.exp(-2*alpha*(r-r0)) + 2*alpha*np.exp(-alpha*(r-r0)))
		k = D0*(4*alpha**2*np.exp(-2*alpha*(r-r0)) - 2*alpha**2*np.exp(-alpha*(r-r0)))
		return (E,g,k)






class MorseXPotential(MorsePotential):
	#parameters:
	# rmin --> assumed to be zero!
	# rmax
	# D0
	# alpha
	# r0
	# ron

	@staticmethod
	def GetDefaultParams():
		return dict(D0=0,alpha=5.0,r0=1,ron=0.9)

	@staticmethod
	def E(r,rmin,rmax,D0,alpha,r0,ron):
		return MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax)

	@staticmethod
	def dE(r,rmin,rmax,D0,alpha,r0,ron):
		return(MorsePotential().dE(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax) +
                 MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_dS(r,ron,rmax))

	@staticmethod
	def d2E(r,rmin,rmax,D0,alpha,r0,ron):
		return(MorsePotential().d2E(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax) +
                 2*MorsePotential().dE(r,rmin,rmax,D0,alpha,r0)*Potential_dS(r,ron,rmax) +
                 MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_d2S(r,ron,rmax))

	@staticmethod
	def E_dE(r,rmin,rmax,D0,alpha,r0,ron):
		(V, dV) = MorsePotential().E_dE(r,rmin,rmax,D0,alpha,r0)
		(S, dS) = Potential_S_dS(r,ron,rmax)
		return (V*S, dV*S+V*dS)

	@staticmethod
	def E_f(r,rmin,rmax,D0,alpha,r0,ron):
		(E,g) = MorseXPotential().E_dE(r,rmin,rmax,D0,alpha,rmin,ron)
		return (E,-g)

	@staticmethod
	def E_dE_d2E(r,rmin,rmax,D0,alpha,r0,ron):
		(V, dV, d2V) = MorsePotential().E_dE_d2E(r,rmin,rmax,D0,alpha,r0)
		(S, dS, d2S) = Potential_S_dS_d2S(r,ron,rmax)
		return (V*S, dV*S+V*dS, d2V*S+2*dV*dS+V*d2S)





class MorseXRepulsivePotential(MorsePotential):
	#parameters:
	# rmin --> assumed to be zero!
	# rmax
	# D0
	# alpha
	# r0
	# ron

	@staticmethod
	def GetDefaultParams():
		return dict(D0=0,alpha=5.0,r0=1,ron=0.9)

	@staticmethod
	def E(r,rmin,rmax,D0,alpha,r0,ron):
		return -MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax)

	@staticmethod
	def dE(r,rmin,rmax,D0,alpha,r0,ron):
		return(-MorsePotential().dE(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax) -
                 MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_dS(r,ron,rmax))

	@staticmethod
	def d2E(r,rmin,rmax,D0,alpha,r0,ron):
		return(-MorsePotential().d2E(r,rmin,rmax,D0,alpha,r0)*Potential_S(r,ron,rmax)  -
                 2*MorsePotential().dE(r,rmin,rmax,D0,alpha,r0)*Potential_dS(r,ron,rmax) -
                 MorsePotential().E(r,rmin,rmax,D0,alpha,r0)*Potential_d2S(r,ron,rmax))

	@staticmethod
	def E_dE(r,rmin,rmax,D0,alpha,r0,ron):
		(V, dV) = MorsePotential().E_dE(r,rmin,rmax,D0,alpha,r0)
		(S, dS) = Potential_S_dS(r,ron,rmax)
		return (-V*S, -dV*S-V*dS)

	@staticmethod
	def E_f(r,rmin,rmax,D0,alpha,r0,ron):
		(E,g) = MorseXRepulsivePotential().E_dE(r,rmin,rmax,D0,alpha,rmin,ron)
		return (E,-g)

	@staticmethod
	def E_dE_d2E(r,rmin,rmax,D0,alpha,r0,ron):
		(V, dV, d2V) = MorsePotential().E_dE_d2E(r,rmin,rmax,D0,alpha,r0)
		(S, dS, d2S) = Potential_S_dS_d2S(r,ron,rmax)
		return (-V*S, -dV*S-V*dS, -d2V*S-2*dV*dS-V*d2S)





class RepulsivePotential():
	#parameters:
	#rmin
	#rmax
	# A
	# alpha

	@staticmethod
	def GetDefaultParams():
		return dict(A=0,alpha=2.5)

	@staticmethod
	def E(r,rmin,rmax,A,alpha):
		if r >= rmax:
			return 0.
		return (A/(alpha*rmax))*(rmax-r)**alpha

	@staticmethod
	def dE(r,rmin,rmax,A,alpha):
		if r >= rmax:
			return 0.
		return (-A/rmax)*(rmax-r)**(alpha-1)

	@staticmethod
	def d2E(r,rmin,rmax,A,alpha):
		if r >= rmax:
			return 0.
		return (A*(alpha-1)/rmax)*(rmax-r)**(alpha-2)

	@staticmethod
	def E_dE(r,rmin,rmax,A,alpha):
		if r >= rmax:
			return (0.,0.)
		E = (A/(alpha*rmax))*(rmax-r)**alpha
		g = (-A/rmax)*(rmax-r)**(alpha-1)
		return (E,g)

	@staticmethod
	def E_f(r,rmin,rmax,A,alpha):
		(E,g) = RepulsivePotential().E_dE(r,rmin,rmax,A,alpha)
		return (E,-g)

	@staticmethod
	def E_dE_d2E(r,rmin,rmax,A,alpha):
		if r >= rmax:
			return (0.,0.,0.)
		E = (A/(alpha*rmax))*(rmax-r)**alpha
		g = (-A/rmax)*(rmax-r)**(alpha-1)
		k = (A*(alpha-1)/rmax)*(rmax-r)**(alpha-2)
		return (E,g,k)
