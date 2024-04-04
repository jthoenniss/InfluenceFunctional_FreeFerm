

#check that the exact single mode IF matches the one where we integrate out explicitly In the limit:
B_spec_dens = np.zeros((dim_B_temp,dim_B_temp),dtype=np.complex_)
#this block below initialized the fine IF with T_ren, which is used to benchmark our exact continuous-time solution against the previous procedure
#___________________for IF defined from a spectral density____________________
for m in range (B_spec_dens.shape[0]//2):
    tau = m * delta_t
    for n in range (m+1,B_spec_dens.shape[0]//2):
        tau_p = n * delta_t
        B_spec_dens[2*m, 2*n + 1] += - (-1.) * delta_t**2 *t_hop**2* g_greater(0,beta,tau_p,tau+delta_t* alpha)#this is the element in lowest order (delta\tau) for the single mode environment: the spectral density corresponds to a delta function at w=0
        B_spec_dens[2*m+1, 2*n ] += - delta_t**2*t_hop**2 *g_lesser(0,beta,tau,tau_p+delta_t* alpha)
    B_spec_dens[2*m, 2*m + 1] += - (-1.) * delta_t**2*t_hop**2*g_lesser(0,beta,tau,tau+delta_t* alpha)
    B_spec_dens[2*m, 2*m + 1] += - 1. #constant from overlap at m=n
B_spec_dens -= B_spec_dens.T#antisymmetrize. This is twice the exponent matrix (this is assumed for later part in the code)


