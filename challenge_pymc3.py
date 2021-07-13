#%%
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
# %%
#正規分布3っつの重なり
def gauss3(X, a1, mu1, b1, \
              a2, mu2, b2, \
              a3, mu3, b3):
    y = a1*np.exp(-b1*(X-mu1)*(X-mu1)/2) + \
        a2*np.exp(-b2*(X-mu2)*(X-mu2)/2) + \
        a3*np.exp(-b3*(X-mu3)*(X-mu3)/2)
    
    return y
# %%
#グラフ描画
def gauss_grf(X, Y_true, Y, title):
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('x')
    ax.set_ylabel('g(x)')
    ax.plot(X,Y_true, color='red', linewidth=2.0, label=r'$Y_{ture}$')
    ax.scatter(X, Y, c='b', s=10.0, label=r'$Y$')
    plt.title(title)
    plt.legend()

    return fig ,ax 
# %%
#ばらつきありガウス
N = 301
X = np.linspace(0, 3.0, num=N, endpoint=True)

a1_true = 0.587
mu1_true = 1.210
b1_true = 95.689

a2_true = 1.522
mu2_true = 1.455
b2_true = 146.837

a3_true = 1.183
mu3_true = 1.703
b3_true = 164.469

Y_true = gauss3(X, a1_true,mu1_true,b1_true, \
                a2_true,mu2_true,b2_true, \
                a3_true,mu3_true,b3_true )

sigma_noise = 0.01
np.random.seed(seed=1)
Y_noise = np.random.normal(loc=0, scale=sigma_noise, size=N)
Y_noise_mean = np.mean(Y_noise)
Y_noise_std = np.std(Y_noise)

Y = Y_true + Y_noise

print('std',Y_noise_std, '\tmean',Y_noise_mean)
# %%
fig, ax = gauss_grf(X, Y_true, Y, '3peaks')
# %%
prior_mu_sd = 1.0 / np.sqrt(5.0)
print (prior_mu_sd)

def MCMC(Ydata):
    np.random.seed(seed=1)
    with pm.Model() as model:
        mu1 = pm.Normal('mu1', mu=1.5, sd=prior_mu_sd)
        a1 = pm.Gamma('a1', alpha=5.0, beta=5.0)
        b1 = pm.Gamma('b1', alpha=5.0, beta=0.04)

        mu2 = pm.Normal('mu2', mu=1.5, sd=prior_mu_sd)
        a2 = pm.Gamma('a2', alpha=5.0, beta=5.0)
        b2 = pm.Gamma('b2', alpha=5.0, beta=0.04)
        
        mu3 = pm.Normal('mu3', mu=1.5, sd=prior_mu_sd)
        a3 = pm.Gamma('a3', alpha=5.0, beta=5.0)
        b3 = pm.Gamma('b3', alpha=5.0, beta=0.04)

        rmsd = pm.Uniform('rmsd', lower=0, upper=1)

        y = pm.Normal('y', mu=gauss3(X, a1, mu1, b1,\
                                        a2, mu2, b2,\
                                        a3, mu3, b3,),\
                      sd=rmsd, observed=Ydata)
        
        start = pm.find_MAP(method='powell')
        print(start)

        return pm.sample(10000, start=start, chains=2)



# %%
trace = MCMC(Y)
# %%
pm.summary(trace)

# %%
pm.traceplot(trace)

# %%
