K_list = [12, 12, 12]
model = PLGPM(K_list)
model.fit(S_train)

print("Held-out PLL:", model.score_pseudologlik(S_test))

samples = model.sample(n_samples=50000, burn=5000, thin=10)
C = model.coupling_matrix()

results = model.evaluate(
    S_test,
    n_perm=200,
    gibbs_kwargs={"n_samples": 20000, "burn": 2000, "thin": 5},
)

print(results.pll_gain_mrf_minus_ind)
print(results.full_joint)
