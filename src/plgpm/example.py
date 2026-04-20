from plgpm import PLGPM, mi, plots

K_list = [12, 12, 12]
model = PLGPM(K_list)
model.fit(S_train)

print("Held-out PLL:", model.score_pseudologlik(S_test))

analysis = model.analyze(
    data_states=S_test,
    pair=(0, 1),
    sample_kwargs={"n_samples": 50_000, "burn": 5_000, "thin": 10},
)

plots.plot_coupling_heatmap(analysis.coupling)
plots.plot_probability_heatmap(analysis.pair["prob_model"], title="Model P(x0, x1)")
plots.plot_probability_heatmap(analysis.pair["prob_data"], title="Data P(x0, x1)")
mi.plot_mi_comparison(analysis.mi_data, analysis.mi_model)
print(analysis.mi_summary)

results = model.evaluate(
    S_test,
    n_perm=200,
    gibbs_kwargs={"n_samples": 20_000, "burn": 2_000, "thin": 5},
)
print(results.pll_gain_gpm_minus_ind)
print(results.full_joint)
