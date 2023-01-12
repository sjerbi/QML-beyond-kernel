from main import *
import sklearn.neural_network
import sklearn.svm
import sklearn.ensemble

# Load previously generated data
sys_args = sys.argv
n_qubits = int(sys_args[1])
pickle_path = './results/n'+str(sys_args[1])+'_L'+str(sys_args[2])+'_T'+str(sys_args[3])+'_'+str(sys_args[4])+'_fashion'+(str(sys_args[5])=='True')*'_heisen'+'gauss.pckl'
l = pickle.load(open(pickle_path, 'rb'))
gen = l[0]
x_train, x_test, x_test2, y_train, y_test, y_test2 = gen.x_train_save, gen.x_test_save, gen.x_test2_save, gen.y_train, gen.y_test, gen.y_test2


# Start training classical models

# Neural network
hs = [10, 25, 50, 75, 100, 125, 150, 200]
nn_errs = np.zeros(len(hs))
nn_vals = np.zeros(len(hs))
nn_tests = np.zeros(len(hs))
for i, h in enumerate(hs):
	regr = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(h, h), max_iter=500)
	regr.fit(x_train, np.array(y_train).flatten())
	nn_errs[i] = mse(y_train, regr.predict(x_train))
	nn_vals[i] = mse(y_test, regr.predict(x_test))
	nn_tests[i] = mse(y_test2, regr.predict(x_test2))

print('NN', np.min(nn_vals))

# Linear kernel (SVR and Kernel ridge)
Cs = [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]
lk_errs = np.zeros((2, len(Cs)))
lk_vals = np.zeros((2, len(Cs)))
lk_tests = np.zeros((2, len(Cs)))
for i, C in enumerate(Cs):
	regr = sklearn.kernel_ridge.KernelRidge(kernel='linear', alpha=1/(2*C))
	regr.fit(x_train, np.array(y_train).flatten())
	lk_errs[0][i] = mse(y_train, regr.predict(x_train))
	lk_vals[0][i] = mse(y_test, regr.predict(x_test))
	lk_tests[0][i] = mse(y_test2, regr.predict(x_test2))
	regr = sklearn.svm.SVR(kernel='linear', C=C)
	regr.fit(x_train, np.array(y_train).flatten())
	lk_errs[1][i] = mse(y_train, regr.predict(x_train))
	lk_vals[1][i] = mse(y_test, regr.predict(x_test))
	lk_tests[1][i] = mse(y_test2, regr.predict(x_test2))

print('LK', np.min(lk_vals))

# Gaussian kernel
Cs = [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]
gammas = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0])/(n_qubits*np.std(x_train)**2)
gk_errs = np.zeros((2, len(Cs), len(gammas)))
gk_vals = np.zeros((2, len(Cs), len(gammas)))
gk_tests = np.zeros((2, len(Cs), len(gammas)))
for i, C in enumerate(Cs):
	for j, gamma in enumerate(gammas):
		regr = sklearn.kernel_ridge.KernelRidge(kernel='rbf', alpha=1/(2*C), gamma=gamma)
		regr.fit(x_train, np.array(y_train).flatten())
		gk_errs[0][i][j] = mse(y_train, regr.predict(x_train))
		gk_vals[0][i][j] = mse(y_test, regr.predict(x_test))
		gk_tests[0][i][j] = mse(y_test2, regr.predict(x_test2))
		regr = sklearn.svm.SVR(kernel='rbf', C=C, gamma=gamma)
		regr.fit(x_train, np.array(y_train).flatten())
		gk_errs[1][i][j] = mse(y_train, regr.predict(x_train))
		gk_vals[1][i][j] = mse(y_test, regr.predict(x_test))
		gk_tests[1][i][j] = mse(y_test2, regr.predict(x_test2))

print('GK', np.min(gk_vals))

# Random forest
depths = [2, 3, 4, 5]
n_trees = [25, 50, 100, 200, 500]
rf_errs = np.zeros((len(depths), len(n_trees)))
rf_vals = np.zeros((len(depths), len(n_trees)))
rf_tests = np.zeros((len(depths), len(n_trees)))
for i, depth in enumerate(depths):
	for j, n_tree in enumerate(n_trees):
		regr = sklearn.ensemble.RandomForestRegressor(max_depth=depth, n_estimators=n_tree)
		regr.fit(x_train, np.array(y_train).flatten())
		rf_errs[i][j] = mse(y_train, regr.predict(x_train))
		rf_vals[i][j] = mse(y_test, regr.predict(x_test))
		rf_tests[i][j] = mse(y_test2, regr.predict(x_test2))

print('RF', np.min(rf_vals))

# Gradient boosting
depths = [2, 3, 4, 5]
n_estimators = [25, 50, 100, 200, 500]
gb_errs = np.zeros((len(depths), len(n_estimators)))
gb_vals = np.zeros((len(depths), len(n_estimators)))
gb_tests = np.zeros((len(depths), len(n_estimators)))
for i, depth in enumerate(depths):
	for j, n_est in enumerate(n_estimators):
		regr = sklearn.ensemble.GradientBoostingRegressor(max_depth=depth, n_estimators=n_est)
		regr.fit(x_train, np.array(y_train).flatten())
		gb_errs[i][j] = mse(y_train, regr.predict(x_train))
		gb_vals[i][j] = mse(y_test, regr.predict(x_test))
		gb_tests[i][j] = mse(y_test2, regr.predict(x_test2))

print('GB', np.min(gb_vals))

# Adaboost
n_estimators = [25, 50, 100, 200, 500]
ada_errs = np.zeros(len(n_estimators))
ada_vals = np.zeros(len(n_estimators))
ada_tests = np.zeros(len(n_estimators))
for i, n_est in enumerate(n_estimators):
	regr = sklearn.ensemble.GradientBoostingRegressor(n_estimators=n_est)
	regr.fit(x_train, np.array(y_train).flatten())
	ada_errs[i] = mse(y_train, regr.predict(x_train))
	ada_vals[i] = mse(y_test, regr.predict(x_test))
	ada_tests[i] = mse(y_test2, regr.predict(x_test2))

print('Ada', np.min(ada_vals))


# Gather best performances
names = ['NN', 'linear', 'gaussian', 'random forest', 'gradient boost', 'Adaboost']
all_data = []

nn_val = np.min(nn_vals)
nn_argmin = np.unravel_index(np.argmin(nn_vals), nn_vals.shape)
nn_err = nn_errs[nn_argmin]
nn_test = nn_tests[nn_argmin]
all_data += [(nn_val, nn_err, nn_test, nn_argmin)]
lk_val = np.min(lk_vals)
lk_argmin = np.unravel_index(np.argmin(lk_vals), lk_vals.shape)
lk_err = lk_errs[lk_argmin]
lk_test = lk_tests[lk_argmin]
all_data += [(lk_val, lk_err, lk_test, lk_argmin)]
gk_val = np.min(gk_vals)
gk_argmin = np.unravel_index(np.argmin(gk_vals), gk_vals.shape)
gk_err = gk_errs[gk_argmin]
gk_test = gk_tests[gk_argmin]
all_data += [(gk_val, gk_err, gk_test, gk_argmin)]
rf_val = np.min(rf_vals)
rf_argmin = np.unravel_index(np.argmin(rf_vals), rf_vals.shape)
rf_err = rf_errs[rf_argmin]
rf_test = rf_tests[rf_argmin]
all_data += [(rf_val, rf_err, rf_test, rf_argmin)]
gb_val = np.min(gb_vals)
gb_argmin = np.unravel_index(np.argmin(gb_vals), gb_vals.shape)
gb_err = gb_errs[gb_argmin]
gb_test = gb_errs[gb_argmin]
all_data += [(gb_val, gb_err, gb_test, gb_argmin)]
ada_val = np.min(ada_vals)
ada_argmin = np.unravel_index(np.argmin(ada_vals), ada_vals.shape)
ada_err = ada_errs[ada_argmin]
ada_test = ada_tests[ada_argmin]
all_data += [(ada_val, ada_err, ada_test, ada_argmin)]

# Best classical
idx = np.argmin([a[0] for a in all_data])
print('Best classical:', names[idx], all_data[idx])

# Store in the same pickle file
if len(l) == 9:
	l += [(names[idx], all_data[idx])]
	pickle.dump(l, open(pickle_path, 'wb'))
elif len(l) == 11:
	l = l[:-1] + [(names[idx], all_data[idx]), l[-1]]
	pickle.dump(l, open(pickle_path, 'wb'))

