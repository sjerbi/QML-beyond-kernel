from main import *

# str(sys_args[6]) is heisenberg for the new explicit model

# Load previously generated data
sys_args = sys.argv
pickle_path = './results/n'+str(sys_args[1])+'_L'+str(sys_args[2])+'_T'+str(sys_args[3])+'_'+str(sys_args[4])+'_fashion'+(str(sys_args[5])=='True')*'_heisen'+'gauss.pckl'
l = pickle.load(open(pickle_path, 'rb'))
gen = l[0]
x_train, x_test, x_test2, y_train, y_test, y_test2 = gen.x_train, gen.x_test, gen.x_test2, gen.y_train, gen.y_test, gen.y_test2

# Generate new explicit model
n_qubits = int(sys_args[1])
qubits = cirq.GridQubit.rect(1, n_qubits)
observables = [cirq.Z(qubits[0])]
n_layers = int(sys_args[2])
heisenberg = (str(sys_args[6]) == 'True')
new_trn = Explicit(qubits, n_layers, observables, train=True, heisenberg=heisenberg)

# Train new explicit model
nb_steps = 500
for i in range(nb_steps):
	print(i, '/' + str(nb_steps))
		l, val = new_trn.learning_step(x_train, y_train, x_test, y_test)
        test = tf.keras.losses.MeanSquaredError()(new_trn.model(x_test2), y_test2).numpy()
        new_trn.test_history += [test]
		print('Training, validation, test: ', l, val, test)
		if val.numpy()<10**(-5):
            break

# For storage
new_trn.variables = new_trn.model.variables
new_trn.model = None

# Store in the same pickle file
if len(l) == 11:
	new_trns = l[-1]
else:
	new_trns = []
new_trns += [new_trn]
if len(l) == 9:
	l += [None, new_trns]
elif len(l) == 10:
	l += [new_trns]
else:
	l[-1] = new_trns
pickle.dump(l, open(pickle_path, 'wb'))