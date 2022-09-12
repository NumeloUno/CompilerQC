import numpy as np
from CompilerQC import *
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def create_MLP_scaling(measure: bool, exponent: float):
    data = functions_for_database.get_all_distances(problem_folder='validation_set', scope_measure=measure, exponent=exponent)
    data += functions_for_database.get_all_distances(problem_folder='training_set', scope_measure=measure, exponent=exponent)

    # prepare X, y data for fitting
    Ns, Ks, Cs, n_3plaqss, n_4plaqss, n_3_cycless, n_4_cycless, n_n3plaqss, n_n4plaqss, plaq_ratio = (
        [],[],[],[],[],[],[],[],[], [])
    X_data, Y_data = [], []
    for i in range(len(data)):
        sums = [d.sum() for d in data[i][0]]
        n_3plaqs, n_4plaqs = sums[2], sums[3]
        N, K , C = data[i][1]
        n_3_cycles, n_4_cycles = data[i][2]
        n_n3plaqs, n_n4plaqs = n_3_cycles - n_3plaqs, n_4_cycles - n_4plaqs
        Ns.append(N)
        Ks.append(K)
        Cs.append(C)
        n_3plaqss.append(n_3plaqs)
        n_4plaqss.append(n_4plaqs)
        n_3_cycless.append(n_3_cycles)
        n_4_cycless.append(n_4_cycles)
        n_n3plaqss.append(n_n3plaqs)
        n_n4plaqss.append(n_n4plaqs)
        X_data.append([N, K, C, n_3_cycles, n_4_cycles])
        Y_data.append(sum(sums))


    X_data, Y_data = np.array(X_data), np.array(Y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,
                                                        random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=50000).fit(X_train, y_train)

    # save the model to disk
    if measure: 
        measure = "Scope"
    else:
        measure = "MoI"
    filename = paths.energy_scalings / f'MLP_{measure}_{exponent}.sav'
    pickle.dump(regr, open(filename, 'wb')) 

#     # load the model from disk
#     loaded_model = pickle.load(open(filename, 'rb'))
    
        # average performance
    print('score: ',regr.score(X_test, y_test))
    u = len(y_test)
    print('avg energy', y_test.sum() / u, 'avg error',(np.abs(y_test - regr.predict(X_test))).sum() / u)
    
    
def create_LHZ_scaling(measure: bool, exponent: float):
    energies, Ns = [], []
    for N in range(4, 20):
        graph = Graph.complete(N)
        qbits = Qbits.init_qbits_from_dict(graph, dict())
        nodes_object = Nodes(qbits)
        polygon_object = Polygons(nodes_object, scope_measure=measure, exponent=exponent)
        energy = Energy(polygon_object)
        e,_ = energy(qbits)
        energies.append(e)
        Ns.append(graph.C)

    x,y = Ns[:5], energies[:5]
    z = np.polyfit(x, y, 4)
    if measure: 
        measure = "Scope"
    else:
        measure = "MoI"
    np.save(paths.energy_scalings / f'LHZ_{measure}_{exponent}.npy', z)
    
if __name__=="__main__":
    for measure in [False, True]:
        for exponent in [0.5, 1, 2, 3, 4, 5, 6]:
       #     create_MLP_scaling(measure=measure, exponent=exponent)
            create_LHZ_scaling(measure=measure, exponent=exponent)
        
        
