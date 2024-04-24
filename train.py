'''
Implementation by Eric Eckstrand
eric.eckstrand@nps.edu
'''

import os
import time
import pickle
import argparse
import uuid
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from sklearn.utils import shuffle
from pyomo.opt import SolverFactory
import pyomo.environ as pyo
import ast

import model
import viz


def swap_labels(nobs, swap_pct, isoutlier, y, small_mnist):
    '''modifies `isoutlier` and `y` in place'''
    selected_idxs = np.random.choice(np.arange(0,nobs), int(nobs*swap_pct), replace=False)
    cls_lst = [0,1,2] if small_mnist else [0,1,2,3,4,5,6,7,8,9]
    for idx in selected_idxs:
        isoutlier[idx] = 1
        swp_lst = cls_lst.copy()
        swp_lst.remove(y[idx])
        y[idx] = np.random.choice(swp_lst, 1)[0]
    return


def main(args):
    log_txt = os.path.join(args.results_dir, 'log.txt')
    with open(log_txt, 'w', buffering=1) as log:
        # load MNIST dataset
        (x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
        if args.small_mnist:
            # only use MNIST digits 0, 1, 2
            select_0_1_2_trn = (y_trn == 0) | (y_trn == 1) | (y_trn == 2)
            y_trn = y_trn[select_0_1_2_trn]
            x_trn = x_trn[select_0_1_2_trn]

            select_0_1_2_tst = (y_tst == 0) | (y_tst == 1) | (y_tst == 2)
            y_tst = y_tst[select_0_1_2_tst]
            x_tst = x_tst[select_0_1_2_tst]

        # add channel dimension
        x_trn = np.reshape(x_trn, (len(x_trn), *(28, 28, 1)))
        x_tst = np.reshape(x_tst, (len(x_tst), *(28, 28, 1)))
        x_trn, y_trn = shuffle(x_trn, y_trn)
        x_tst, y_tst = shuffle(x_tst, y_tst)

        if args.trn_val_tst:
            # hold out 20% of the training examples for validation
            n_trn = int(len(x_trn)*0.80)
            x_trn, x_val = x_trn[0:n_trn, :, :, :], x_trn[n_trn: , :, :, :]
            y_trn, y_val = y_trn[0:n_trn], y_trn[n_trn:]
        else:
            x_val, y_val = [], []

        nobs_trn = len(x_trn)
        nobs_val = len(x_val)
        nobs_tst = len(x_tst)
        log.write("nobs_trn = {} \n".format(nobs_trn))
        log.write("nobs_tst = {} \n".format(nobs_tst))
        log.write("nobs_val = {} \n".format(nobs_val))

        # add and track contaminated examples in training set
        if args.open_set:
            raise Exception("Open set contamination not supported!")
        else: # closed-set contamination
            isoutlier_trn = np.zeros(nobs_trn, dtype=np.int32)
            isoutlier_val = np.zeros(nobs_val, dtype=np.int32)

            swap_labels(nobs_trn, args.swap_pct, isoutlier_trn, y_trn, args.small_mnist)
            swap_labels(nobs_val, args.swap_pct, isoutlier_val, y_val, args.small_mnist)
            # test set is uncontaminated
        
        y_trn = to_categorical(y_trn)
        y_val = to_categorical(y_val) if args.trn_val_tst else []
        y_tst = to_categorical(y_tst)
            
        # shuffle
        x_trn, y_trn, isoutlier_trn = shuffle(x_trn, y_trn, isoutlier_trn)
        x_val, y_val, isoutlier_val = shuffle(x_val, y_val, isoutlier_val)
        x_tst, y_tst                = shuffle(x_tst, y_tst               )  
            
        # data normalization
        x_trn_mean = x_trn.mean()
        x_trn_var  = x_trn.std()**2
        preprocess = tf.keras.Sequential([
            tf.keras.layers.Normalization(mean=x_trn_mean, variance=x_trn_var),
            ])

        # assign each sample a unique id. this is used to track the u values of each sample
        sids_trn = [str(uuid.uuid4()).encode('utf-8') for _ in range(nobs_trn)]

        ds_trn_  = tf.data.Dataset.from_tensor_slices((sids_trn, x_trn, y_trn)) 
        ds_trn   = ds_trn_.map(lambda sid, x, y: (sid, preprocess(x), y)).shuffle(1024).batch(args.batch).prefetch(buffer_size=tf.data.AUTOTUNE)

        ds_val_  = tf.data.Dataset.from_tensor_slices(([str(uuid.uuid4()).encode('utf-8') for _ in range(nobs_val)], x_val, y_val)) 
        ds_val   = ds_val_.map(lambda sid, x, y: (sid, preprocess(x), y)).batch(args.batch).prefetch(buffer_size=tf.data.AUTOTUNE)

        ds_tst_  = tf.data.Dataset.from_tensor_slices(([str(uuid.uuid4()).encode('utf-8') for _ in range(nobs_tst)], x_tst, y_tst)) 
        ds_tst   = ds_tst_.map(lambda sid, x, y: (sid, preprocess(x), y)).batch(args.batch).prefetch(buffer_size=tf.data.AUTOTUNE)

        n_classes = len(y_trn[0])

        # keep track of the set of known contaminated examples
        kc_set = set()  
        c = 0
        for sid, _, _ in ds_trn_:
            if isoutlier_trn[c] == 1:  # isoutlier_trn is in the same order as ds_trn_
                kc_set.add(sid.numpy())
            c += 1
        with open(os.path.join(args.results_dir, 'kc_set.pkl'), 'wb') as f:
            pickle.dump(kc_set, f)

        if args.u_reg == 'max' and n_classes != 2:
            raise Exception("`max` u-regularization method only supports binary classification, but n_classes = {}".format(n_classes))

        # the NN model        
        if args.use_model == 'basic':
            nn_model = model.cnn_basic(n_classes)
        elif args.use_model == 'fc_royset_norton':
            nn_model = model.fc_royset_norton(n_classes)
        else: 
            raise Exception("Unknown use_model = {}".format(args.use_model))
        
        if args.nn_opt == 'sgd':
            nn_opt = SGD(learning_rate=args.lr)
        elif args.nn_opt == 'adam':
            nn_opt = Adam(learning_rate=args.lr)  
        else:
            raise Exception("Unknown nn_opt = {}".format(args.nn_opt))

        metrics = ['accuracy']
        loss_fn = CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE)
        nn_model.compile(metrics=metrics)
        nn_model.summary()
        
        # maintain p, u, and previous-u. 
        # key-value pairs, where the key is the sample id.
        u_prev_dict = {}
        u_dict      = {sid:0 for sid in sids_trn}  # set it to 0 initially
        p           = 1 / nobs_trn  # just set it to 1/N for the time being

        @tf.function
        def train_step(sample_weight, x, y):
            with tf.GradientTape() as tape:
                y_pred = nn_model(x, training=True)

                loss = loss_fn(
                    y,
                    y_pred,
                    sample_weight=sample_weight
                    )

            trainable_vars = nn_model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            nn_opt.apply_gradients(zip(gradients, trainable_vars))
            nn_model.compiled_metrics.update_state(y, y_pred)
            return loss

        # adversarial perturbation
        @tf.function
        def adv_pert(x, l):
            with tf.GradientTape() as tape:
                tape.watch(x)
                pred = nn_model(x)
                loss = loss_fn(l, pred)

            grad = tape.gradient(loss, x)
            signed_grad = tf.sign(grad)
            return signed_grad

        if args.eager:
            tf.config.run_functions_eagerly(True)
        
        # keep a running history of training, validation and test performance
        history = {
            'loss'     : [], 
            'accuracy' : [],
            'val_loss' : [],
            'val_acc'  : [],
            'tst_loss' : [],
            'tst_acc'  : [],
            }

        # keep track of adversarial test performance for various epsilon values
        eps_tsts = [0]
        if args.adv_tst:
            eps_tsts = ast.literal_eval(args.eps_tst)
            for e in eps_tsts:
                history['tst_loss_' + str(e)] = []
                history['tst_acc_'  + str(e)] = []

        # keep track of fit and solve times
        t_fit   = []
        t_solve = []

        # main loop
        for iteration in range(args.n_iterations):
            iteration_artifacts = os.path.join(args.results_dir, str(iteration))
            if not os.path.exists(iteration_artifacts):
                os.mkdir(iteration_artifacts)

            log.write("------ iteration = {} ------\n\n".format(iteration))

            # compute p + u
            pu_dict = {}
            for k, u in u_dict.items():
                pu_dict[k] = p + u
            assert abs(sum([pu for _, pu in pu_dict.items()]) - 1.0) < 10**-6

            # update training observation weights
            nn_model.set_sample_weight(pu_dict)

            # phase 1: optimize wrt w
            log.write("*** phase 1 - the w's\n")
            for epoch in range(args.epochs):
                trn_losses_btchs = []
                ts = time.time()
                for sid, x, y in ds_trn:
                    pert = adv_pert(x, y) if args.adv_trn else 0
                    x = x + args.eps_trn * pert

                    sample_weight = nn_model.get_sample_weight(sid)
                    batch_losses = train_step(sample_weight, x, y)
                    trn_losses_btchs += list(batch_losses.numpy())

                trn_loss_epoch = sum(trn_losses_btchs) / len(trn_losses_btchs)
                history['loss'].append(trn_loss_epoch)

                metrics = {m.name: m.result() for m in nn_model.metrics}
                trn_acc_epoch = metrics['accuracy'].numpy()
                history['accuracy'].append(trn_acc_epoch)

                if args.trn_val_tst:
                    val_lsss_btchs = []
                    val_accs_btchs = []
                    for _, x_val, y_val in ds_val:
                        val_preds = nn_model.predict(x_val, verbose=0)
                        val_sample_weights = tf.tile(tf.constant(p, shape=[1]), [tf.shape(val_preds)[0]])

                        val_lsss_btchs += list(loss_fn(y_val, val_preds, sample_weight=val_sample_weights).numpy())
                        val_accs_btchs += list(tf.keras.metrics.categorical_accuracy(y_val, val_preds).numpy())

                    val_loss_epoch = sum(val_lsss_btchs) / len(val_lsss_btchs)
                    history['val_loss'].append(val_loss_epoch)
                    
                    val_acc_epoch = sum(val_accs_btchs) / len(val_accs_btchs)
                    history['val_acc'].append(val_acc_epoch)

                tst_lsss_eps_btchs = {k: [] for k in eps_tsts} 
                tst_accs_eps_btchs = {k: [] for k in eps_tsts} 
                for _, x_tst, y_tst in ds_tst:
                    pert = adv_pert(x_tst, y_tst) if args.adv_tst else 0
                    for eps_tst in eps_tsts:
                        x_tst = x_tst + eps_tst * pert  

                        tst_preds = nn_model.predict(x_tst, verbose=0)
                        test_sample_weights = tf.tile(tf.constant(p, shape=[1]), [tf.shape(tst_preds)[0]])

                        tst_lsss_eps_btchs[eps_tst] += list(loss_fn(y_tst, tst_preds, sample_weight=test_sample_weights).numpy())
                        tst_accs_eps_btchs[eps_tst] += list(tf.keras.metrics.categorical_accuracy(y_tst, tst_preds).numpy())

                for eps_tst in eps_tsts:
                    tst_loss_epoch = sum(tst_lsss_eps_btchs[eps_tst]) / len(tst_lsss_eps_btchs[eps_tst])
                    tst_acc_epoch  = sum(tst_accs_eps_btchs[eps_tst]) / len(tst_accs_eps_btchs[eps_tst])
                    history['tst_loss' + (('_' + str(eps_tst)) if args.adv_tst else '')].append(tst_loss_epoch)
                    history['tst_acc'  + (('_' + str(eps_tst)) if args.adv_tst else '')].append(tst_acc_epoch )

                et = time.time() - ts
                t_fit.append(et)

                epoch_str = "Epoch {}: trn_loss = {:.5f}, trn_acc = {:.5f}, sec = {:.3f}"
                epoch_str = epoch_str.format(epoch, trn_loss_epoch, trn_acc_epoch, et) + '\n'
                log.write(epoch_str)
                
                nn_model.compiled_metrics.reset_state()

            with open(os.path.join(iteration_artifacts, 'history.pkl'), 'wb') as f:
                    pickle.dump(history, f)

            # plot history
            viz.plot_learning_curves(history, iteration_artifacts, eps_tsts if args.adv_tst else [])

            # compute the loss for each observation using the weights from the last epoch
            if (iteration != args.n_iterations - 1) and args.u_opt:
                log.write("*** training observation losses\n")
                ts = time.time()
                l_dict      = {}
                y_true_dict = {}
                y_pred_dict = {}
                for s, x, y in ds_trn:
                    s_np = s.numpy()
                    y_np = y.numpy()

                    pert = adv_pert(x, y) if args.adv_trn else 0
                    x = x + args.eps_trn * pert

                    y_p_np = nn_model.predict(x, verbose=0)
                    l_np   = loss_fn(y_np, y_p_np).numpy()
                    
                    for i in range(len(s_np)):
                        sid = s_np[i]
                        y_true_dict[sid] = y_np[i]
                        y_pred_dict[sid] = y_p_np[i]
                        l_dict     [sid] = l_np[i]
                te = time.time()
                log.write("*** training observation losses = {} seconds\n\n".format(te-ts))

                # cache previous u values for comparison purposes with current u values
                u_prev_dict = u_dict

                # phase 2: optimize wrt u
                log.write("*** phase 2 - the u's\n")
                l         = []
                y_true    = []
                key_order = []
                for k in y_true_dict.keys():
                    l        .append(l_dict[k])
                    y_true   .append(y_true_dict[k])
                    key_order.append(k)
                pyo_model = model.pyo_model(p, l, args.alpha, args.beta, args.theta, args.eta, args.u_reg, y_true)
                opt = SolverFactory(args.solver, executable=args.solver_exe)
                ts = time.time()
                opt.solve(pyo_model, tee=False)
                te = time.time()
                log.write("objective value = {obj:.3E}\n".format(obj=pyo.value(pyo_model.obj)))
                t_solve.append(te-ts)
                log.write("*** phase 2 (solve) = {} seconds\n\n".format(t_solve[iteration-1]))

                # update and save u
                log.write("*** update and save the u's\n")
                ts = time.time()
                u = pyo.value(pyo_model.u[:])
                u_dict = {k: v for k, v in zip(key_order, u)}

                # cap the amount that any u can change from its previous u value.
                for k in key_order:
                    u_dict[k] = args.mu * u_dict[k] + (1-args.mu) * u_prev_dict[k]
                
                with open(os.path.join(iteration_artifacts, 'u_dict.pkl'), 'wb') as f:
                    pickle.dump(u_dict, f)

                te = time.time()
                log.write("*** update and save the u's = {} seconds\n\n".format(te-ts))
                
                # analyze the u values
                bins = 25
                viz.u_value_analysis(bins, u_dict, kc_set, iteration_artifacts)

            t_fit_total = sum(t_fit)
            t_solve_total = sum(t_solve)
            t_solve_avg = 0 if len(t_solve) == 0 else t_solve_total / len(t_solve) 
            t_fit_solve_total = t_fit_total + t_solve_total
            t_fit_pct = t_fit_total / t_fit_solve_total
            t_solve_pct = t_solve_total / t_fit_solve_total

            log.write("*** fit/solve time stats:\n")
            log.write("total fit   time = {} seconds\n".format(t_fit_total))
            log.write("total solve time = {} seconds\n".format(t_solve_total))
            log.write("fit   pct        = {}        \n".format(t_fit_pct))
            log.write("solve pct        = {}        \n".format(t_solve_pct))
            log.write("solve time/iter  = {}        \n".format(t_solve_avg))
            
            # flush the log file buffer
            log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results_dir', 
        type=str,   
        required=True,
        help="location to put the program artifacts."
        )
    parser.add_argument(
        '--small_mnist', 
        action='store_true',
        help="only use MNIST digits 0, 1, and 2. False if left unspecified."
        )
    parser.add_argument(
        '--trn_val_tst', 
        action='store_true',
        help="reserve some training data for validation. False if left unspecified."
        )
    parser.add_argument(
        '--open_set',     
        action='store_true',
        help="use open-set contamination. False if left unspecified."
        )
    parser.add_argument(
        '--swap_pct', 
        type=float,   
        required=False,
        default=0.5,
        help="[0-1) the percentage of labels to swap in the training dataset."
        )
    parser.add_argument(
        '--u_reg', 
        type=str,   
        required=False,
        help="[l1|l2|max] u-regularization type."
        )
    parser.add_argument(
        '--use_model', 
        type=str,   
        required=True,
        default='basic',
        help="[basic|fc_royset_norton] the model to use."
        )
    parser.add_argument(
        '--solver', 
        type=str,   
        required=True,
        help="[cplex] optimization solver"
        )
    parser.add_argument(
        '--solver_exe', 
        type=str,   
        required=True,
        help="location of solver executable"
        )
    parser.add_argument(
        '--nn_opt', 
        type=str,   
        required=True,
        help="[sgd|adam] the neural network optimizer to use."
        )
    parser.add_argument(
        '--lr', 
        type=float,   
        required=False,
        default=0.01,
        help="optimizer learning rate."
        )
    parser.add_argument(
        '--adv_trn', 
        action='store_true',
        help="apply adversarial perturbation to training images. False if left unspecified."
        )
    parser.add_argument(
        '--adv_tst', 
        action='store_true',
        help="apply adversarial perturbation to test images. False if left unspecified."
        )
    parser.add_argument(
        '--eps_trn', 
        type=float,   
        required=False,
        default=0.0,
        help="epsilon for training example perturbations."
        )
    parser.add_argument(
        '--eps_tst', 
        type=str,   
        required=False,
        default='0.0',
        help="comma-separated string of epsilons for test example perturbations."
        )
    parser.add_argument(
        '--n_iterations', 
        type=int,   
        required=True,
        help="number of iterations to perform. a single iteration consists of one round of phase 1 (neural network " \
            "training) and phase 2 (u-optimization routine)."
        )
    parser.add_argument(
        '--epochs', 
        type=int,   
        required=True,
        help="number of epochs to perform during a single iteration of neural network training."
        )
    parser.add_argument(
        '--batch', 
        type=int,   
        required=True,
        help="batch size used for both training and validation."
        )
    parser.add_argument(
        '--u_opt',     
        action='store_true',
        help="perform u optimization and update u. False if left unspecified."
        )
    parser.add_argument(
        '--mu', 
        type=float,   
        required=False,
        default=1.0,
        help="[0-1] mu for lp-based u-update scheme"
        )
    parser.add_argument(
        '--theta', 
        type=float,   
        required=False,
        help="regularization weight for u-optimization"
        )
    parser.add_argument(
        '--alpha', 
        type=float,   
        required=False,
        help="`max` u-regularization parameter."
        )
    parser.add_argument(
        '--beta', 
        type=float,   
        required=False,
        help="`max` u-regularization parameter."
        )
    parser.add_argument(
        '--eta', 
        type=float,   
        required=False,
        help="`max` u-regularization parameter."
        )
    parser.add_argument(
        '--eager', 
        action='store_true',
        help="eager execution. False if left unspecified."
        )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    args_txt = os.path.join(args.results_dir, 'args.txt')
    args_var = vars(args)
    with open(args_txt, 'w') as f:
        for arg in args_var:
            f.write(arg + ': ' + str(args_var[arg]) + '\n')

    main(args)

