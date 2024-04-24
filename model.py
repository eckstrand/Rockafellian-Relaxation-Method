import pyomo.environ as pyo
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampleid_pu = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.float32,
            default_value=tf.constant(np.nan))


    def set_sample_weight(self, pu_dict):
        sids = []
        pu   = []
        for k, v in pu_dict.items():
            sids.append(k)
            pu.append(v)
        self.sampleid_pu.insert(sids, pu)


    def get_sample_weight(self, sid):
        return self.sampleid_pu.lookup(sid)
    

def fc_royset_norton(n_classes):
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(320, activation='relu', use_bias=False)(x)
    x = layers.Dense(320, activation='relu', use_bias=False)(x)
    x = layers.Dense(200, activation='relu', use_bias=False)(x)
    outputs = layers.Dense(n_classes, activation='softmax', use_bias=False)(x)

    model = CustomModel(inputs = inputs, outputs = outputs, name = "fc_royset_norton")
    return model


def cnn_basic(n_classes):
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = CustomModel(inputs = inputs, outputs = outputs, name = "cnn_basic")
    return model


def pyo_model(p, losses, alpha, beta, theta, eta, u_reg, label): 
    '''
    Assumes `label` is an array of arrays. Each element of the array is the 1-hot encoding of the class. For example,
    if there are 3 classes, then the class "1" would be encoded as [0, 1, 0] and the class "0" would be encoded as
    [1, 0, 0].
    '''
    N = len(losses)
    model = pyo.ConcreteModel()
    model.u = pyo.Var(range(N), initialize=0, domain=pyo.Reals)

    if u_reg == 'l1':
        model.z = pyo.Var(range(N), initialize=0, domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p)*losses[i] + theta * model.z[i] for i in range(N)),
            sense = pyo.minimize
        ) 

        def boundL_rule(model, i):
            return -model.u[i] <= model.z[i]   
        model.boundL = pyo.Constraint(range(N), rule=boundL_rule)   

        def boundU_rule(model, i):
            return model.u[i] <= model.z[i]   
        model.boundU = pyo.Constraint(range(N), rule=boundU_rule)           

    elif u_reg == 'max':
        model.z = pyo.Var(range(N), initialize=0, domain=pyo.NonNegativeReals)

        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p)*losses[i] + model.z[i] for i in range(N)),
            sense = pyo.minimize
        ) 

        def boundL_rule(model, i): 
            if int(label[i][1]) == 1: # class "1"
               return alpha*model.u[i] <= model.z[i]   
            else:                     # class "0"
               return eta*model.u[i] <= model.z[i]
        model.boundL = pyo.Constraint(range(N), rule=boundL_rule)   

        def boundU_rule(model, i):             
            if int(label[i][1]) == 1: 
               return beta*model.u[i] <= model.z[i]   
            else:
               return theta*model.u[i] <= model.z[i]
        model.boundU = pyo.Constraint(range(N), rule=boundU_rule)

    elif u_reg == 'l2':
        model.obj = pyo.Objective(
            expr = sum((model.u[i] + p)*losses[i] + 0.5 * theta * model.u[i]**2 for i in range(N)),
            sense = pyo.minimize
        )
    else:
        raise Exception("Unsupported u-regularization = {}".format(u_reg))

    def rule1(model, i):
        return model.u[i] + p >= 0
    model.c1 = pyo.Constraint(range(N), rule=rule1)

    model.c2 = pyo.Constraint(expr = sum(model.u[i] for i in range(N)) == 0)
    
    return model