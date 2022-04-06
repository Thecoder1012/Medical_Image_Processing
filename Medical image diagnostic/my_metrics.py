from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops,math_ops,nn
import tensorflow.keras as keras
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf

class gmean(keras.metrics.Metric):

  def __init__(self,
                thresholds=None,
                top_k=None,
                class_id=None,
                name=None,
                dtype=None):
      
    super(gmean, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id =None

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
      
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives, 
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    specificity = math_ops.div_no_nan(self.true_positives,
                                  self.true_positives + self.false_positives)
    sensitivity = math_ops.div_no_nan(self.true_positives,
                                  self.true_positives + self.false_negatives)
    result = math_ops.sqrt(specificity*sensitivity)
    return result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(gmean, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class average_class_specific_accuracy(keras.metrics.Metric):
# this code is specifically for three class classification
    def __init__(self,
                 name="acsa", **kwargs):
        super(average_class_specific_accuracy, self).__init__(name=name, **kwargs)

   
        self.cat_true_positives0 = self.add_weight(name="ctp0", initializer="zeros")
        self.cat_true_positives1 = self.add_weight(name="ctp1", initializer="zeros")
        self.cat_true_positives2 = self.add_weight(name="ctp2", initializer="zeros")
        self.cat_no_sample_class0 = self.add_weight(name="nc0", initializer="zeros")
        self.cat_no_sample_class1 = self.add_weight(name="nc1", initializer="zeros")
        self.cat_no_sample_class2 = self.add_weight(name="nc2", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)
        tp = K.equal(y_true,y_pred)
        tp0 = K.equal(K.ones_like(y_true)*0,y_pred)
        tp1 = K.equal(K.ones_like(y_true)*1,y_pred)
        tp2 = K.equal(K.ones_like(y_true)*2,y_pred)
        true_poss0 = K.sum(K.cast(K.all(K.stack([tp,tp0],axis = 0),axis = 0),dtype = tf.float32))
        true_poss1 = K.sum(K.cast(K.all(K.stack([tp,tp1],axis = 0),axis = 0),dtype = tf.float32))
        true_poss2 = K.sum(K.cast(K.all(K.stack([tp,tp2],axis = 0),axis = 0),dtype = tf.float32))
        no_sample_class0 = K.sum(K.cast(K.equal(K.ones_like(y_true)*0,y_true),dtype = tf.float32))
        no_sample_class1 = K.sum(K.cast(K.equal(K.ones_like(y_true)*1,y_true),dtype = tf.float32))
        no_sample_class2 = K.sum(K.cast(K.equal(K.ones_like(y_true)*2,y_true),dtype = tf.float32))

        self.cat_true_positives0.assign_add(true_poss0)
        self.cat_true_positives1.assign_add(true_poss1)
        self.cat_true_positives2.assign_add(true_poss2)
        self.cat_no_sample_class0.assign_add(no_sample_class0)
        self.cat_no_sample_class1.assign_add(no_sample_class1)
        self.cat_no_sample_class2.assign_add(no_sample_class2)


    def result(self):
        result = (math_ops.div_no_nan(self.cat_true_positives0,self.cat_no_sample_class0)+math_ops.div_no_nan(self.cat_true_positives1,self.cat_no_sample_class1)+math_ops.div_no_nan(self.cat_true_positives2,self.cat_no_sample_class2))/3
        return result
    
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives0.assign(0.0)
        self.cat_true_positives1.assign(0.0)
        self.cat_true_positives2.assign(0.0)
        self.cat_no_sample_class0.assign(0.0)
        self.cat_no_sample_class1.assign(0.0)
        self.cat_no_sample_class2.assign(0.0)
        
        
class average_class_specific_accuracy2(keras.metrics.Metric):
# this code is specifically for three class classification
    def __init__(self,
                 name="acsa", **kwargs):
        super(average_class_specific_accuracy2, self).__init__(name=name, **kwargs)

   
        self.cat_true_positives0 = self.add_weight(name="ctp0", initializer="zeros")
        self.cat_true_positives1 = self.add_weight(name="ctp1", initializer="zeros")
        self.cat_no_sample_class0 = self.add_weight(name="nc0", initializer="zeros")
        self.cat_no_sample_class1 = self.add_weight(name="nc1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)
        tp = K.equal(y_true,y_pred)
        tp0 = K.equal(K.ones_like(y_true)*0,y_pred)
        tp1 = K.equal(K.ones_like(y_true)*1,y_pred)
        true_poss0 = K.sum(K.cast(K.all(K.stack([tp,tp0],axis = 0),axis = 0),dtype = tf.float32))
        true_poss1 = K.sum(K.cast(K.all(K.stack([tp,tp1],axis = 0),axis = 0),dtype = tf.float32))
        no_sample_class0 = K.sum(K.cast(K.equal(K.ones_like(y_true)*0,y_true),dtype = tf.float32))
        no_sample_class1 = K.sum(K.cast(K.equal(K.ones_like(y_true)*1,y_true),dtype = tf.float32))

        self.cat_true_positives0.assign_add(true_poss0)
        self.cat_true_positives1.assign_add(true_poss1)
        self.cat_no_sample_class0.assign_add(no_sample_class0)
        self.cat_no_sample_class1.assign_add(no_sample_class1)


    def result(self):
        result = (math_ops.div_no_nan(self.cat_true_positives0,self.cat_no_sample_class0)+math_ops.div_no_nan(self.cat_true_positives1,self.cat_no_sample_class1))/2
        return result
    
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives0.assign(0.0)
        self.cat_true_positives1.assign(0.0)

        self.cat_no_sample_class0.assign(0.0)
        self.cat_no_sample_class1.assign(0.0)

class average_class_specific_accuracy8(keras.metrics.Metric):
# this code is specifically for three class classification
    def __init__(self,
                 name="acsa", **kwargs):
        super(average_class_specific_accuracy8, self).__init__(name=name, **kwargs)

   
        self.cat_true_positives0 = self.add_weight(name="ctp0", initializer="zeros")
        self.cat_true_positives1 = self.add_weight(name="ctp1", initializer="zeros")
        self.cat_true_positives2 = self.add_weight(name="ctp2", initializer="zeros")
        self.cat_true_positives3 = self.add_weight(name="ctp3", initializer="zeros")
        self.cat_true_positives4 = self.add_weight(name="ctp4", initializer="zeros")
        self.cat_true_positives5 = self.add_weight(name="ctp5", initializer="zeros")
        self.cat_true_positives6 = self.add_weight(name="ctp6", initializer="zeros")
        self.cat_true_positives7 = self.add_weight(name="ctp7", initializer="zeros")
        
        self.cat_no_sample_class0 = self.add_weight(name="nc0", initializer="zeros")
        self.cat_no_sample_class1 = self.add_weight(name="nc1", initializer="zeros")
        self.cat_no_sample_class2 = self.add_weight(name="nc2", initializer="zeros")
        self.cat_no_sample_class3 = self.add_weight(name="nc3", initializer="zeros")        
        self.cat_no_sample_class4 = self.add_weight(name="nc4", initializer="zeros")
        self.cat_no_sample_class5 = self.add_weight(name="nc5", initializer="zeros")        
        self.cat_no_sample_class6 = self.add_weight(name="nc6", initializer="zeros")
        self.cat_no_sample_class7 = self.add_weight(name="nc7", initializer="zeros")        


    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)
        tp = K.equal(y_true,y_pred)
        tp0 = K.equal(K.ones_like(y_true)*0,y_pred)
        tp1 = K.equal(K.ones_like(y_true)*1,y_pred)
        tp2 = K.equal(K.ones_like(y_true)*2,y_pred)
        tp3 = K.equal(K.ones_like(y_true)*3,y_pred)
        tp4 = K.equal(K.ones_like(y_true)*4,y_pred)
        tp5 = K.equal(K.ones_like(y_true)*5,y_pred)
        tp6 = K.equal(K.ones_like(y_true)*6,y_pred)
        tp7 = K.equal(K.ones_like(y_true)*7,y_pred)
        true_poss0 = K.sum(K.cast(K.all(K.stack([tp,tp0],axis = 0),axis = 0),dtype = tf.float32))
        true_poss1 = K.sum(K.cast(K.all(K.stack([tp,tp1],axis = 0),axis = 0),dtype = tf.float32))
        true_poss2 = K.sum(K.cast(K.all(K.stack([tp,tp2],axis = 0),axis = 0),dtype = tf.float32))
        true_poss3 = K.sum(K.cast(K.all(K.stack([tp,tp3],axis = 0),axis = 0),dtype = tf.float32))
        true_poss4 = K.sum(K.cast(K.all(K.stack([tp,tp4],axis = 0),axis = 0),dtype = tf.float32))
        true_poss5 = K.sum(K.cast(K.all(K.stack([tp,tp5],axis = 0),axis = 0),dtype = tf.float32))
        true_poss6 = K.sum(K.cast(K.all(K.stack([tp,tp6],axis = 0),axis = 0),dtype = tf.float32))
        true_poss7 = K.sum(K.cast(K.all(K.stack([tp,tp7],axis = 0),axis = 0),dtype = tf.float32))
        
        
        no_sample_class0 = K.sum(K.cast(K.equal(K.ones_like(y_true)*0,y_true),dtype = tf.float32))
        no_sample_class1 = K.sum(K.cast(K.equal(K.ones_like(y_true)*1,y_true),dtype = tf.float32))
        no_sample_class2 = K.sum(K.cast(K.equal(K.ones_like(y_true)*2,y_true),dtype = tf.float32))
        no_sample_class3 = K.sum(K.cast(K.equal(K.ones_like(y_true)*3,y_true),dtype = tf.float32))
        no_sample_class4 = K.sum(K.cast(K.equal(K.ones_like(y_true)*4,y_true),dtype = tf.float32))
        no_sample_class5 = K.sum(K.cast(K.equal(K.ones_like(y_true)*5,y_true),dtype = tf.float32))
        no_sample_class6 = K.sum(K.cast(K.equal(K.ones_like(y_true)*6,y_true),dtype = tf.float32))
        no_sample_class7 = K.sum(K.cast(K.equal(K.ones_like(y_true)*7,y_true),dtype = tf.float32))

        self.cat_true_positives0.assign_add(true_poss0)
        self.cat_true_positives1.assign_add(true_poss1)
        self.cat_true_positives2.assign_add(true_poss2)
        self.cat_true_positives3.assign_add(true_poss3)
        self.cat_true_positives4.assign_add(true_poss4)
        self.cat_true_positives5.assign_add(true_poss5)
        self.cat_true_positives6.assign_add(true_poss6)
        self.cat_true_positives7.assign_add(true_poss7)

        self.cat_no_sample_class0.assign_add(no_sample_class0)
        self.cat_no_sample_class1.assign_add(no_sample_class1)
        self.cat_no_sample_class2.assign_add(no_sample_class2)
        self.cat_no_sample_class3.assign_add(no_sample_class3)        
        self.cat_no_sample_class4.assign_add(no_sample_class4)
        self.cat_no_sample_class5.assign_add(no_sample_class5)        
        self.cat_no_sample_class6.assign_add(no_sample_class6)
        self.cat_no_sample_class7.assign_add(no_sample_class7)        

    def result(self):
        result = (math_ops.div_no_nan(self.cat_true_positives0,self.cat_no_sample_class0)+
                  math_ops.div_no_nan(self.cat_true_positives1,self.cat_no_sample_class1)+
                  math_ops.div_no_nan(self.cat_true_positives2,self.cat_no_sample_class2)+
                  math_ops.div_no_nan(self.cat_true_positives3,self.cat_no_sample_class3)+
                  math_ops.div_no_nan(self.cat_true_positives4,self.cat_no_sample_class4)+
                  math_ops.div_no_nan(self.cat_true_positives5,self.cat_no_sample_class5)+
                  math_ops.div_no_nan(self.cat_true_positives6,self.cat_no_sample_class6)+
                  math_ops.div_no_nan(self.cat_true_positives7,self.cat_no_sample_class7))/8
        return result
    
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives0.assign(0.0)
        self.cat_true_positives1.assign(0.0)
        self.cat_true_positives2.assign(0.0)
        self.cat_true_positives3.assign(0.0)
        self.cat_true_positives4.assign(0.0)
        self.cat_true_positives5.assign(0.0)
        self.cat_true_positives6.assign(0.0)
        self.cat_true_positives7.assign(0.0)
        
        self.cat_no_sample_class0.assign(0.0)
        self.cat_no_sample_class1.assign(0.0)
        self.cat_no_sample_class2.assign(0.0)
        self.cat_no_sample_class3.assign(0.0)
        self.cat_no_sample_class4.assign(0.0)
        self.cat_no_sample_class5.assign(0.0)
        self.cat_no_sample_class6.assign(0.0)
        self.cat_no_sample_class7.assign(0.0)
        
class average_class_specific_accuracy7(keras.metrics.Metric):
# this code is specifically for three class classification
    def __init__(self,
                 name="acsa", **kwargs):
        super(average_class_specific_accuracy7, self).__init__(name=name, **kwargs)

   
        self.cat_true_positives0 = self.add_weight(name="ctp0", initializer="zeros")
        self.cat_true_positives1 = self.add_weight(name="ctp1", initializer="zeros")
        self.cat_true_positives2 = self.add_weight(name="ctp2", initializer="zeros")
        self.cat_true_positives3 = self.add_weight(name="ctp3", initializer="zeros")
        self.cat_true_positives4 = self.add_weight(name="ctp4", initializer="zeros")
        self.cat_true_positives5 = self.add_weight(name="ctp5", initializer="zeros")
        self.cat_true_positives6 = self.add_weight(name="ctp6", initializer="zeros")
        
        self.cat_no_sample_class0 = self.add_weight(name="nc0", initializer="zeros")
        self.cat_no_sample_class1 = self.add_weight(name="nc1", initializer="zeros")
        self.cat_no_sample_class2 = self.add_weight(name="nc2", initializer="zeros")
        self.cat_no_sample_class3 = self.add_weight(name="nc3", initializer="zeros")        
        self.cat_no_sample_class4 = self.add_weight(name="nc4", initializer="zeros")
        self.cat_no_sample_class5 = self.add_weight(name="nc5", initializer="zeros")        
        self.cat_no_sample_class6 = self.add_weight(name="nc6", initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)
        tp = K.equal(y_true,y_pred)
        tp0 = K.equal(K.ones_like(y_true)*0,y_pred)
        tp1 = K.equal(K.ones_like(y_true)*1,y_pred)
        tp2 = K.equal(K.ones_like(y_true)*2,y_pred)
        tp3 = K.equal(K.ones_like(y_true)*3,y_pred)
        tp4 = K.equal(K.ones_like(y_true)*4,y_pred)
        tp5 = K.equal(K.ones_like(y_true)*5,y_pred)
        tp6 = K.equal(K.ones_like(y_true)*6,y_pred)
        true_poss0 = K.sum(K.cast(K.all(K.stack([tp,tp0],axis = 0),axis = 0),dtype = tf.float32))
        true_poss1 = K.sum(K.cast(K.all(K.stack([tp,tp1],axis = 0),axis = 0),dtype = tf.float32))
        true_poss2 = K.sum(K.cast(K.all(K.stack([tp,tp2],axis = 0),axis = 0),dtype = tf.float32))
        true_poss3 = K.sum(K.cast(K.all(K.stack([tp,tp3],axis = 0),axis = 0),dtype = tf.float32))
        true_poss4 = K.sum(K.cast(K.all(K.stack([tp,tp4],axis = 0),axis = 0),dtype = tf.float32))
        true_poss5 = K.sum(K.cast(K.all(K.stack([tp,tp5],axis = 0),axis = 0),dtype = tf.float32))
        true_poss6 = K.sum(K.cast(K.all(K.stack([tp,tp6],axis = 0),axis = 0),dtype = tf.float32))
        
        
        no_sample_class0 = K.sum(K.cast(K.equal(K.ones_like(y_true)*0,y_true),dtype = tf.float32))
        no_sample_class1 = K.sum(K.cast(K.equal(K.ones_like(y_true)*1,y_true),dtype = tf.float32))
        no_sample_class2 = K.sum(K.cast(K.equal(K.ones_like(y_true)*2,y_true),dtype = tf.float32))
        no_sample_class3 = K.sum(K.cast(K.equal(K.ones_like(y_true)*3,y_true),dtype = tf.float32))
        no_sample_class4 = K.sum(K.cast(K.equal(K.ones_like(y_true)*4,y_true),dtype = tf.float32))
        no_sample_class5 = K.sum(K.cast(K.equal(K.ones_like(y_true)*5,y_true),dtype = tf.float32))
        no_sample_class6 = K.sum(K.cast(K.equal(K.ones_like(y_true)*6,y_true),dtype = tf.float32))

        self.cat_true_positives0.assign_add(true_poss0)
        self.cat_true_positives1.assign_add(true_poss1)
        self.cat_true_positives2.assign_add(true_poss2)
        self.cat_true_positives3.assign_add(true_poss3)
        self.cat_true_positives4.assign_add(true_poss4)
        self.cat_true_positives5.assign_add(true_poss5)
        self.cat_true_positives6.assign_add(true_poss6)

        self.cat_no_sample_class0.assign_add(no_sample_class0)
        self.cat_no_sample_class1.assign_add(no_sample_class1)
        self.cat_no_sample_class2.assign_add(no_sample_class2)
        self.cat_no_sample_class3.assign_add(no_sample_class3)        
        self.cat_no_sample_class4.assign_add(no_sample_class4)
        self.cat_no_sample_class5.assign_add(no_sample_class5)        
        self.cat_no_sample_class6.assign_add(no_sample_class6)

    def result(self):
        result = (math_ops.div_no_nan(self.cat_true_positives0,self.cat_no_sample_class0)+
                  math_ops.div_no_nan(self.cat_true_positives1,self.cat_no_sample_class1)+
                  math_ops.div_no_nan(self.cat_true_positives2,self.cat_no_sample_class2)+
                  math_ops.div_no_nan(self.cat_true_positives3,self.cat_no_sample_class3)+
                  math_ops.div_no_nan(self.cat_true_positives4,self.cat_no_sample_class4)+
                  math_ops.div_no_nan(self.cat_true_positives5,self.cat_no_sample_class5)+
                  math_ops.div_no_nan(self.cat_true_positives6,self.cat_no_sample_class6))/7
        return result
    
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives0.assign(0.0)
        self.cat_true_positives1.assign(0.0)
        self.cat_true_positives2.assign(0.0)
        self.cat_true_positives3.assign(0.0)
        self.cat_true_positives4.assign(0.0)
        self.cat_true_positives5.assign(0.0)
        self.cat_true_positives6.assign(0.0)
        
        self.cat_no_sample_class0.assign(0.0)
        self.cat_no_sample_class1.assign(0.0)
        self.cat_no_sample_class2.assign(0.0)
        self.cat_no_sample_class3.assign(0.0)
        self.cat_no_sample_class4.assign(0.0)
        self.cat_no_sample_class5.assign(0.0)
        self.cat_no_sample_class6.assign(0.0)
