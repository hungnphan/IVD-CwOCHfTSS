import tensorflow as tf

class TreeProperties(object):
    '''
    :param max_leafs: maximum number of leafs
    :param n_features: maximum number of feature available within the data
    :param n_classes: number of classes
    '''
    def __init__(self,max_depth,max_leafs,n_features,n_classes,regularisation_penality=10.,decay_penality=0.9):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.n_features = n_features
        self.n_classes = n_classes
        self.epsilon = 1e-8
        self.decay_penality = decay_penality
        self.regularisation_penality = regularisation_penality

class Node(object):
    def __init__(self,id,depth,pathprob,tree):
        self.id = id
        self.depth = depth
        self.prune(tree)

        if self.isLeaf:
            self.W = tf.get_variable(name='weight_' + self.id,
                                     shape=(tree.params.n_features,tree.params.n_classes),
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            self.b = tf.get_variable(name='bias_' + self.id, shape=(tree.params.n_classes,), dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)
        else:
            self.W = tf.get_variable(name='weight_' + self.id, shape=(tree.params.n_features,1), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            self.b = tf.get_variable(name='bias_' + self.id, shape=(1,), dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)
        self.leftChild = None
        self.rightChild = None
        self.pathprob = pathprob
        self.epsilon = 1e-8 #this is a correction to avoid log(0)

    def prune(self,tree):
        '''
        prunes the leaf by setting isLeaf to True if the pruning condition applies.
        :param tree:
        '''
        self.isLeaf = (self.depth>=tree.params.max_depth)

    def build(self,x,tree):
        '''
        define the output probability of the node and build the children
        :param x:
        :return:
        '''
        self.prob = self.forward(x)

        if not(self.isLeaf):
            self.leftChild = Node(id=self.id + str(0), depth=self.depth + 1, pathprob=self.pathprob * self.prob,
                                  tree=tree)
            self.rightChild = Node(self.id + str(1), depth=self.depth + 1, pathprob=self.pathprob * (1. - self.prob),
                                   tree=tree)

    def forward(self,x):
        '''
        defines the output probability
        :param x:
        :return:
        '''
        if self.isLeaf:
            # TODO: replace by logsoft max for improved stability
            return tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        else:
            #return tf.keras.backend.hard_sigmoid(tf.matmul(x, self.W) + self.b)
            return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)
            #return tf.nn.softmax(tf.matmul(x, self.W) + self.b)


    def regularise(self,tree):
        if self.isLeaf:
            return 0.0
        else:
            alpha = tf.reduce_mean(self.pathprob * self.prob) / (self.epsilon + tf.reduce_mean(self.pathprob))
            return (-0.5 * tf.log(alpha + self.epsilon) - 0.5 * tf.log(1. - alpha + self.epsilon)) * (tree.params.decay_penality** self.depth)

    def get_loss(self,y,tree):
        if self.isLeaf:
            #return -tf.reduce_mean( tf.log( self.epsilon+tf.reduce_mean(y *self.prob, axis=1) )*self.pathprob )
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prob* self.pathprob,labels=y) )
        else:
            return tree.params.regularisation_penality * self.regularise(tree)

class SoftDecisionTree(object):
    def __init__(self, *args,**kwargs):
        self.params = TreeProperties(*args,**kwargs)
        self.n_nodes = 0
        self.n_leafs = 0
        self.loss = tf.Variable(initial_value = 0.0, dtype=tf.float32)
        self.output = list()
        self.leafs_distribution = list()

    def build_tree(self):
        self.tf_X = tf.placeholder(tf.float32, [None, self.params.n_features])
        self.tf_y = tf.placeholder(tf.float32, [None, self.params.n_classes])

        leafs = list()
        self.root = Node(id='0',depth=0,pathprob=tf.constant(1.0,shape=(1,)),tree=self)
        leafs.append(self.root )

        for node in leafs:
            self.n_nodes+=1
            node.build(x=self.tf_X,tree=self)
            self.loss = self.loss + node.get_loss(y=self.tf_y, tree=self)

            if node.isLeaf:
                self.n_leafs+=1
                self.output.append(node.prob)
                self.leafs_distribution.append(node.pathprob)
            else:
                leafs.append(node.leftChild)
                leafs.append(node.rightChild)

        self.leafs_distribution = tf.concat(self.leafs_distribution,axis=1)
        self.output = tf.concat(self.output,axis=1)

        # [_nBatches_,_nLeaves,_nClass_]
        self.output = tf.reshape(self.output, [-1, 2**self.params.max_depth, self.params.n_classes])

        print('Tree has {} leafs and {} nodes'.format(self.n_leafs,self.n_nodes))
        
        ## New way: v10.1.0
        # select a leaf_idx for each batch
        self.leaf_idx_of_batch = tf.argmax(input = self.leafs_distribution, 
                                           axis = 1, 
                                           output_type=tf.int32)

        ## Slitting for output classes
        # crop cls matrix for each batch, base on the selected leaf
        self.leaf_batch = self.slice_batch_by_leaf_idx(leaves_cls_of_batches = self.output, 
                                                       leaves_idx_of_each_batch = self.leaf_idx_of_batch)
        self.output_class = tf.argmax(input=self.leaf_batch, 
                                      axis=1)
        
        ## Merging for final output
        batch_idx = tf.reshape(tf.cast(tf.range(tf.shape(self.leaf_batch)[0]), tf.int32), [-1, 1])
        output_class_reshape = tf.reshape(tf.cast(self.output_class, tf.int32), [-1, 1])
        idx = tf.concat([batch_idx, output_class_reshape], 1)

        self.cls_prob = tf.gather_nd(params=self.leaf_batch, indices=idx)
        self.final_output = tf.concat([tf.reshape(self.cls_prob, [-1,1]), 
                                       tf.cast(tf.reshape(self.output_class, [-1,1]), tf.float32)], 
                                       axis=1, name="final_output")

    def slice_batch_by_leaf_idx(self, leaves_cls_of_batches, leaves_idx_of_each_batch):
        leaf_ifx = tf.reshape(leaves_idx_of_each_batch, shape=[-1, 1])
        batch_idx = tf.reshape(tf.cast(tf.range(tf.shape(leaf_ifx)[0]), tf.int32), 
                               shape=[-1, 1])
        crop_idx = tf.concat([batch_idx, leaf_ifx], axis=1)

        return tf.gather_nd(leaves_cls_of_batches, crop_idx)

    