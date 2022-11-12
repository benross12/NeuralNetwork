class NeuralNetwork:

    def __init__(self, inputs, hiddens, outputs, functionInformation, learningRate):
        '''
        Constructor for the NeuralNetwork object
        Parameters:
                inputs - (int) the number of inputs to the NN
                hiddens - (int[]) - the number of hidden nodes at each layer of the NN
                outputs - (int) the number of outputs from the NN
                functionInformation (tuple):
                    function - (lambda float : float) the activation function for the NN
                    dFunction - (lambda float : float) the derrivative of the activation function
                learningRate - (float) the learning rate for the NN
        Returns:
            A new NeuralNetwork object
        '''

        # save the data about the neural net
        self.numberOfInputs = inputs
        self.numberOfHiddens = [x for x in hiddens]
        self.numberOfOutputs = outputs
        self.learningRate = learningRate
        self.function = functionInformation[0]
        self.dFunction = functionInformation[1]

        # initialise the weights and biases for all layers
        self.weights = []
        self.biases = []

        # input layer
        self.weights.append(np.random.rand(self.numberOfHiddens[0], self.numberOfInputs))
        self.biases.append(np.zeros((self.numberOfHiddens[0], 1)))

        # hidden layers
        for i in range(1, len(self.numberOfHiddens)):
            self.weights.append(np.random.rand(self.numberOfHiddens[i], self.numberOfHiddens[i - 1]))
            self.biases.append(np.zeros((self.numberOfHiddens[i], 1)))

        # output layer
        self.weights.append(np.random.rand(self.numberOfOutputs, self.numberOfHiddens[-1]))
        self.biases.append(np.zeros((self.numberOfOutputs, 1)))
        
    def setLearningRate(self, mult):
        '''
        multiply the learning rate by a given parameter
        Parameters
        ----------
        mult : float - the multiplyer the change learning rate by
        '''
        self.learningRate *= mult

    def transposeMatrix(self, arr):
        '''
        Parameters
        ----------
        arr : (numpy array) input matrix to be tranposed
        Returns
        -------
        (numpy array) transposed matrix
        '''
        if len(arr.shape) == 1:
            return arr[:, np.newaxis]
        if arr.shape[1] == 1:
            return [x[0] for x in arr]
        return np.transpose(arr)

    def train(self, inputs, targets):
        '''
        Train NeuralNetwork, given a set of inputs and their corrresponding outputs
        Parameters:
            inputs - (int[]) - a set of inputs to the NN
            targets - (int[]) - the values the network should predict
        Returns:
            None
        '''

        # store the activation values for each layer
        # activation values of layer n =
        # function(weights from layer n - 1 to n * activationValues of layer n - 1 + biases of layer n)
        activationValues = []

        # deal with input values outside of loop
        inputMatrix = self.transposeMatrix(np.array(inputs))
        firstLayerActivationValues = np.matmul(self.weights[0], inputMatrix);
        firstLayerActivationValues = np.add(firstLayerActivationValues, self.biases[0])
        firstLayerActivationValues = self.function(firstLayerActivationValues)
        activationValues.append(firstLayerActivationValues)

        # find the rest of the activation values
        for i in range(1, len(self.weights)):
            layerActivationValues = np.matmul(self.weights[i], activationValues[i - 1])
            layerActivationValues = np.add(layerActivationValues, self.biases[i])
            layerActivationValues = self.function(layerActivationValues)
            activationValues.append(layerActivationValues)

        # get the errors at each layer
        # error = transposed weights matrix * error matrix
        errors = []

        # deal with the last layer errors outside of the loop
        targetMatrix = self.transposeMatrix(np.array(targets))
        outputErrors = np.add(np.multiply(activationValues[-1][0], -1), (targetMatrix))
        errors.append(outputErrors)

        # find the rest of the errors
        for i in range(1, len(self.numberOfHiddens) + 1):
            matrixTransposed = self.transposeMatrix(self.weights[-i])
            errorVector = errors[-1]

            newErrors = np.matmul(matrixTransposed, errorVector)
            errors.append(newErrors)

        errors.reverse()

        # find delta weight and biases
        dWeight = []
        dBias = []

        deltaB = np.copy(activationValues[0])
        deltaB = self.dFunction(deltaB)
        deltaB = np.multiply(deltaB, errors[0])
        deltaB = np.multiply(deltaB, self.learningRate)
        dBias.append(deltaB)
        activationValuesTransposed = np.array(inputs)
        deltaW = np.multiply(deltaB, activationValuesTransposed)
        dWeight.append(deltaW)

        # find the hiddne layers deltas
        for i in range(1, len(self.numberOfHiddens) + 1):
            deltaB = np.copy(activationValues[i])
            deltaB = self.dFunction(deltaB)
            deltaB = np.multiply(deltaB, self.learningRate)
            deltaB = np.multiply(deltaB, errors[i])
            dBias.append(deltaB)

            activationValuesTransposed = self.transposeMatrix(activationValues[i - 1])
            deltaW = np.multiply(deltaB, activationValuesTransposed)
            dWeight.append(deltaW)

        for i in range(len(self.weights)):
            self.weights[i] = np.add(self.weights[i], dWeight[i])
            self.biases[i] = np.add(self.biases[i], dBias[i])

    def predictSingle(self, inputs):
        '''
        Make a single prediction, given a set of inputs
        Parameters:
            inputs - (int[]) - a set of inputs to feed into the NN
        Returns:
            (float[]) - the result of the feed forward algorithm
        '''

        # store the activation values for each layer
        # activation values of layer n =
        # function(weights from layer n - 1 to n * activationValues of layer n - 1 + biases of layer n)
        activationValues = []

        # deal with input values outside of loop
        inputMatrix = self.transposeMatrix(np.array(inputs))
        firstLayerActivationValues = np.matmul(self.weights[0], inputMatrix);
        firstLayerActivationValues = np.add(firstLayerActivationValues, self.biases[0])
        firstLayerActivationValues = self.function(firstLayerActivationValues)
        activationValues.append(firstLayerActivationValues)

        # find the rest of the activation values
        for i in range(1, len(self.weights)):
            layerActivationValues = np.matmul(self.weights[i], activationValues[i - 1])
            layerActivationValues = np.add(layerActivationValues, self.biases[i])
            layerActivationValues = self.function(layerActivationValues)
            activationValues.append(layerActivationValues)

        return activationValues[-1].transpose()[0]
    
    def predict(self, data):
        '''
        classifies an array of different data points
        Parameters
        ----------
        data : (int[][] array containing sets of input data to predict)
        Returns
        -------
        predictions : (int[] array containing classification of each data input)
        '''
        predictions = []
        rawOutput = []
        for point in data:
            current = self.predictSingle(point)[0]
            rawOutput.append(current)
            predictions.append(0 if current < 0.5 else 1)
        
        return predictions
    
def kFoldModelTest(k):
    '''
    test the network using the k-fold method
    Parameters
    ----------
    k : the number of different batches
    '''
    # get the data to train / test on
    testing = np.loadtxt(open(path + "testing.csv"), delimiter=",").astype(int)
    training = np.loadtxt(open(path + "training.csv"), delimiter = ",").astype(int)
    data = np.concatenate((testing, training))
    batches = []
    numberInBatch = (int)(data.shape[0]/k)
    for i in range(k):
        batch = []
        for j in range(numberInBatch):
            batch.append(data[i * numberInBatch + j])
        batches.append(batch)

    # create a new model
    epochs = 20
    f = lambda x : 1/(1 + np.exp(-x))
    df = lambda x: x * (1 - x)
    learningRate = 2/epochs
    numberOfInputs = 54
    numberOfHiddenNodes = [1] # only use one hidden layer - this is the most efficient
    numberOfOutputs = 1
    agent = NeuralNetwork(numberOfInputs, numberOfHiddenNodes, numberOfOutputs, (f, df), learningRate)
    
    # train the network on all batches bar 1, then test on this other batch
    average = 0
    for i in range(k): # for each batch....
        print(f"{i*100/k}% done")
        
        # get the batches to test and train with
        trainingBatches = []
        testingBatch = []        
        for j in range(k):
            if j != i:
                trainingBatches.append(batches[j])
            else:
                testingBatch = batches[j]
                
        # train with the training batches
        for j in range(epochs * (k-1) * len(testingBatch)):
            index1 = j % (k - 1)
            index2 = j % (len(trainingBatches[0]))
            inputData = trainingBatches[index1][index2]
            trainingData = inputData[1:]
            labels = inputData[:1]
            agent.train(trainingData, labels)
            agent.setLearningRate(0.9999)
        
        # test the agent against the other batch
        testingBatch = np.array(testingBatch)
        testingInputs = testingBatch[:, 1:]
        testingOutputs = testingBatch[:, 0]
        answers = agent.predict(testingInputs)
        accuracy = np.count_nonzero(answers == testingOutputs)/testingOutputs.shape[0]
        average += accuracy
    
    print(f"average accuracy was{average / k}")
