import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow_probability import distributions as dist
import os
import matplotlib.pyplot as plt

class CharDataset():

    def __init__(self, datasetOfWords, uniqueChars, maxWordLength, context = 3):
        assert type(datasetOfWords) is list, "datasetOfWords needs to be a list of words"
        assert type(uniqueChars) is list or type(uniqueChars) is str, "uniqueChars must be a string of characters or list of characters"
        assert type(maxWordLength) is int, "maxWordLength must be an integer"
        assert type(context) is int, "context must be an integer. It defines the number of characters to preditct the next one"
        self.words = datasetOfWords
        self.chars = uniqueChars
        self.maxWordLength = maxWordLength
        self.context = context
        self.word2index = {w:(i + 1) for i, w in enumerate(uniqueChars)}
        self.index2word = {i:w for w, i in self.word2index.items()}
    def __len__(self):
        return len(self.words)

    def getVocabularySize(self):
        return len(self.chars)

    def getOutputLength(self):
        return self.maxWordLength + 1  #Plus one special character token < Start >

    def encode(self, word):
        return np.array([self.word2index[ch] for ch in word], dtype=np.int16)

    def decode(self, indexes):
        word = ''.join(self.index2word[idx] for idx in indexes)
        return word

    def getTrainAndTestDataFromSignleWord(self, word):
        x, y = [], []
        batchContext = [0] * self.context
        wordEncoded = np.concatenate([self.encode(word), [0]])
        for i in range(len(wordEncoded)):
            x.append(batchContext)
            y.append(wordEncoded[i])
            batchContext = batchContext[1:] + [y[i]]
        return x, y

    def __getitem__(self, index):
        x, y = [], []
        if isinstance(index, int):
            word = self.words[index]
            x, y = self.getTrainAndTestDataFromSignleWord(word)
            return x, y
        elif isinstance(index, slice):
            for i in range(*index.indices(len(self))):
                xx, yy = self.getTrainAndTestDataFromSignleWord(self.words[i])
                x += xx
                y += yy
            return x, y
        else:
            raise TypeError("Invalid argument type.")

def CreateDatasets(filePath):
    assert type(filePath) is str, "Only strings acceptable"
    with open(filePath, "r") as text:
        words = text.readlines()
    words = [word.strip() for word in words]    #get rid of any leading or trailing white space
    words = [word for word in words if word]    #get rid of empty string
    words = [word.lower() for word in words]    #lowering all the characters
    #Creating distinct characters from all the words
    chars = sorted(set("".join(words)))
    maxWordLength = max(len(word) for word in words)
    print(f"Number of words in the whole dataset: {len(words)}")
    print(f"Max length word: {maxWordLength}")
    print(f"number of unique characters in the whole dataset: {len(chars)}")
    print(f"Vocabulary: {''.join(chars)}")
    np.random.shuffle(words)
    #Splitting dataset for training, developing and testing sets 80%, 10%, 10%
    numberOfTrainData = np.int32((np.floor(0.8*len(words))))
    numberOfDevData = np.int32(np.ceil(0.1*len(words)))
    trainWords = words[:numberOfTrainData]
    devWords = words[numberOfTrainData:numberOfTrainData + numberOfDevData]
    testWords = words[numberOfTrainData + numberOfDevData:]
    print(f"train, dev and test together length: {len(trainWords) + len(devWords) + len(testWords)}")
    print(f"Train dataset contains {len(trainWords)} elements\n"
          f"Development dataset contains {len(devWords)} elements\n"
          f"Test dataset contains {len(testWords)} elements")
    trainWords = CharDataset(trainWords, chars, maxWordLength, context=4)
    devWords = CharDataset(devWords, chars, maxWordLength, context=4)
    testWords = CharDataset(testWords, chars, maxWordLength, context=4)
    return trainWords, devWords, testWords

def CreateModel(vocabularySize:int, contextValue:int):
    embeddingSize = 10
    initializer = tf.keras.initializers.random_uniform(minval=-1, maxval=1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocabularySize, output_dim=embeddingSize, input_length=contextValue, embeddings_initializer=initializer))
    model.add(tf.keras.layers.Reshape((contextValue*embeddingSize,)))
    model.add(tf.keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer, bias_initializer=initializer))
    model.add(tf.keras.layers.Dense(vocabularySize, activation="softmax", kernel_initializer=initializer, bias_initializer=initializer))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model

def CreateModel2(vocabularySize:int, contextValue:int):
    embeddingSize = 10
    initializer = tf.keras.initializers.random_uniform(minval=-1, maxval=1)
    inputs = tf.keras.Input((None, contextValue))
    layer1 = tf.keras.layers.Embedding(input_dim=vocabularySize, output_dim=embeddingSize, embeddings_initializer=initializer, input_length=contextValue)(inputs)
    layer2 = tf.keras.layers.Reshape((contextValue*embeddingSize,))(layer1)
    layer3 = tf.keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer, bias_initializer=initializer)(layer2)
    output = tf.keras.layers.Dense(vocabularySize, activation="softmax", kernel_initializer=initializer,
                                   bias_initializer=initializer)(layer3)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def EvalModel(model, xSamples, ySamples, should_print = True):
    if isinstance(xSamples, np.ndarray):
        raise("Wrong input type, should be Dataset or numpy array")
    elif isinstance(xSamples, tf.data.Dataset):
        xSamples = xSamples.as_numpy_iterator()
        ySamples = ySamples.as_numpy_iterator()
    else:
        xSamples = np.array(xSamples)
        ySamples = np.array(ySamples)
    ypred = model(xSamples)
    loss = tf.keras.losses.categorical_crossentropy(ySamples, ypred)
    if should_print == True:
        print(f"loss: {np.mean(loss)}")
    return np.mean(loss)

#variable which decides should the training be continued
shouldTrain = False

checkpointPath = "./checkpoints/cp.cpkt"
checkpointPathDir = os.path.dirname(checkpointPath)


filePath = "./names.txt"
saveFilePathForCustomModel = "./saveCustomModel/cp.cpkt"

trainWords, developmentWords, testWords = CreateDatasets(filePath)
model = CreateModel2(trainWords.getVocabularySize() + 1, trainWords.context)

labelOneHotEncodingLayer = tf.keras.layers.CategoryEncoding(trainWords.getVocabularySize() + 1, output_mode="one_hot")
trainSamples, trainLabels = trainWords[:]
devSamples, devLabels = developmentWords[:]
testSamples, testLabels = testWords[:]

if shouldTrain == True:
    # try:
    #     model = tf.keras.models.load_model("./model")
    #     model.load_weights(checkpointPath)
    # except:
    #     model = CreateModel(vocabularySize=trainWords.getVocabularySize() + 1, contextValue=trainWords.context)
    #     model.save("./model", overwrite=True, save_format="tf")

    # result = model.evaluate(x=np.array(devSamples, dtype=np.int16), y=np.array(labelOneHotEncodingLayer(devLabels), dtype=np.int16), verbose=1)
    # print("Validation before fitting the model")
    # print("%s: %.6f" % (model.metrics_names[0], result[0]))
    # print("%s: %.2f%%" % (model.metrics_names[1], result[1]*100))

    #checkpoint callback created
    callbackCheckpoint = tf.keras.callbacks.ModelCheckpoint(checkpointPath, save_weights_only=True)
    trainDataset = tf.data.Dataset.from_tensor_slices((trainSamples, labelOneHotEncodingLayer(trainLabels)))

    # dataset = trainDataset.shuffle(50000).batch(64)

    #custom training loop
    batchNumber = 128
    learningRate = tf.keras.optimizers.schedules.ExponentialDecay(0.0032, 60000, 1/200)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)

    lossData = []
    learningRateData = []
    # try:
    #     model.load_weights(saveFilePathForCustomModel)
    # except:
    #     pass
    print("Evaluation model on the training data")
    EvalModel(model, trainSamples, labelOneHotEncodingLayer(trainLabels))
    print("Evaluation model on the validation data")
    EvalModel(model, devSamples, labelOneHotEncodingLayer(devLabels))
    prevValidatingLoss = [100., 10.]
    validatingLoss = 0

    for _ in range(60000):
        trainData = trainDataset.shuffle(50000).batch(batchNumber).take(1)
        x, y = [], []

        for data, label in trainData.unbatch():
            x.append(list(data.numpy()))
            y.append(list(label.numpy()))
        x = np.array(x)
        y = np.array(y)

        with tf.GradientTape() as tape:
            #forward pass
            logits = model(x)
            #Calculating loss value for this batch
            loss = tf.keras.losses.categorical_crossentropy(y, logits)
            # lossData.append(EvalModel(model, x, y, should_print=False))
        #Getting gradients of loss

        # learningRateData.append(optimizer._decayed_lr("float32").numpy())

        gradients = tape.gradient(loss, model.trainable_weights)

        #Update the wights of the model
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        validatingLoss = EvalModel(model, devSamples, labelOneHotEncodingLayer(devLabels), should_print=False)
        # if _ > 10000:
        #     if validatingLoss > np.max(prevValidatingLoss):
        #         print(_)
        #         break
        #     if len(prevValidatingLoss) < 200:
        #         prevValidatingLoss.append(validatingLoss)
        #     else:
        #         prevValidatingLoss = prevValidatingLoss[1:]
        #         prevValidatingLoss.append(validatingLoss)

        if _ % 5000 == 0:
            print("Train dataset loss: {}\nValidation dataset loss: {}".format(
                EvalModel(model, trainSamples, labelOneHotEncodingLayer(trainLabels), should_print=False),
                EvalModel(model, devSamples, labelOneHotEncodingLayer(devLabels), should_print=False)
            ))


    print("Model evaluation after fitting")
    print("Evaluation model on the training data")
    EvalModel(model, trainSamples, labelOneHotEncodingLayer(trainLabels))
    print("Evaluation model on the validation data")
    EvalModel(model, devSamples, labelOneHotEncodingLayer(devLabels))
    model.save_weights(saveFilePathForCustomModel)

    print("Model testing on the test samples:")
    EvalModel(model, testSamples, labelOneHotEncodingLayer(testLabels))
else:
    model.load_weights(saveFilePathForCustomModel)
    print("Model evaluation on train data")
    EvalModel(model, trainSamples, labelOneHotEncodingLayer(trainLabels))
    print("Model evaluation on development data")
    EvalModel(model, devSamples, labelOneHotEncodingLayer(devLabels))
    print("Model evaluation on test data")
    EvalModel(model, testSamples, labelOneHotEncodingLayer(testLabels))
    rngSeed = tf.random.set_seed(2147483656)

    print("Creating new names list:")
    for _ in range(100):
        startingString = [0] * testWords.context  # number has to be the same as context
        forPredictionHoldingString = np.array(startingString)
        newName = []
        while True:
            newName.append(np.argmax(
                tfp.distributions.Multinomial(total_count=1, probs=model(forPredictionHoldingString.reshape((1, 1, len(startingString))))).sample()[0].numpy()))
            if newName[-1] == 0:
                newName = newName[:len(newName) - 1]
                break
            forPredictionHoldingString = np.concatenate([forPredictionHoldingString[1:], [newName[-1]]])
        print(testWords.decode(newName))

# plt.plot(learningRateData, lossData)
# plt.show()

# model.fit(trainDataset.batch(64), epochs=10, verbose=1, callbacks=[callbackCheckpoint])
# model.fit(trainSamples, labelOneHotEncodingLayer(trainLabels), batch_size=32, epochs=100, verbose=1, callbacks=[callbackCheckpoint])

# result = model.evaluate(x=np.array(devSamples, dtype=np.int16), y=np.array(labelOneHotEncodingLayer(devLabels), dtype=np.int16), verbose=1)
# print("Validation after fitting the model")
# print("%s: %.6f" % (model.metrics_names[0], result[0]))
# print("%s: %.2f%%" % (model.metrics_names[1], result[1]*100))