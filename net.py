import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")


class Conv(object):

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)/9

    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape

        output = np.zeros((h-2, w-2, self.num_filters))

        for im_regions, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_regions * self.filters, axis=(1, 2))
        return output

    def backprop(self, d_l_d_out, learn_rate):

        d_l_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i, j, f] * im_region

        # update filters
        self.filters -= learn_rate * d_l_d_filters

        return None


class MaxPool(object):
    def iterate_regions(self, image):
        h, w, _ = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_l_d_out):
        d_l_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # if the pixel was the max value, copy the gradient to it
                        if(im_region[i2, j2, f2] == amax[f2]):
                            d_l_d_input[i*2+i2, j*2+j2,
                                        f2] = d_l_d_out[i, j, f2]
                            break
        return d_l_d_input


class Softmax(object):
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):

        self.last_input_shape = input.shape
        print(input.shape)

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return(exp/np.sum(exp, axis=0))

    def backprop(self, d_l_d_out, learn_rate):
        for i, gradient in enumerate(d_l_d_out):
            if(gradient == 0):
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) / (S**2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_l_d_t = gradient * d_out_d_t

            d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
            d_l_d_b = d_l_d_t * d_t_d_b
            d_l_d_inputs = d_t_d_inputs @ d_l_d_t

            self.weights -= learn_rate * d_l_d_w
            self.biases -= learn_rate * d_l_d_b
            return d_l_d_inputs.reshape(self.last_input_shape)


class ConvNet(object):
    def __init__(self, num_filters, num_classes, shape):
        self.conv = Conv(num_filters)
        self.pool = MaxPool()
        self.softmax = Softmax(shape, num_classes)

    def _forward(self, image, label):

        out = self.conv.forward(image)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
        loss = -np.log(out[label])
        acc = 1 if(np.argmax(out) == label) else 0

        return out, loss, acc

    def predict(self, image):

        out = self.conv.forward(image)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
        return out

    def _train(self, im, label, lr):

        out, loss, acc = self._forward(im, label)

        gradient = np.zeros(10)
        gradient[label] = -1/out[label]

        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, lr)

        return loss, acc

    def fit(self, train_images, train_labels, lr=0.00005, EPOCHS=1):

        self.train_images = train_images
        self.train_labels = train_labels
        self.lr = lr
        self.epochs = EPOCHS

        losses, accs = [], []

        for epoch in range(EPOCHS):
            print(f'----EPOCH {epoch+1}/{EPOCHS} ---')

            permutation = np.random.permutation(len(train_images))
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]

            loss = 0
            num_correct = 0

            for i, (im, label) in enumerate(zip(train_images, train_labels)):
                if(i > 0 and i % 100 == 99):
                    print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (
                        i + 1, loss / 100, num_correct))

                    loss = 0
                    num_correct = 0
                l, acc = self._train(im, label, lr)
                loss += l
                losses.append(loss)
                accs.append(acc)
                num_correct += acc
        hist = {
            'loss': losses,
            'acc': accs
        }
        with open('history.pkl', 'wb') as outp:
            pickle.dump(hist, outp, pickle.HIGHEST_PROTOCOL)
        return hist


def save_weights(model, file_name):
    weights = model.softmax.weights.copy()
    biases = model.softmax.biases.copy()

    with open(file_name, 'wb') as outp:
        pickle.dump({
            'weights': weights,
            'biases': biases
        }, outp, pickle.HIGHEST_PROTOCOL)


def save_model(model, file_name):
    with open(file_name, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
