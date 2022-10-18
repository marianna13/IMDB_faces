import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_gallery(images, h, w, n_row=3, n_col=4, gray=None, titles=None, filename='faces.png'):
    if gray:
        cmap = plt.cm.gray
        shape = (h, w)
    else:
        cmap = None
        shape = (h, w, 3)
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        img = images[i]
        img = img.reshape(shape)
        plt.subplot(n_row, n_col, i + 1)
        if not gray:
            img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(img, cmap=cmap)
        if titles:
            plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig(filename)


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


def get_vis(test_images, target_names, model, h, w):

    permutation = np.random.permutation(len(test_images))
    test_images = test_images[permutation]
    test_labels = test_labels[permutation]

    y_pred = [np.argmax(model.predict(img), axis=0)
              for img in test_images[:12]]

    prediction_titles = [
        title(y_pred, test_labels, target_names, i) for i in range(12)
    ]

    plot_gallery(test_images[:12], prediction_titles, h, w)
