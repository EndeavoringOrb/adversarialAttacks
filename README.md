## An implementation of adversarial attacks against a ML model.

### Helper files
models.py defines the PyTorch models.

data.py defines loadDataset for easy data loading.

### Main scripts
classifier.py trains a simple CNN classifier on the MNIST training data.

attacker.py finds a single 28x28 image that minimizes the accuracy of the classifier when added to the training data images.

betterClassifier.py trains the same CNN, but at the same time trains an attacker whose outputs are fed to the CNN. This makes the CNN more robust to attacks (of this specific type).

test.py evaluates the classifier on the test data, then evaluates the classifier on the attacker output to see the effect on accuracy that the attacker has. It saves images to images/ for visualization of the attack process. It also measures the average pixel distance of the attack image.