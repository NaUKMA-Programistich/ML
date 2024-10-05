import math

import click
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

logistic_regression = "LogisticRegression"
knn = "KNN"
decision_tree = "DecisionTree"

simple = "Simple"
kfold = "Kfold"
stratified_kfold = "Stratified Kfold"

def convert_label(label: str) -> int:
    if label == "human":
        return 0
    elif label == "animal":
        return 1
    else:
        raise ValueError(f"Unknown label: {label}")

def convert_int_label(label: int) -> str:
    return "human" if label == 0 else "animal"

def load_data(image_folder: str, label_file: str) -> (np.array, np.array):
    """ Loads images and labels from the specified folder and file."""
    raw_data = []

    with open(label_file, mode='r') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        _ = next(reader)
        for row in reader:
            raw_data.append({"image_name": row[0], "label": row[3]})

    images = []
    labels = []

    for raw in raw_data:
        image_name = raw["image_name"]
        label_name = raw["label"]

        label = convert_label(label_name)

        image_path = image_folder+image_name
        with Image.open(image_path) as image:
            image_resized = image.resize(size=(256, 256))
            image_converted = image_resized.convert('RGB')
            image_grayscale = np.mean(np.array(image_converted), axis=2).flatten()
            images.append(image_grayscale)
            labels.append(label)

    return images, labels

# def vectorize_images(images: list) -> np.ndarray:
#     X = np.stack(images, axis=0)
#     X = X.astype('float32') / 255.0
#     return X

def vectorize_images(images: list) -> np.ndarray:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    processed_images = []
    for img_array in images:
        img = img_array.reshape(256, 256).astype('uint8')
        img = Image.fromarray(img).convert('RGB')
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = base_model.predict(x)
        processed_images.append(features.flatten())

    X = np.stack(processed_images, axis=0)
    return X


def validation_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
        stratify: bool = True
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test

def create_model(model_name: str):
    if model_name == logistic_regression:
        return LogisticRegression(
            max_iter=5000,
            C=0.1,
            random_state=42,
            class_weight='balanced'
        )
    elif model_name == knn:
        return KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='euclidean'
        )
    elif model_name == decision_tree:
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )


def train_and_evaluate(X_train, y_train, X_test, y_test, model, strategy):
    y_pred = None
    if strategy == simple:
        X_train_sub, X_val, y_train_sub, y_val = validation_split(X_train, y_train)
        model.fit(X_train_sub, y_train_sub)
        y_pred = model.predict(X_test)
    elif strategy in [kfold, stratified_kfold]:
        k = 5
        if strategy == kfold:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
        else:
            kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        scores = []
        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def error_analysis(X_test, y_test, y_pred, images, model_name, validation_strategy):
    misclassified_indices = np.where(y_test != y_pred)[0]
    num_misclassified = len(misclassified_indices)
    print(f"Number of misclassified labels: {num_misclassified}")

    cols = 5
    rows = math.ceil(num_misclassified / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(misclassified_indices):
        plt.subplot(rows, cols, i + 1)
        image = images[idx].reshape(256, 256)
        plt.imshow(image, cmap='gray')
        true_label = convert_int_label(y_test[idx])
        predicted_label = convert_int_label(y_pred[idx])
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"part2/{model_name}_{validation_strategy}_analysis.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['human', 'animal'])
    disp.plot()
    plt.savefig(f"part2/{model_name}_{validation_strategy}_confusion.png")
    plt.close()


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
@click.option("--validation_strategy", help="Validation strategy to use")
def main(image_folder: str, label_file: str, model_name: str, test_size: float, validation_strategy: str):
    main_internal(image_folder, label_file, model_name, test_size, validation_strategy)


def main_internal(image_folder: str, label_file: str, model_name: str, test_size: float, validation_strategy: str):
    # Create dataset of image <-> label pairs
    images, labels = load_data(image_folder, label_file)

    # preprocess images and labels
    X = vectorize_images(images)
    y = np.array(labels)

    # split data into train and test
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X, y, images, test_size=test_size, random_state=42, stratify=y
    )

    # create model
    model = create_model(model_name)

    # Train model using different validation strategies (refere to https://scikit-learn.org/stable/modules/cross_validation.html)
    # 1. Train, validation, test splits: so you need to split train into train and validation
    # 2. K-fold cross-validation: apply K-fold cross-validation on train data
    # 3. Leave-one-out cross-validation: apply Leave-one-out cross-validation on train data

    # Make a prediction on test data
    # Calculate accuracy
    accuracy, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, model, validation_strategy)
    print(f"Accuracy {validation_strategy} {model_name}: {accuracy:.2f}")

    # Make error analysis
    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    # 2. Plot the confusion matrix
    error_analysis(X_test, y_test, y_pred, images_test, model_name, validation_strategy)


if __name__ == "__main__":
    image_folder = "dataset/flickr30k_images/"
    label_file = "dataset/labels.csv"
    test_size = 0.2

    models = [logistic_regression, knn, decision_tree]
    validation_strategy = [simple, kfold, stratified_kfold]

    for model in models:
        for strategy in validation_strategy:
            main_internal(
                image_folder=image_folder,
                label_file=label_file,
                model_name=model,
                test_size=test_size,
                validation_strategy=strategy
            )
            print("-----")