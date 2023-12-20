import os
import random
import cv2
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import json

# Define annotation classes
annotation_classes = ["mcp2", "pip2", "dip2", "mcp3", "pip3", "dip3", "mcp4", "pip4", "dip4", "mcp5", "pip5", "dip5"]

# Function to load dataset
def load_dataset(dataset_dir, split_ratio=0.85):
    dataset_dicts = []
    files = os.listdir(dataset_dir)
    files = [file for file in files if file.endswith('.jpg')]

    random.shuffle(files)
    num_files = len(files)
    num_test = int(num_files * split_ratio)
    print("Total number of images taken into Training dataset : {0} \n".format(num_test))

    for i, file_name in enumerate(files):
        record = {}
        image_path = os.path.join(dataset_dir, file_name)
        record["file_name"] = image_path
        record["image_id"] = i
        img = cv2.imread(image_path)

        # Check if the image was read correctly
        if img is None:
            print(f"Unable to read image: {file_name}")
            continue

        # Get image dimensions
        height, width, _ = img.shape
        record["width"] = width
        record["height"] = height

        # Read corresponding label file
        label_path = os.path.join(dataset_dir, file_name.replace('.jpg', '.txt'))
        objs = []
        try:
            with open(label_path, 'r', encoding='utf-8') as label_file:
                lines = label_file.readlines()
                for line in lines:
                    line_values = line.strip().split()
                    if len(line_values) < 4:
                        continue  # Skip this line if it doesn't contain enough values
                    category, x, y, _ = line_values[:4]

                    # Handle "q" class as requested
                    if category == "q":
                        continue

                    x, y = float(x), float(y)

                    # Check if the category is in the annotation classes
                    if category in annotation_classes:
                        # Calculate bounding box coordinates accordingly
                        bbox = [x, y, x + 5, y + 5]

                        # Convert bounding box coordinates to the BoxMode format
                        bbox_mode = BoxMode.XYXY_ABS

                        obj = {
                            "category_id": annotation_classes.index(category),
                            "bbox": bbox,
                            "bbox_mode": bbox_mode,
                        }
                        objs.append(obj)
            record["annotations"] = objs

            if i < num_test:
                dataset_dicts.append(record)

        except FileNotFoundError:
            print(f"Label file not found for {file_name}")
            continue  # Skip this image if the label file is not found

    return dataset_dicts

# Function to register custom dataset
def register_custom_dataset(dataset_name, dataset_dir):
    DatasetCatalog.register(dataset_name, lambda: load_dataset(dataset_dir))
    MetadataCatalog.get(dataset_name).set(thing_classes=annotation_classes)
    MetadataCatalog.get(dataset_name).evaluator_type = "coco"
    pass

# Visualize dataset for debugging
def visualize_dataset(dataset_dicts, save_dir="/content/drive/MyDrive/comp_vis/viz_19"):
    os.makedirs(save_dir, exist_ok=True)
    for idx, d in enumerate(random.sample(dataset_dicts, 100)):
        img = cv2.imread(d["file_name"], cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale

        if img is None:
            print(f"Unable to read image: {d['file_name']}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale image to RGB for visualization

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')  # Display the grayscale image

        if "annotations" in d:
            annos = d["annotations"]
            for anno in annos:
                bbox = anno["bbox"]
                category_id = anno["category_id"]
                label = annotation_classes[category_id]

                bbox = [int(coord) for coord in bbox]

                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0], bbox[1] - 5, label, color='green', fontsize=8, weight='bold')

        save_path = os.path.join(save_dir, f"visualization_{idx}.jpg")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# Function to train the Detectron model
def train_detectron_model(dataset_dir, yaml_file_path):
    dataset_dicts = load_dataset(dataset_dir)
    visualize_dataset(dataset_dicts)

    # Configure and train the model
    cfg = get_cfg()
    cfg.merge_from_file(yaml_file_path)
    cfg.DATASETS.TRAIN = ("Wow_keypoint_Detection",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.DATALOADER.NUM_WORKERS = 2

    # Set the output directory for trained model weights
    cfg.OUTPUT_DIR = "/content/drive/MyDrive/comp_vis/output_19"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create a trainer
    trainer = DefaultTrainer(cfg)

    # Train the model
    trainer.resume_or_load(resume=False)
    trainer.train()
    pass

# Function to load metrics line by line from a file
def load_metrics_line_by_line(file_path):
    metrics_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                metrics_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip lines that cannot be parsed as JSON
    return metrics_data

# Function to plot learning rate and different components of training loss
def plot_learning_rate_and_loss(metrics_data):
    iterations = [entry['iteration'] for entry in metrics_data if 'iteration' in entry]
    learning_rate = [entry['lr'] for entry in metrics_data if 'lr' in entry]
    total_loss = [entry['total_loss'] for entry in metrics_data if 'total_loss' in entry]
    loss_box_reg = [entry['loss_box_reg'] for entry in metrics_data if 'loss_box_reg' in entry]
    loss_cls = [entry['loss_cls'] for entry in metrics_data if 'loss_cls' in entry]
    loss_rpn_cls = [entry['loss_rpn_cls'] for entry in metrics_data if 'loss_rpn_cls' in entry]
    loss_rpn_loc = [entry['loss_rpn_loc'] for entry in metrics_data if 'loss_rpn_loc' in entry]
    
    # Creating subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plot Learning Rate
    axs[0, 0].plot(iterations, learning_rate, color='purple')
    axs[0, 0].set_title('Learning Rate over Iterations')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Learning Rate')

    # Plot Total Loss
    axs[0, 1].plot(iterations, total_loss, color='red')
    axs[0, 1].set_title('Total Loss over Iterations')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Total Loss')

    # Plot Loss Box Regression
    axs[1, 0].plot(iterations, loss_box_reg, color='blue')
    axs[1, 0].set_title('Loss Box Regression over Iterations')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Loss Box Reg')

    # Plot Loss Classification
    axs[1, 1].plot(iterations, loss_cls, color='green')
    axs[1, 1].set_title('Loss Classification over Iterations')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Loss Classification')

    # Plot Loss RPN Classification
    axs[2, 0].plot(iterations, loss_rpn_cls, color='orange')
    axs[2, 0].set_title('Loss RPN Classification over Iterations')
    axs[2, 0].set_xlabel('Iteration')
    axs[2, 0].set_ylabel('Loss RPN Classification')

    # Plot Loss RPN Localization
    axs[2, 1].plot(iterations, loss_rpn_loc, color='brown')
    axs[2, 1].set_title('Loss RPN Localization over Iterations')
    axs[2, 1].set_xlabel('Iteration')
    axs[2, 1].set_ylabel('Loss RPN Localization')

    plt.tight_layout()
    plt.show()
    pass

# Function to evaluate the trained model
def evaluate_model(trained_model, test_dataset):
    cfg = trained_model.cfg.clone()  # Clone the configuration of the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions

    predictor = DefaultPredictor(cfg)  # Create a predictor using the trained model

    evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=cfg.OUTPUT_DIR)  # Create an evaluator
    val_loader = build_detection_test_loader(cfg, test_dataset)  # Build the data loader

    # Perform evaluation
    inference_on_dataset(trained_model, val_loader, evaluator)
    pass

# Define paths and dataset configurations
dataset_name = "Wow_keypoint_Detection"
dataset_dir = "/path/to/your/dataset"  # Replace with the directory containing your dataset
yaml_file_path = "/path/to/your/config.yaml"  # Replace with the path to your configuration file
metrics_file = "/path/to/your/metrics.json"  # Replace with the path to your metrics file

# Load and register the dataset
register_custom_dataset(dataset_name, dataset_dir)

# Train the model
train_detectron_model(dataset_dir, yaml_file_path)

# Load and plot metrics
metrics_data = load_metrics_line_by_line(metrics_file)
plot_learning_rate_and_loss(metrics_data)

# Evaluate the trained model

trained_model = "/path/to/your/trained_model"
test_dataset_name = "/path/to/your/test_dataset_name"

evaluate_model(trained_model, test_dataset_name)
