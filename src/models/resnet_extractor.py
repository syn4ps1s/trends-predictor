"""ResNet Feature Extractor for Collection Images."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import logging
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class ResNetFeatureExtractor:
    """
    Extract feature embeddings from fashion collection images using ResNet.

    Converts high-dimensional image data into compact embeddings suitable
    for integration with DeepAR+ forecasting pipeline.
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 256,
        freeze_encoder: bool = True,
        input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize ResNet feature extractor.

        Args:
            architecture: ResNet architecture ('resnet50', 'resnet101', etc.)
            pretrained: Whether to use ImageNet pretrained weights
            output_dim: Dimension of output embeddings
            freeze_encoder: Whether to freeze encoder weights
            input_size: Input image size (height, width)
        """
        self.architecture = architecture
        self.output_dim = output_dim
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet
        if architecture == "resnet50":
            self.encoder = models.resnet50(pretrained=pretrained)
        elif architecture == "resnet101":
            self.encoder = models.resnet101(pretrained=pretrained)
        elif architecture == "resnet34":
            self.encoder = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Remove classification head
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        # Add projection head for embedding dimension
        encoder_out_dim = 2048  # ResNet50/101 output dimension
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.model = nn.Sequential(self.encoder, nn.Flatten(), self.projection)
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Initialized ResNet{architecture} feature extractor with output_dim={output_dim}"
        )

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract embeddings from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Feature embedding array (output_dim,)
        """
        image = self._load_and_preprocess_image(image_path)
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image_tensor)

        return embedding.squeeze(0).cpu().numpy()

    def extract_batch_features(self, image_paths: list) -> np.ndarray:
        """
        Extract embeddings from multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            Feature embedding array (num_images, output_dim)
        """
        embeddings = []

        for image_path in image_paths:
            try:
                embedding = self.extract_features(image_path)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {e}")
                # Return zero embedding on error
                embeddings.append(np.zeros(self.output_dim))

        return np.array(embeddings)

    def aggregate_collection_embeddings(
        self, image_paths: list, method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate embeddings from multiple images (e.g., all images in a collection).

        Args:
            image_paths: List of image paths for collection
            method: Aggregation method ('mean', 'max', 'weighted')

        Returns:
            Single aggregated embedding
        """
        embeddings = self.extract_batch_features(image_paths)

        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "max":
            return np.max(embeddings, axis=0)
        elif method == "weighted":
            # Weight by image index (newer images higher weight)
            weights = np.linspace(0.5, 1.5, len(embeddings))
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def compute_collection_similarity(
        self, embeddings_1: np.ndarray, embeddings_2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two collections.

        Args:
            embeddings_1: Embedding of first collection
            embeddings_2: Embedding of second collection

        Returns:
            Similarity score [0, 1]
        """
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity([embeddings_1], [embeddings_2])[0, 0]

    def detect_style_shift(
        self, collection_embeddings: list, threshold: float = 0.7
    ) -> list:
        """
        Detect significant style shifts between consecutive collections.

        Args:
            collection_embeddings: List of collection embeddings
            threshold: Similarity threshold for detecting shift

        Returns:
            List of indices where style shifts detected
        """
        shifts = []

        for i in range(1, len(collection_embeddings)):
            similarity = self.compute_collection_similarity(
                collection_embeddings[i - 1], collection_embeddings[i]
            )

            if similarity < threshold:
                shifts.append(i)
                logger.info(f"Style shift detected at collection {i} (similarity={similarity:.3f})")

        return shifts

    @staticmethod
    def _load_and_preprocess_image(image_path: str) -> np.ndarray:
        """
        Load and preprocess image for ResNet.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array (3, 224, 224)
        """
        from torchvision import transforms

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocessing pipeline
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        return preprocess(image).numpy()

    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
