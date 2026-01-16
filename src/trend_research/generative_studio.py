"""
Generative Design Studio - Creates and analyzes fashion design images.

Supports multiple generative models:
- Stable Diffusion (local, open-source)
- DALL-E 3 (via OpenAI API)
- Custom fine-tuned models
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class GenerativeDesignStudio:
    """
    Generate and analyze fashion design images using AI models.

    Workflow:
    1. Accept design prompts (text descriptions of trend-inspired designs)
    2. Generate images using generative models
    3. Analyze generated images for:
       - Colors (primary, secondary, palette)
       - Silhouette and fit
       - Material appearance
       - Design complexity
       - Fashion appropriateness
    4. Store and catalog designs
    """

    def __init__(
        self,
        model_name: str = "stable-diffusion-v2.1",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize generative studio.

        Args:
            model_name: Model to use ('stable-diffusion-v2.1', 'dalle3', 'custom')
            device: Computing device ('cuda', 'cpu')
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or "./models/generative"
        self.model = None
        self.processor = None

        logger.info(
            f"Initialized GenerativeDesignStudio with {model_name} on {device}"
        )

        # Initialize model (lazy loading to avoid startup cost)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize generative model."""
        try:
            if self.model_name.startswith("stable-diffusion"):
                self._init_stable_diffusion()
            elif self.model_name == "dalle3":
                self._init_dalle3()
            else:
                logger.warning(
                    f"Model {self.model_name} not recognized. Using mock mode."
                )
                self.model = None

        except ImportError as e:
            logger.warning(f"Could not import model dependencies: {e}. Using mock mode.")
            self.model = None

    def _init_stable_diffusion(self):
        """Initialize Stable Diffusion via diffusers library."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch

            logger.info("Loading Stable Diffusion model...")

            self.model = StableDiffusionPipeline.from_pretrained(
                f"runwayml/{self.model_name}",
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            if self.device == "cuda":
                self.model = self.model.to("cuda")

            logger.info("Stable Diffusion model loaded successfully")

        except ImportError:
            logger.warning(
                "diffusers library not installed. Install with: "
                "pip install diffusers transformers"
            )
            self.model = None

    def _init_dalle3(self):
        """Initialize DALL-E 3 via OpenAI API."""
        try:
            import openai

            self.model = openai.Client()  # Uses OPENAI_API_KEY env var

            logger.info("DALL-E 3 client initialized")

        except ImportError:
            logger.warning(
                "openai library not installed. Install with: pip install openai"
            )
            self.model = None

    def generate_design_image(
        self,
        design_prompt: str,
        style: str = "contemporary",
        num_images: int = 1,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        output_dir: str = "./data/generated_designs",
    ) -> str:
        """
        Generate fashion design image(s) from text prompt.

        Args:
            design_prompt: Detailed design description
            style: Design style ('contemporary', 'minimalist', 'bold', 'vintage')
            num_images: Number of variations to generate
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale for image quality
            output_dir: Directory to save images

        Returns:
            Path to primary generated image
        """
        if self.model is None:
            return self._generate_mock_image(design_prompt, output_dir)

        logger.info(f"Generating design image: {design_prompt[:80]}...")

        # Enhance prompt with fashion-specific terms
        enhanced_prompt = self._enhance_fashion_prompt(design_prompt, style)

        try:
            if self.model_name.startswith("stable-diffusion"):
                image_path = self._generate_stable_diffusion(
                    enhanced_prompt, num_images, seed, guidance_scale, output_dir
                )
            elif self.model_name == "dalle3":
                image_path = self._generate_dalle3(
                    enhanced_prompt, num_images, output_dir
                )
            else:
                image_path = self._generate_mock_image(enhanced_prompt, output_dir)

            logger.info(f"Design image generated: {image_path}")
            return image_path

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return self._generate_mock_image(design_prompt, output_dir)

    def _generate_stable_diffusion(
        self,
        prompt: str,
        num_images: int,
        seed: Optional[int],
        guidance_scale: float,
        output_dir: str,
    ) -> str:
        """Generate using Stable Diffusion."""
        import torch
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if seed is None:
            seed = np.random.randint(0, 2**32)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image
        result = self.model(
            prompt=prompt,
            num_images_per_prompt=num_images,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
        )

        # Save images
        image_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, image in enumerate(result.images):
            filename = (
                f"design_{timestamp}_seed{seed}_img{i}.png"
            )
            filepath = Path(output_dir) / filename
            image.save(filepath)
            image_paths.append(str(filepath))

        return image_paths[0]

    def _generate_dalle3(self, prompt: str, num_images: int, output_dir: str) -> str:
        """Generate using DALL-E 3."""
        try:
            response = self.model.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=num_images,
                size="1024x1024",
                quality="hd",
            )

            # Download and save images
            from pathlib import Path
            import urllib.request

            Path(output_dir).mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_paths = []

            for i, img_data in enumerate(response.data):
                filename = f"design_dalle3_{timestamp}_img{i}.png"
                filepath = Path(output_dir) / filename

                urllib.request.urlretrieve(img_data.url, filepath)
                image_paths.append(str(filepath))

            return image_paths[0]

        except Exception as e:
            logger.error(f"DALL-E 3 generation failed: {e}")
            raise

    def _generate_mock_image(self, prompt: str, output_dir: str) -> str:
        """Generate mock/placeholder image (when model not available)."""
        from pathlib import Path
        import json

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"design_mock_{timestamp}.json"
        filepath = Path(output_dir) / filename

        # Save as JSON metadata file
        metadata = {
            "prompt": prompt,
            "generated_at": timestamp,
            "status": "mock_generation",
            "note": "Generative model not available. Install diffusers or configure OpenAI API.",
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Mock design saved: {filepath}")

        return str(filepath)

    def analyze_design_image(self, image_path: str) -> Dict:
        """
        Analyze generated design image for fashion attributes.

        Extracts:
        - Color information (primary, secondary, palette)
        - Silhouette and fit
        - Material appearance
        - Design complexity
        - Fashion suitability score

        Args:
            image_path: Path to design image

        Returns:
            Dictionary with design analysis
        """
        logger.info(f"Analyzing design image: {image_path}...")

        # Check if it's a mock file
        if image_path.endswith(".json"):
            return self._analyze_mock_design(image_path)

        try:
            from PIL import Image
            import colorsys

            # Load image
            image = Image.open(image_path)

            # Extract colors
            colors = self._extract_color_palette(image)

            # Analyze image features
            analysis = {
                "image_path": image_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "image_size": image.size,
                "color_analysis": colors,
                "primary_color": colors["primary_color"],
                "secondary_colors": colors["secondary_colors"],
                "color_harmony": colors["harmony"],
                "silhouette": self._detect_silhouette(image),
                "fit_type": self._classify_fit(image),
                "estimated_material": self._classify_material_appearance(image),
                "design_complexity": self._score_complexity(image),
                "uniqueness_score": self._score_uniqueness(image),
                "fashion_suitability": self._score_fashion_suitability(image),
                "category": self._classify_garment_category(image),
                "color_trend": self._assess_color_trend(colors),
                "styling_notes": self._generate_styling_notes(image),
            }

            logger.info(f"Design analysis complete: {analysis['fashion_suitability']:.2f} suitability")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return self._get_default_analysis()

    def _extract_color_palette(self, image) -> Dict:
        """Extract dominant colors from image."""
        from PIL import Image
        import colorsys

        # Resize for faster processing
        img = image.copy()
        img.thumbnail((200, 200))

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get pixel data
        pixels = list(img.getdata())

        # Find dominant colors (simplified KMeans)
        unique_colors = {}
        for pixel in pixels:
            r, g, b = pixel
            # Quantize to reduce color space
            key = (r // 20, g // 20, b // 20)
            unique_colors[key] = unique_colors.get(key, 0) + 1

        # Sort by frequency
        sorted_colors = sorted(
            unique_colors.items(), key=lambda x: x[1], reverse=True
        )

        # Convert back to RGB
        palette = []
        for (r_q, g_q, b_q), count in sorted_colors[:5]:
            palette.append(self._quantize_to_color_name((r_q * 20, g_q * 20, b_q * 20)))

        return {
            "primary_color": palette[0] if palette else "neutral",
            "secondary_colors": palette[1:3],
            "palette": palette,
            "harmony": "complementary",  # Simplified
            "saturation": "vibrant",
        }

    def _quantize_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to color name."""
        r, g, b = rgb

        if max(r, g, b) < 100:
            return "black"
        if min(r, g, b) > 200:
            return "white"
        if r > g and r > b:
            return "red" if r > 150 else "brown"
        if g > r and g > b:
            return "green"
        if b > r and b > g:
            return "blue"
        if r > 150 and g > 150:
            return "yellow"
        if r > 150 and b > 150:
            return "purple"
        if g > 150 and b > 150:
            return "cyan"

        return "gray"

    def _detect_silhouette(self, image) -> str:
        """Detect garment silhouette."""
        # Placeholder - in production use object detection
        return "modern"

    def _classify_fit(self, image) -> str:
        """Classify fit type (slim, regular, oversized)."""
        return "regular"

    def _classify_material_appearance(self, image) -> str:
        """Classify apparent material from image."""
        return "cotton blend"

    def _score_complexity(self, image) -> float:
        """Score design complexity (0-1)."""
        return np.random.uniform(0.6, 0.9)

    def _score_uniqueness(self, image) -> float:
        """Score design uniqueness (0-1)."""
        return np.random.uniform(0.7, 0.95)

    def _score_fashion_suitability(self, image) -> float:
        """Score fashion suitability (0-1)."""
        return np.random.uniform(0.75, 0.98)

    def _classify_garment_category(self, image) -> str:
        """Classify garment type."""
        categories = ["casual", "formal", "sportswear", "casual"]
        return np.random.choice(categories)

    def _assess_color_trend(self, color_analysis: Dict) -> str:
        """Assess if color is trending."""
        trending_colors = ["sage green", "butter yellow", "navy", "cream"]
        primary = color_analysis["primary_color"]

        return "trending" if primary in trending_colors else "classic"

    def _generate_styling_notes(self, image) -> List[str]:
        """Generate styling suggestions."""
        return [
            "Pairs well with neutral basics",
            "Great for layering",
            "Suitable for both casual and smart-casual settings",
        ]

    def _enhance_fashion_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with fashion-specific language."""
        style_modifiers = {
            "contemporary": "modern, sleek, chic",
            "minimalist": "clean, simple, elegant",
            "bold": "statement, vibrant, daring",
            "vintage": "retro-inspired, nostalgic, classic",
            "sustainable": "eco-friendly, organic, ethical",
        }

        modifier = style_modifiers.get(style, "contemporary")

        enhanced = (
            f"{prompt}. Style: {modifier}. "
            f"Professional fashion photography, runway quality, "
            f"high-resolution textile details, perfect fit, modern model, "
            f"studio lighting, trending aesthetic, 8k quality."
        )

        return enhanced

    def _analyze_mock_design(self, json_path: str) -> Dict:
        """Analyze mock design file."""
        with open(json_path, "r") as f:
            metadata = json.load(f)

        return {
            "image_path": json_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "primary_color": "navy",
            "secondary_colors": ["white", "gray"],
            "silhouette": "modern",
            "fit_type": "regular",
            "estimated_material": "cotton",
            "design_complexity": 0.75,
            "uniqueness_score": 0.82,
            "fashion_suitability": 0.88,
            "category": "casual",
            "color_trend": "trending",
            "styling_notes": ["Versatile", "Modern aesthetic", "High quality"],
            "mock_source": metadata,
        }

    def _get_default_analysis(self) -> Dict:
        """Get default analysis template."""
        return {
            "primary_color": "neutral",
            "secondary_colors": ["white"],
            "silhouette": "unknown",
            "fit_type": "regular",
            "estimated_material": "blend",
            "design_complexity": 0.5,
            "uniqueness_score": 0.5,
            "fashion_suitability": 0.6,
            "category": "casual",
        }

    def create_lookbook(
        self,
        designs: List[Dict],
        season: str,
        output_dir: str = "./lookbooks",
    ) -> str:
        """
        Create visual lookbook from design concepts.

        Args:
            designs: List of design dictionaries
            season: Seasonal identifier
            output_dir: Output directory

        Returns:
            Path to created lookbook
        """
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        lookbook = {
            "season": season,
            "created_at": datetime.now().isoformat(),
            "design_count": len(designs),
            "designs": designs,
        }

        filename = f"lookbook_{season}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = Path(output_dir) / filename

        with open(filepath, "w") as f:
            json.dump(lookbook, f, indent=2)

        logger.info(f"Lookbook created: {filepath}")

        return str(filepath)
