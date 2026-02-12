import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict
import torch.nn.functional as F

class MinecraftTextureDataset(Dataset):

    def __init__(self, root_dir: str, texture_type: str = 'blocks'):
        
        self.root_dir = Path(root_dir) / 'training_data' / texture_type
        self.texture_type = texture_type
        self.images = []
        self.categories = []
        self.category_to_idx = {}

        if self.root_dir.exists():
            for idx, category_dir in enumerate(sorted(self.root_dir.iterdir())):
                if category_dir.is_dir():
                    category_name = category_dir.name
                    self.category_to_idx[category_name] = idx

                    for img_path in category_dir.glob('*.png'):
                        self.images.append(img_path)
                        self.categories.append(idx)
        
        print(f"Loaded {len(self.images)} {texture_type} across {len(self.category_to_idx)} categories")
        print(f"Categories: {list(self.category_to_idx.keys())}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        category_idx = self.categories[idx]

        img = Image.open(img_path).convert('RGBA')
        img = img.resize((16, 16), Image.NEAREST)  # Ensure 16x16

        img_array = np.array(img, dtype=np.float32) / 255.0

        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4]

        grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        grayscale = grayscale[:, :, np.newaxis]

        grayscale_alpha = np.concatenate([grayscale, alpha], axis=2)

        grayscale_tensor = torch.from_numpy(grayscale_alpha.transpose(2, 0, 1))
        color_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        return {
            'grayscale': grayscale_tensor,
            'color': color_tensor,
            'category': category_idx,
            'path': str(img_path)
        }

class ShadingGenerator(nn.Module):

    def __init__(self, latent_dim=64, num_categories=10):
        super(ShadingGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.category_embed = nn.Embedding(num_categories, 32)

        combined_dim = latent_dim + 32

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 4 * 4 * 64),
            nn.BatchNorm1d(4 * 4 * 64),
            nn.LeakyReLU(0.2)
        )

        self.conv = nn.Sequential(

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 2, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, noise, category):

        cat_embed = self.category_embed(category)

        x = torch.cat([noise, cat_embed], dim=1)

        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        x = self.conv(x)
        
        return x

class ShadingDiscriminator(nn.Module):

    def __init__(self, num_categories=10):
        super(ShadingDiscriminator, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(2, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Colorizer(nn.Module):

    def __init__(self, num_categories=10):
        super(Colorizer, self).__init__()

        self.category_embed = nn.Embedding(num_categories, 32)

        self.conv = nn.Sequential(

            nn.Conv2d(2, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

        self.color_adjust = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),
            nn.Tanh()
        )
    
    def forward(self, grayscale_alpha, category):
        batch_size = grayscale_alpha.size(0)

        rgb = self.conv(grayscale_alpha)

        cat_embed = self.category_embed(category)
        color_bias = self.color_adjust(cat_embed).view(batch_size, 3, 1, 1)

        rgb = torch.clamp(rgb + 0.2 * color_bias, 0, 1)
        
        return rgb

class MinecraftTextureAI:

    def __init__(self, root_dir: str, device=None):
        self.root_dir = Path(root_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.models_dir = self.root_dir / 'models'
        self.output_dir = self.root_dir / 'output'
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.category_maps = {
            'blocks': {},
            'items': {}
        }

        self.generators = {}
        self.colorizers = {}
    
    def train_shading_model(self, texture_type: str, epochs: int = 1000, batch_size: int = 16):

        print(f"\n{'='*60}")
        print(f"Training {texture_type} shading model")
        print(f"{'='*60}\n")

        dataset = MinecraftTextureDataset(self.root_dir, texture_type)
        
        if len(dataset) == 0:
            print(f"No training data found for {texture_type}!")
            return

        self.category_maps[texture_type] = dataset.category_to_idx
        num_categories = len(dataset.category_to_idx)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        generator = ShadingGenerator(num_categories=num_categories).to(self.device)
        discriminator = ShadingDiscriminator(num_categories=num_categories).to(self.device)

        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                real_imgs = batch['grayscale'].to(self.device)
                categories = batch['category'].to(self.device)
                batch_size = real_imgs.size(0)

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                d_optimizer.zero_grad()

                real_output = discriminator(real_imgs)
                d_loss_real = criterion(real_output, real_labels)

                noise = torch.randn(batch_size, generator.latent_dim).to(self.device)
                fake_imgs = generator(noise, categories)
                fake_output = discriminator(fake_imgs.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, generator.latent_dim).to(self.device)
                fake_imgs = generator(noise, categories)
                fake_output = discriminator(fake_imgs)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        self.generators[texture_type] = generator
        torch.save({
            'generator': generator.state_dict(),
            'category_map': self.category_maps[texture_type]
        }, self.models_dir / f'{texture_type}_shading.pth')
        
        print(f"\n✓ {texture_type.capitalize()} shading model trained and saved!")
    
    def train_colorizer(self, texture_type: str, epochs: int = 500, batch_size: int = 16):

        print(f"\n{'='*60}")
        print(f"Training {texture_type} colorizer")
        print(f"{'='*60}\n")

        dataset = MinecraftTextureDataset(self.root_dir, texture_type)
        
        if len(dataset) == 0:
            print(f"No training data found for {texture_type}!")
            return
        
        num_categories = len(dataset.category_to_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        colorizer = Colorizer(num_categories=num_categories).to(self.device)
        optimizer = optim.Adam(colorizer.parameters(), lr=0.0002)

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                grayscale = batch['grayscale'].to(self.device)
                color = batch['color'].to(self.device)
                categories = batch['category'].to(self.device)
                
                optimizer.zero_grad()

                predicted_rgb = colorizer(grayscale, categories)
                target_rgb = color[:, :3, :, :]  # Extract RGB channels

                loss = criterion(predicted_rgb, target_rgb)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        self.colorizers[texture_type] = colorizer
        torch.save({
            'colorizer': colorizer.state_dict(),
            'category_map': self.category_maps[texture_type]
        }, self.models_dir / f'{texture_type}_colorizer.pth')
        
        print(f"\n✓ {texture_type.capitalize()} colorizer trained and saved!")
    
    def load_models(self, texture_type: str):

        shading_path = self.models_dir / f'{texture_type}_shading.pth'
        if shading_path.exists():
            checkpoint = torch.load(shading_path, map_location=self.device)
            self.category_maps[texture_type] = checkpoint['category_map']
            num_categories = len(self.category_maps[texture_type])
            
            generator = ShadingGenerator(num_categories=num_categories).to(self.device)
            generator.load_state_dict(checkpoint['generator'])
            generator.eval()
            self.generators[texture_type] = generator
            print(f"✓ Loaded {texture_type} shading model")
        else:
            print(f"⚠ No shading model found for {texture_type}")

        colorizer_path = self.models_dir / f'{texture_type}_colorizer.pth'
        if colorizer_path.exists():
            checkpoint = torch.load(colorizer_path, map_location=self.device)
            num_categories = len(checkpoint['category_map'])
            
            colorizer = Colorizer(num_categories=num_categories).to(self.device)
            colorizer.load_state_dict(checkpoint['colorizer'])
            colorizer.eval()
            self.colorizers[texture_type] = colorizer
            print(f"✓ Loaded {texture_type} colorizer")
        else:
            print(f"⚠ No colorizer found for {texture_type}")
    
    def generate_texture(self, texture_type: str, category: str, output_name: str, seed: int = None, 
                        smoothness: float = 0.0, tile_seamless: bool = False) -> str:

        if texture_type not in self.generators:
            print(f"Loading {texture_type} models...")
            self.load_models(texture_type)
        
        if texture_type not in self.generators:
            raise ValueError(f"No trained model for {texture_type}. Train first!")

        category_map = self.category_maps[texture_type]
        if category not in category_map:
            print(f"Available categories: {list(category_map.keys())}")
            raise ValueError(f"Unknown category: {category}")
        
        category_idx = category_map[category]

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        generator = self.generators[texture_type]
        with torch.no_grad():
            noise = torch.randn(1, generator.latent_dim).to(self.device)
            category_tensor = torch.tensor([category_idx]).to(self.device)
            
            generated = generator(noise, category_tensor)

            img_array = generated[0].cpu().numpy().transpose(1, 2, 0)

            grayscale = img_array[:, :, 0]
            alpha = img_array[:, :, 1]

            if smoothness > 0:
                from scipy.ndimage import gaussian_filter

                sigma = smoothness * 1.5
                grayscale = gaussian_filter(grayscale, sigma=sigma)
                grayscale = np.clip(grayscale, 0, 1)

            if tile_seamless:
                grayscale = self._make_seamless(grayscale)

            grayscale = (grayscale * 255).astype(np.uint8)
            alpha = (alpha * 255).astype(np.uint8)

            rgba = np.stack([grayscale, grayscale, grayscale, alpha], axis=2)

            output_path = self.output_dir / f"{output_name}_shaded.png"
            Image.fromarray(rgba).save(output_path)
            
            print(f"✓ Generated shaded texture: {output_path}")
            return str(output_path)
    
    def _make_seamless(self, img_array):
        
        h, w = img_array.shape
        blend_width = max(1, h // 4)  # Blend 25% of edges

        blend_mask_h = np.linspace(0, 1, blend_width)
        blend_mask_v = np.linspace(0, 1, blend_width)

        for i in range(blend_width):
            weight = blend_mask_h[i]
            img_array[i, :] = img_array[i, :] * weight + img_array[-(blend_width-i), :] * (1 - weight)
            img_array[-(i+1), :] = img_array[-(i+1), :] * weight + img_array[blend_width-i-1, :] * (1 - weight)

        for i in range(blend_width):
            weight = blend_mask_v[i]
            img_array[:, i] = img_array[:, i] * weight + img_array[:, -(blend_width-i)] * (1 - weight)
            img_array[:, -(i+1)] = img_array[:, -(i+1)] * weight + img_array[:, blend_width-i-1] * (1 - weight)
        
        return img_array
    
    def colorize_texture(self, shaded_texture_path: str, texture_type: str, category: str, output_name: str = None) -> str:

        if texture_type not in self.colorizers:
            print(f"Loading {texture_type} colorizer...")
            self.load_models(texture_type)
        
        if texture_type not in self.colorizers:
            raise ValueError(f"No trained colorizer for {texture_type}. Train first!")

        img = Image.open(shaded_texture_path).convert('RGBA')
        img = img.resize((16, 16), Image.NEAREST)
        img_array = np.array(img, dtype=np.float32) / 255.0

        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4]
        grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        grayscale = grayscale[:, :, np.newaxis]

        grayscale_alpha = np.concatenate([grayscale, alpha], axis=2)
        grayscale_tensor = torch.from_numpy(grayscale_alpha.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        category_map = self.category_maps[texture_type]
        if category not in category_map:
            raise ValueError(f"Unknown category: {category}")
        
        category_idx = category_map[category]
        category_tensor = torch.tensor([category_idx]).to(self.device)

        colorizer = self.colorizers[texture_type]
        with torch.no_grad():
            rgb_out = colorizer(grayscale_tensor, category_tensor)

            rgb_array = rgb_out[0].cpu().numpy().transpose(1, 2, 0)
            alpha_array = alpha

            rgba = np.concatenate([rgb_array, alpha_array], axis=2)
            rgba = (rgba * 255).astype(np.uint8)

            if output_name is None:
                output_name = Path(shaded_texture_path).stem.replace('_shaded', '_colored')
            
            output_path = self.output_dir / f"{output_name}.png"
            Image.fromarray(rgba).save(output_path)
            
            print(f"✓ Colorized texture: {output_path}")
            return str(output_path)

if __name__ == "__main__":

    script_dir = Path(__file__).parent.resolve()
    ai = MinecraftTextureAI(str(script_dir))
    
    print()

    blocks_dir = script_dir / 'training_data' / 'blocks'
    items_dir = script_dir / 'training_data' / 'items'
    
    has_blocks = blocks_dir.exists() and any(blocks_dir.iterdir())
    has_items = items_dir.exists() and any(items_dir.iterdir())
    
    if not has_blocks and not has_items:
        print("⚠ No training data found! Please add PNG files to training_data/blocks/ or training_data/items/")
    else:

        if has_blocks:
            ai.train_shading_model('blocks', epochs=500)
            ai.train_colorizer('blocks', epochs=300)
        
        if has_items:
            ai.train_shading_model('items', epochs=500)
            ai.train_colorizer('items', epochs=300)