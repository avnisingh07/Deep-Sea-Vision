import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms
from model import GeneratorUSRGAN, Discriminator
from dataloader import train_loader
import os

# ✅ Set CUDA optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Training on: {device}")

# ✅ Improved Perceptual Loss (Fixed Normalization)
class PerceptualLoss(nn.Module):
    def __init__(self, weight_path="/kaggle/input/vgg19-weights/vgg19-dcbb9e9d.pth"):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19()
        vgg.load_state_dict(torch.load(weight_path, weights_only=True))
        vgg = vgg.features[:16].eval().to(device)

        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG-19 weights

        self.vgg = vgg
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = self.normalize(x)  # Normalize inputs
        y = self.normalize(y)
        return torch.nn.functional.mse_loss(self.vgg(x), self.vgg(y))

# ✅ Train Function
def train():
    generator = GeneratorUSRGAN().to(device)
    discriminator = Discriminator().to(device)

    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_content = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # ✅ Learning Rate Scheduler
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)

    scaler = torch.amp.GradScaler()

    num_epochs = 50
    accumulation_steps = 4  
    save_dir = "/kaggle/working/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_g_loss = float("inf")

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()  # ✅ Prevents memory fragmentation
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            low_res, high_res = low_res.to(device), high_res.to(device)

            # ✅ Generator Training (Use  Automatic Mixed Precision (AMP)) - reduce memory usage
            optimizer_g.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                fake_high_res = generator(low_res)
                content_loss = criterion_content(fake_high_res, high_res)
                perceptual_loss_val = perceptual_loss(fake_high_res, high_res)
                fake_pred = discriminator(fake_high_res)
                adversarial_loss = criterion_gan(fake_pred, torch.ones_like(fake_pred).to(device))

                g_loss = content_loss + 0.1 * perceptual_loss_val + 0.01 * adversarial_loss
                g_loss /= accumulation_steps  

            scaler.scale(g_loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer_g)
                scaler.update()
                optimizer_g.zero_grad()

            epoch_g_loss += g_loss.item()

            # ✅ Discriminator Training (With AMP Scaling)
            optimizer_d.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                real_pred = discriminator(high_res)
                fake_pred = discriminator(fake_high_res.detach().float())  # Convert to float32

                d_loss_real = criterion_gan(real_pred, torch.ones_like(real_pred).to(device))
                d_loss_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred).to(device))
                d_loss = (d_loss_real + d_loss_fake) / 2  

            scaler.scale(d_loss).backward()  
            scaler.step(optimizer_d)
            scaler.update()

            epoch_d_loss += d_loss.item()

        # ✅ Update Learning Rate Schedulers
        scheduler_g.step()
        scheduler_d.step()

        print(f"✅ Epoch {epoch+1} | G Loss: {epoch_g_loss:.4f} | D Loss: {epoch_d_loss:.4f} | LR: {scheduler_g.get_last_lr()[0]:.6f}")
        
        # ✅ Save Only the Best Model
        if epoch_g_loss < best_g_loss:
            best_g_loss = epoch_g_loss
            torch.save(generator.state_dict(), f"{save_dir}/best_generator.pth")
            torch.save(discriminator.state_dict(), f"{save_dir}/best_discriminator.pth")
            print(f"💾 Best model saved at epoch {epoch+1}!")

if __name__ == "__main__":
    train()
