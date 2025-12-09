import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False

from opentome.models.model_flashattn import FlashAttentionModel

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--local_depth', type=int, default=4)
    parser.add_argument('--latent_depth', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def make_dataloaders(img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def main():
    args = get_args()
    device = args.device

    train_loader, test_loader = make_dataloaders(args.img_size, args.batch_size)

    depth = args.local_depth + args.latent_depth
    model = FlashAttentionModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        depth=depth,
        num_classes=10,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler('cuda') if USE_NEW_AMP and device == 'cuda' else (GradScaler() if device=='cuda' else None)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda') if USE_NEW_AMP else autocast():
                    logits, _ = model(imgs)
                    loss = criterion(logits, labels)
                if torch.isnan(loss):
                    print(f"NaN loss detected at Epoch {epoch+1}, Iteration {i}. Stopping.")
                    return
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                if torch.isnan(loss):
                    print(f"NaN loss detected at Epoch {epoch+1}, Iteration {i}. Stopping.")
                    return
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} iter {i+1} loss {loss.item():.4f}")
        print(f"Epoch {epoch+1} mean loss {running_loss / len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if scaler is not None:
                    with autocast('cuda') if USE_NEW_AMP else autocast():
                        logits, _ = model(imgs)
                else:
                    logits, _ = model(imgs)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Validation acc: {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
