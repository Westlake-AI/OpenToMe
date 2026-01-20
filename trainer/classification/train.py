import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from timm.data import create_dataset, create_loader, resolve_data_config
from opentome.models.mergenet.model import HybridToMeModel
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--latent_depth', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--dtem_window_size', type=int, default=8)
    parser.add_argument('--tome_window_size', type=int, default=None)
    parser.add_argument('--dtem_t', type=int, default=1)
    parser.add_argument('--lambda_local', type=int, default=2)
    parser.add_argument('--merge_latent', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_softkmax', action='store_true', default=False)
    return parser.parse_args()

def make_dataloaders(img_size, batch_size):
    # 使用 timm 的 create_dataset，可以自动处理 ImageNet 的验证集格式
    dataset_train = create_dataset(
        '',  # 空字符串表示使用 ImageFolder
        root='/ssdwork/yuchang/ImageNet',
        split='train',
        is_training=True,
        batch_size=batch_size
    )
    
    dataset_eval = create_dataset(
        '',  # 空字符串表示使用 ImageFolder  
        root='/ssdwork/yuchang/ImageNet',
        split='validation',
        is_training=False,
        batch_size=batch_size
    )

    # 使用 timm 的 create_loader
    train_loader = create_loader(
        dataset_train,
        input_size=(3, img_size, img_size),
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = create_loader(
        dataset_eval,
        input_size=(3, img_size, img_size),
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, test_loader

def main():
    args = get_args()
    device = args.device

    train_loader, test_loader = make_dataloaders(args.img_size, args.batch_size)

    model = HybridToMeModel(img_size=args.img_size, patch_size=args.patch_size,
                            embed_dim=args.embed_dim, num_heads=args.num_heads,
                            mlp_ratio=args.mlp_ratio, dtem_feat_dim=64,
                            latent_depth=args.latent_depth,
                            tome_use_naive_local=False, lambda_local=args.lambda_local, total_merge_latent=args.merge_latent,
                            dtem_window_size=args.dtem_window_size, tome_window_size=args.tome_window_size, use_softkmax=args.use_softkmax)

    # DTEM 温度与超参
    model.local.vit._tome_info["k2"] = 4
    model.local.vit._tome_info["tau1"] = 1.0
    model.local.vit._tome_info["tau2"] = 0.1

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler('cuda') if USE_NEW_AMP and device == 'cuda' else (GradScaler() if device=='cuda' else None)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("\nModel architecture:")
    print(model)
    

    for epoch in range(args.epochs):
        # import pdb;pdb.set_trace()
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            # import pdb;pdb.set_trace()
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda') if USE_NEW_AMP else autocast():
                    logits, aux = model(imgs)
                    loss = criterion(logits, labels)
                if torch.isnan(loss):
                    print(f"NaN loss detected at Epoch {epoch+1}, Iteration {i}. Stopping.")
                    return
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, aux = model(imgs)
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
        loss = 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if scaler is not None:
                    with autocast('cuda') if USE_NEW_AMP else autocast():
                        logits, aux = model(imgs)
                else:
                    logits, aux = model(imgs)
                # import pdb;pdb.set_trace()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loss += criterion(logits, labels).item()
        print(f"Validation acc: {correct/total*100:.2f}%, Validation loss: {loss:.4f}")

if __name__ == "__main__":
    main()
# python /yuchang/yk/OpenToMe/trainer/classification/train.py --epochs 20 --batch_size 512 --img_size 224 --latent_depth 12 --lambda_local 4 --merge_latent 4 --use_softkmax