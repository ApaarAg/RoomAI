"""Test loading MiDaS small via torch.hub."""
import torch


def main():
    try:
        print("Testing torch.hub.load for MiDaS_small...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        print("MiDaS_small loaded successfully!")
    except Exception as e:
        print(f"MiDaS load failed: {e}")


if __name__ == "__main__":
    main()

