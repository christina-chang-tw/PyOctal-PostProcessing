import torch
import torch.nn as nn

def main()

    model = torch.load
    model.eval()
    with torch.no_grad():
        # raw_exp should be a tensor of shape (1, 1, input_length)
        prediction_scaled = model(raw_exp.to(device))

        # Convert back to real units (C, atoms/cm2, keV)
        prediction_real = scaler.inverse_transform(prediction_scaled.cpu().numpy())

        print(f"Estimated Energy: {prediction_real[0][0]:.1f} keV")
        print(f"Estimated Temp:   {prediction_real[0][1]:.1f} C")
        print(f"Estimated Dose:   {prediction_real[0][2]:.2e} /cm2")

if __name__ == "__main__":
    main()