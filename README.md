# LumenCalculation
Vessel lumen area calculation based on optical offset detected on catheter sheath border.
Sheath border is detected on the thresholded probability map from the output of the U-Net segmentation model inference.
Vessel lumen area is adjusted based on the discrepancy between physical location and the apparent location of the sheath border, and verified on phantoms with known dimentions.
