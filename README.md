# 🏗️ Beam Deflection Measurement Tool

This interactive Python tool allows users to **measure the deflection of a bent beam** from an image using mouse clicks, visual magnification, and calibration to physical units. It also provides a **theoretical deflection calculation** using classical beam bending equations for comparison.

---

## 📸 Features

- Manual selection of beam supports and curve points via mouse.
- Real-time magnifier window for precision clicking.
- Interpolated deflection curve plotting with dual y-axes (pixels + meters).
- Optional calibration using a known physical reference in the image.
- Theoretical deflection calculation using input beam and load parameters.
- Visual and numerical comparison between measured and theoretical results.

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- SciPy
- Image (only ```.png``` and ```.jpeg```)

Install dependencies via:

```python
pip install opencv-python numpy matplotlib scipy
```
## 📂 Folder Structure

├── BeamImages/

├── beam_deflection.py

└── README.md

## 🚀 Usage Instructions

- Place images of the bent beam inside the BeamImages/ folder.
- Run the script:
```bash
python beam_deflection.py
```
- Follow the prompts.

## 📊 Output

- A Matplotlib plot showing:
  - Deflection curve (in pixels and meters)
  - Maximum deflection value annotated

- Console prints:
  - Scaling factor (meters/pixel)
  - Maximum measured deflection
  - Theoretical deflection and error percentage (if manually calculated)

## 🤝 Contributing

Feel free to fork and improve! Open a PR for enhancements, bug fixes, or new features.
