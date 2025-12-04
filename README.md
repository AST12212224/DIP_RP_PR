# Pixel Rearrangement Art Generator  
Digital Image Processing Mini Project  
**Author:** Aadesh Tikhe  
**Enrolment No:** A71004823070  
**Assisted By:** Dr. C Kalpana  
**Amity University Mumbai**

---

## Overview
This project transforms one image into another by **rearranging pixels** based on grayscale intensity ranking.  
Unlike traditional morphing, every pixel **keeps its original RGB color**, only its **position changes**, producing a unique artistic effect.

---

## Features
- Grayscale intensity extraction  
- Pixel sorting and intensity-based mapping  
- Final reconstructed rearranged image  
- Optional pixel migration animation  
- Contour detection on 5+ images  
- Harris corner detection analysis  
- Histogram comparison (source vs rearranged)

---

## Tech Stack
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

---

## Folder Structure
```

/src
preprocess.py
pixel_sort.py
rearrange.py
contour_detect.py
corner_detect.py
animation.py

/images
source/
destination/
output/

/notebooks
experiments.ipynb

README.md

````

---

## How It Works
1. Load source and destination images  
2. Convert both to grayscale  
3. Extract all pixels with positions and intensity  
4. Sort source pixels and destination positions separately  
5. Map i-th source pixel to i-th destination pixel slot  
6. Rebuild the final image using original RGB values  

---

## Example Usage
```python
from pixel_sort import sort_pixels
from rearrange import rearrange_pixels

src = "images/source/a.jpg"
dst = "images/destination/b.jpg"

output = rearrange_pixels(src, dst)
````

---

## Outputs Generated

* Rearranged artistic image
* Contour overlays using CHAIN_APPROX_NONE & CHAIN_APPROX_SIMPLE
* Harris corner visualizations
* Pixel-movement animation (optional)

---

## Future Enhancements

* Region-wise adaptive sorting
* Live interactive GUI
* Video support (frame-by-frame rearrangement)
* Mobile application version

---

## License

Free for academic and personal use.
