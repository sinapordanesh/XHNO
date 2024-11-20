# 2D Hilbert Transform Kernel Evaluation

This repository extends the custom implementation of the Hilbert Transform kernel to 2D using PyTorch. The kernel is validated through multiple analytical and numerical test cases to ensure accuracy and reliability.

---

## Overview

The 2D Hilbert Transform applies phase shifting to multidimensional signals in the frequency domain. This project evaluates the custom kernel's performance using:
- Analytical solutions for known 2D functions.
- Numerical comparisons with classical methods.

Key functionalities include:
1. **2D Hilbert Transform Implementation**: PyTorch-based kernel for multidimensional signals.
2. **Analytical Validation**: Comparison with derived 2D solutions.
3. **Numerical Validation**: Comparison with SciPy's Hilbert transform.

---

## Test Cases and Validation

We designed six test cases using well-known mathematical properties of the Hilbert Transform in 2D.

### **Test Case 1: \( \sin(kx) \cdot \cos(ly) \)**

- **Input Signal**: \( f(x, y) = \sin(kx) \cdot \cos(ly) \)
- **Expected Analytical Result**: 
  \[
  H_x(f) = -\cos(kx) \cdot \cos(ly), \quad H_y(f) = \sin(kx) \cdot \sin(ly)
  \]
- **Validation**:
  - Compare the PyTorch implementation against the analytical result.

---

### **Test Case 2: \( \cos(kx) \cdot \sin(ly) \)**

- **Input Signal**: \( f(x, y) = \cos(kx) \cdot \sin(ly) \)
- **Expected Analytical Result**: 
  \[
  H_x(f) = -\sin(kx) \cdot \sin(ly), \quad H_y(f) = \cos(kx) \cdot \cos(ly)
  \]
- **Validation**:
  - Compare the PyTorch implementation against the analytical result.

---

### **Test Case 3: \( e^{i(kx + ly)} \)**

- **Input Signal**: \( f(x, y) = e^{i(kx + ly)} \)
- **Expected Analytical Result**:
  \[
  H_x(f) = e^{i(kx + ly - \pi/2)}, \quad H_y(f) = e^{i(kx + ly + \pi/2)}
  \]
- **Validation**:
  - Compare the PyTorch implementation against the analytical result.

---

### **Test Case 4: Gaussian Function**

- **Input Signal**: \( f(x, y) = e^{-(x^2 + y^2)} \)
- **Expected Analytical Result**:
  - Numerical validation using the gradient property of the Gaussian.
- **Validation**:
  - Compare the PyTorch implementation against numerical integration.

---

### **Test Case 5: \( \text{sinc}(x) \cdot \text{sinc}(y) \)**

- **Input Signal**: \( f(x, y) = \frac{\sin(kx)}{kx} \cdot \frac{\sin(ly)}{ly} \)
- **Expected Analytical Result**:
  - Numerical validation using properties of the sinc function.
- **Validation**:
  - Compare the PyTorch implementation against numerical results.

---

### **Test Case 6: Checkerboard Pattern**

- **Input Signal**: Alternating \( \pm 1 \) pattern across \( x \) and \( y \) axes.
- **Expected Analytical Result**:
  - Symmetric Hilbert phase shifts in both axes.
- **Validation**:
  - Compare the PyTorch implementation against analytical phase shifts.

---

## Code Structure

### **`hilbert_transform_torch_2d`**
Custom implementation of the 2D Hilbert Transform in PyTorch. Operates in the frequency domain using FFT and applies multipliers along both axes.

### **Test Cases**
1. **Sinusoidal Functions**:
   - \( \sin(kx) \cdot \cos(ly) \) and \( \cos(kx) \cdot \sin(ly) \): Validate phase shifts along each axis.
2. **Complex Exponentials**:
   - Validate phase shifts in the complex plane for multidimensional signals.
3. **Gaussian and sinc Functions**:
   - Test the transform's behavior on smooth and decaying signals.
4. **Checkerboard Pattern**:
   - Validate phase shifts in symmetric patterns.

### **Validation Metrics**
- **Relative Error**:
  \[
  \text{Relative Error} = \frac{\|\text{Computed Result} - \text{Analytical Result}\|}{\|\text{Analytical Result}\|}
  \]
- **Visualization**:
  - Plots comparing analytical and computed results for each test case.

---

## Relative Errors for 2D Hilbert Transform

| Test Case                                  | Relative Error (PyTorch vs. SciPy) |
|--------------------------------------------|-------------------------------------|
| \( \sin(\omega_x x) \cdot \sin(\omega_y y) \) | \( 1.174550 \times 10^{-7} \)      |
| \( \cos(\omega_x x) \cdot \cos(\omega_y y) \) | \( 1.180263 \times 10^{-7} \)      |
| \( \sin(\omega_x x + \omega_y y) \)         | \( 9.043851 \times 10^{-8} \)      |
| Gaussian Function                           | \( 1.468306 \times 10^{-7} \)      |
| Exponential Function (\( e^{i(\omega_x x + \omega_y y)} \)) | \( 0.999999944 \) **               |
| Sinc Function (\( \text{sinc}(x) \cdot \text{sinc}(y) \)) | \( 1.803591 \times 10^{-7} \)   |

- ** This is an immaginary funciton and was not executable on SciPy package!

### **Plots**
- Visual comparisons confirm that the PyTorch implementation closely matches analytical and SciPy results.

---