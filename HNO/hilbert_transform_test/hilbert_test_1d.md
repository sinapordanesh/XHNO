# Hilbert Transform Kernel Evaluation

This repository contains a custom implementation of the Hilbert Transform kernel in PyTorch, designed as a replacement for Fourier transforms in the context of Fourier Neural Operators (FNO). The kernel is tested and validated against analytical solutions and classical numerical methods to ensure accuracy and reliability.

---

## Overview

The Hilbert Transform is applied to signals in the frequency domain, enabling phase shifting without altering magnitude. This project evaluates the custom kernel's performance using:
- Analytical solutions for known signals.
- Numerical comparison with classical methods (e.g., SciPy).

Key functionalities include:
1. **Hilbert Transform Implementation**: A PyTorch-based implementation for 1D signals.
2. **Analytical Validation**: Comparison with mathematically derived results.
3. **Numerical Validation**: Comparison with the SciPy `hilbert` function.

---

## Test Cases and Validation

We designed several test cases using well-known mathematical properties of the Hilbert Transform. Each test case compares the results of the PyTorch implementation against analytical solutions and SciPy's Hilbert transform.

### **Test Case 1: \( \sin(\omega t) \)**

- **Input Signal**: \( \sin(\omega t) \)
- **Expected Analytical Result**: \( -\cos(\omega t) \)
- **Validation**:
  - Compute Hilbert Transform using the PyTorch implementation.
  - Compare with analytical result and SciPy's Hilbert transform.
- **Relative Error Metrics**:
  - PyTorch vs. Analytical: **Very small (~\( 10^{-8} \))**
  - SciPy vs. Analytical: **Very small (~\( 10^{-15} \))**

---

### **Test Case 2: \( \cos(\omega t) \)**

- **Input Signal**: \( \cos(\omega t) \)
- **Expected Analytical Result**: \( \sin(\omega t) \)
- **Validation**:
  - Compute Hilbert Transform using the PyTorch implementation.
  - Compare with analytical result and SciPy's Hilbert transform.
- **Relative Error Metrics**:
  - PyTorch vs. Analytical: **Very small (~\( 10^{-8} \))**
  - SciPy vs. Analytical: **Very small (~\( 10^{-15} \))**

---

### **Test Case 3: \( e^{i\omega t} \)**

- **Input Signal**: \( e^{i\omega t} \) (complex exponential)
- **Expected Analytical Result**: \( e^{i(\omega t - \pi / 2)} \) for \( \omega > 0 \)
- **Validation**:
  - Compute Hilbert Transform for real and imaginary parts separately using PyTorch.
  - Combine results and compare with analytical solution.
- **Relative Error Metrics**:
  - PyTorch vs. Analytical: **Very small (~\( 10^{-8} \))**

---

## Code Structure

### **`hilbert_transform_torch`**
Custom implementation of the Hilbert Transform in PyTorch. Operates in the frequency domain using FFT and applies the Hilbert multiplier \( -i \cdot \text{sign}(\omega) \).

### **Test Cases**
1. **Sinusoidal Input**:
   - \( \sin(\omega t) \): Validate phase shift to \( -\cos(\omega t) \).
   - \( \cos(\omega t) \): Validate phase shift to \( \sin(\omega t) \).
2. **Complex Exponential**:
   - Validate phase shift in the complex plane.

### **Validation Metrics**
- **Relative Error**: 
  \[
  \text{Relative Error} = \frac{\|\text{Computed Result} - \text{Analytical Result}\|}{\|\text{Analytical Result}\|}
  \]
- **Visualization**:
  - Analytical and computed results are plotted for visual inspection.

---

## Relative Errors for 1D Hilbert Transform

| Test Case            | Relative Error (PyTorch vs. Analytical) | Relative Error (SciPy vs. Analytical) |
|-----------------------|-----------------------------------------|---------------------------------------|
| \( \sin(\omega t) \)  | \( 7.426054 \times 10^{-8} \)           | \( 1.697598 \times 10^{-15} \)       |
| \( \cos(\omega t) \)  | \( 7.482067 \times 10^{-8} \)           | \( 1.684605 \times 10^{-15} \)       |
| \( e^{i\omega t} \)   | \( 7.454113 \times 10^{-8} \)           | N/A                                   |
